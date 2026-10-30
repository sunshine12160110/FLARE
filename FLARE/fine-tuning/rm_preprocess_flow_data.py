#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Multi-GPU Flow-level Data Preprocessing Script (Enhanced Temporal Features)
Fixed multi-threading result collection issues
"""
import os
import sys
import json
import torch
import random
import argparse
import numpy as np
from collections import defaultdict
import time
from threading import Thread, Lock
import queue
import signal
import gc
import hashlib


_packet_encoder_init_lock = Lock()

# Recursively convert numpy types to Python native types for JSON serialization
def to_py(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, list):
        return [to_py(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    else:
        return obj


def validate_encoded_flow_dict(flow):
    """Return (ok, reason) for a single encoded flow dict before writing JSONL."""
    if not isinstance(flow, dict) or 'category' not in flow or 'src' not in flow:
        return False, "not a dict or missing keys"
    required_fields = ['src', 'time_intervals', 'packet_length_directions', 'category', 'packet_count']
    missing_fields = [f for f in required_fields if f not in flow]
    if missing_fields:
        return False, f"missing fields: {missing_fields}"
    if not isinstance(flow['packet_length_directions'], list):
        return False, "packet_length_directions not a list"
    if len(flow['packet_length_directions']) > 0:
        if not all(isinstance(x, (int, float)) for x in flow['packet_length_directions']):
            return False, "packet_length_directions non-numeric"
    return True, ""


def parse_tsv_packet_line(line, line_id):
    """Parse one TSV line into packet_data dict or None if skip/invalid."""
    line = line.strip()
    if not line:
        return None
    if line.startswith('category') or line.startswith('Category'):
        return None
    line_parts = line.split('\t')
    if len(line_parts) < 5:
        return None
    try:
        return {
            'category': int(line_parts[0]),
            'flow_order': int(line_parts[1]),
            'relative_time': float(line_parts[2]),
            'packet_length_direction': line_parts[3],
            'text_a': line_parts[4],
        }
    except (ValueError, IndexError) as e:
        print(f"Skipping invalid line {line_id + 1}: {line[:100]}... (error: {e})")
        return None


def iter_contiguous_flows_from_tsv(input_file, min_flow_length):
    """
    Stream TSV: assume packets of the same logical flow appear on consecutive lines
    (same category + flow_order). When flow_key changes, the previous buffer is one finished flow.
    Yields (flow_key, packets) for flows with len(packets) >= min_flow_length.
    """
    current_key = None
    current_packets = []
    header_skipped_log = False

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            packet_data = parse_tsv_packet_line(line, line_id)
            if packet_data is None:
                if line.strip() and (line.strip().lower().startswith('category')) and not header_skipped_log:
                    print(f"Skipping header line: {line.strip()[:120]}")
                    header_skipped_log = True
                continue
            flow_key = f"{packet_data['category']}_{packet_data['flow_order']}"
            if current_key is None:
                current_key = flow_key
                current_packets = [packet_data]
                continue
            if flow_key == current_key:
                current_packets.append(packet_data)
            else:
                if len(current_packets) >= min_flow_length:
                    yield current_key, current_packets
                current_key = flow_key
                current_packets = [packet_data]

    if current_key is not None and len(current_packets) >= min_flow_length:
        yield current_key, current_packets


def reservoir_maybe_keep_flow(category, flow_key, packets, reservoirs, seen_count, k, rng):
    """
    Reservoir sampling: keep at most k flows per category in reservoirs[category].
    Memory bounded by O(num_categories * k) raw flows.
    """
    if k <= 0:
        return
    seen_count[category] += 1
    n = seen_count[category]
    bucket = reservoirs[category]
    item = (flow_key, packets)
    if len(bucket) < k:
        bucket.append(item)
        return
    j = rng.randrange(n)
    if j < k:
        bucket[j] = item


def iter_reservoir_flows_flat(reservoirs):
    """Yield (flow_key, packets) from all category buckets (order undefined)."""
    for _cat, bucket in reservoirs.items():
        for item in bucket:
            yield item


def result_writer_worker(result_queue, output_file, stats, stats_lock, progress_lock):
    """Read (gpu_id, results) from queue until None sentinel; write each valid flow as one JSON line."""
    written = 0
    invalid = 0
    category_counts = defaultdict(int)
    batches_done = 0
    err = None
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            while True:
                item = result_queue.get()
                if item is None:
                    break
                gpu_id, results = item
                batches_done += 1
                for flow in results:
                    ok, _reason = validate_encoded_flow_dict(flow)
                    if ok:
                        f.write(json.dumps(to_py(flow), ensure_ascii=False) + '\n')
                        written += 1
                        category_counts[flow['category']] += 1
                    else:
                        invalid += 1
                if batches_done % 20 == 0:
                    with progress_lock:
                        print(
                            f"Writer: {batches_done} batches, {written} flows written, "
                            f"{invalid} invalid skipped"
                        )
    except Exception as e:
        err = e
        print(f"Result writer error: {e}")
    with stats_lock:
        stats['written'] = written
        stats['invalid'] = invalid
        stats['category_counts'] = dict(category_counts)
        stats['writer_error'] = err


def stream_fallback_single_gpu_write(args, gpu_id, input_file, output_file, min_flow_length, flow_batch_size, max_k, seed):
    """If multi-GPU pipeline wrote nothing, stream TSV on one GPU and write incrementally."""
    rng = random.Random(seed) if max_k > 0 else None
    reservoirs = defaultdict(list)
    seen_count = defaultdict(int)
    encoder = StableMultiGPUEncoder(args, gpu_id)
    current_batch = []
    written = 0
    invalid = 0
    category_counts = defaultdict(int)

    file_started = False

    def run_batch(batch):
        nonlocal written, invalid, file_started
        if not batch:
            return
        results = encode_flow_batch_stable(batch, encoder, args, gpu_id)
        mode = 'w' if not file_started else 'a'
        file_started = True
        with open(output_file, mode, encoding='utf-8') as f:
            for flow in results:
                ok, _ = validate_encoded_flow_dict(flow)
                if ok:
                    f.write(json.dumps(to_py(flow), ensure_ascii=False) + '\n')
                    written += 1
                    category_counts[flow['category']] += 1
                else:
                    invalid += 1

    if max_k > 0:
        for flow_key, packets in iter_contiguous_flows_from_tsv(input_file, min_flow_length):
            cat = packets[0]['category']
            reservoir_maybe_keep_flow(cat, flow_key, packets, reservoirs, seen_count, max_k, rng)
        flows_iter = iter_reservoir_flows_flat(reservoirs)
    else:
        flows_iter = iter_contiguous_flows_from_tsv(input_file, min_flow_length)

    for flow_key, packets in flows_iter:
        current_batch.append((flow_key, packets))
        if len(current_batch) >= flow_batch_size:
            run_batch(current_batch)
            current_batch.clear()
    run_batch(current_batch)

    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Fallback single-GPU stream: wrote {written} flows, invalid={invalid}")
    return written, invalid, category_counts


# Add UER path
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)
uer_path = "/home/sunshine/Projects/ET-BERT-main"
sys.path.append(uer_path)

from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.layers import *
from uer.encoders import *
from uer.utils.config import load_hyperparam
from uer.opts import model_opts, optimization_opts, training_opts

class StableMultiGPUEncoder:
    """Stable multi-GPU packet-level encoder"""
    def __init__(self, args, gpu_id):
        self.args = args
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(self.device)
        # Model construction consumes PyTorch's process-global RNG. Workers are
        # threads, so serialize construction and restore the RNG afterwards.
        # This guarantees that every GPU uses exactly the same random packet
        # encoder instead of silently producing GPU-dependent representations.
        with _packet_encoder_init_lock:
            if args.random_init_packet_encoder:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(args.packet_encoder_seed)
                    self.embedding = str2embedding[args.embedding](
                        args, len(args.tokenizer.vocab)
                    )
                    self.encoder = str2encoder[args.encoder](args)
            else:
                self.embedding = str2embedding[args.embedding](
                    args, len(args.tokenizer.vocab)
                )
                self.encoder = str2encoder[args.encoder](args)
        if args.packet_pretrained_model_path:
            print(f"GPU {gpu_id}: Loading packet-level pretrained model")
            model_state = torch.load(args.packet_pretrained_model_path, map_location='cpu')
            full_state_dict = {}
            embedding_keys = [k for k in model_state.keys() if k.startswith('embedding.')]
            for key in embedding_keys:
                new_key = key.replace('embedding.', '')
                full_state_dict[new_key] = model_state[key]
            encoder_keys = [k for k in model_state.keys() if k.startswith('encoder.')]
            for key in encoder_keys:
                new_key = key.replace('encoder.', '')
                full_state_dict[new_key] = model_state[key]
            try:
                self.embedding.load_state_dict(full_state_dict, strict=False)
                print(f"GPU {gpu_id}: Successfully loaded embedding state")
            except Exception as e:
                print(f"GPU {gpu_id}: Error loading embedding state: {e}")
            try:
                self.encoder.load_state_dict(full_state_dict, strict=False)
                print(f"GPU {gpu_id}: Successfully loaded encoder state")
            except Exception as e:
                print(f"GPU {gpu_id}: Error loading encoder state: {e}")
        else:
            first_parameter = next(self.embedding.parameters()).detach().view(-1)[:1024]
            fingerprint = hashlib.sha256(first_parameter.numpy().tobytes()).hexdigest()[:16]
            print(
                f"GPU {gpu_id}: Random packet encoder initialized "
                f"(seed={args.packet_encoder_seed}, fingerprint={fingerprint})"
            )
        self.embedding.to(self.device)
        self.encoder.to(self.device)
        self.embedding.eval()
        self.encoder.eval()
    
    def encode_packet_batch(self, packet_texts):
        """Encode a batch of packet texts to embeddings"""
        if not packet_texts:
            return []
        batch_src = []
        batch_seg = []
        for packet_text in packet_texts:
            tokens = self.args.tokenizer.tokenize(packet_text)
            src = self.args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + tokens)
            seg = [1] * len(src)
            if len(src) > self.args.seq_length:
                src = src[:self.args.seq_length]
                seg = seg[:self.args.seq_length]
            while len(src) < self.args.seq_length:
                src.append(0)
                seg.append(0)
            batch_src.append(src)
            batch_seg.append(seg)
        src = torch.LongTensor(batch_src).to(self.device)
        seg = torch.LongTensor(batch_seg).to(self.device)
        with torch.no_grad():
            emb = self.embedding(src, seg)
            output = self.encoder(emb, seg)
            packet_representations = output[:, 0, :].cpu().numpy()
        return packet_representations

def encode_flow_batch_stable(flow_batch, encoder, args, gpu_id):
    """Stable flow batch encoding with temporal information"""
    results = []
    total_flows = len(flow_batch)
    for i, flow_data in enumerate(flow_batch):
        flow_key, packets = flow_data
        if i % 100 == 0:
            print(f"GPU {gpu_id}: Processing progress {i}/{total_flows} ({i/total_flows*100:.1f}%)")
        if len(packets) < args.min_flow_length:
            continue
        if len(packets) > args.max_flow_length:
            packets = packets[:args.max_flow_length]
        packet_texts = [packet['text_a'] for packet in packets]
        try:
            packet_representations = encoder.encode_packet_batch(packet_texts)
        except Exception as e:
            print(f"GPU {gpu_id}: Error in batch packet encoding: {e}")
            continue
        if len(packet_representations) < args.min_flow_length:
            continue
        rel_times = [p['relative_time'] for p in packets[:len(packet_representations)]]
        # Convert packet_length_direction to float (consistent with pretraining format)
        # Format: positive = outgoing, negative = incoming
        # Input may be: numeric value, string like "1234" or "-1234", or "1234/1" format
        pkt_length_dirs = []
        for p in packets[:len(packet_representations)]:
            pld = p.get('packet_length_direction', '0')
            try:
                if isinstance(pld, str):
                    pld = pld.strip()  # Remove whitespace
                    if '/' in pld:
                        # Handle format: "1234/1" (length/direction)
                        parts = pld.split('/')
                        length = abs(float(parts[0]))  # Always use absolute value for length
                        direction = int(parts[1]) if len(parts) > 1 else 1
                        # Convert to signed value: positive = outgoing, negative = incoming
                        pld_val = length if direction > 0 else -length
                    else:
                        # Handle numeric string: "1234" or "-1234" (already signed)
                        # If negative, it's incoming; if positive, it's outgoing
                        pld_val = float(pld)
                elif isinstance(pld, (int, float)):
                    # Already numeric, use as is (preserve sign)
                    pld_val = float(pld)
                else:
                    # Unknown type, default to 0
                    pld_val = 0.0
            except (ValueError, AttributeError, TypeError) as e:
                # If conversion fails, use 0.0 as default
                pld_val = 0.0
            pkt_length_dirs.append(pld_val)
        
        # Padding
        pad_len = args.max_flow_length - len(packet_representations)
        if pad_len > 0:
            dim = len(packet_representations[0])
            packet_representations = list(packet_representations) + [[0.0] * dim for _ in range(pad_len)]
            rel_times = rel_times + [0.0] * pad_len
            pkt_length_dirs = pkt_length_dirs + [0.0] * pad_len
        else:
            packet_representations = list(packet_representations)[:args.max_flow_length]
            rel_times = rel_times[:args.max_flow_length]
            pkt_length_dirs = pkt_length_dirs[:args.max_flow_length]
        # Ensure numpy arrays are converted to Python native lists
        packet_representations = [list(vec) if hasattr(vec, 'tolist') else vec for vec in packet_representations]
        flow_data = {
            'category': packets[0]['category'],
            'flow_order': packets[0]['flow_order'],
            'packet_count': len([x for x in packet_representations if sum(x) != 0]),
            'src': packet_representations,
            'time_intervals': rel_times,
            'packet_length_directions': pkt_length_dirs  # 数值格式：正数=向外，负数=向内（与预训练一致）
        }
        results.append(flow_data)
    print(f"GPU {gpu_id}: Processing completed, {len(results)} valid flows")
    return results

def gpu_worker_stable(gpu_id, flow_queue, result_queue, args, progress_lock, gpu_status):
    """Stable GPU worker thread for processing flow batches"""
    try:
        encoder = StableMultiGPUEncoder(args, gpu_id)
        print(f"GPU {gpu_id}: Encoder initialization completed")
        with progress_lock:
            gpu_status[gpu_id] = "ready"
        
        processed_batches = 0
        total_processed_flows = 0
        
        while True:
            try:
                batch = flow_queue.get(timeout=10)  # Increased timeout
                if batch is None:
                    break
                
                results = encode_flow_batch_stable(batch, encoder, args, gpu_id)
                processed_batches += 1
                total_processed_flows += len(results)
                
                # Ensure results are not empty before putting in queue
                if results:
                    with progress_lock:
                        result_queue.put((gpu_id, results))
                        print(f"GPU {gpu_id}: Batch {processed_batches} results put in queue, {len(results)} flows")
                else:
                    # Even if results are empty, put in queue to ensure main thread can count correctly
                    with progress_lock:
                        result_queue.put((gpu_id, []))
                        print(f"GPU {gpu_id}: Batch {processed_batches} results empty, put in queue")
                        
            except queue.Empty:
                print(f"GPU {gpu_id}: Queue timeout, exiting")
                break
            except Exception as e:
                print(f"GPU {gpu_id} processing error: {e}")
                processed_batches += 1
                with progress_lock:
                    result_queue.put((gpu_id, []))
        
        print(f"GPU {gpu_id}: Work completed, processed {processed_batches} batches, {total_processed_flows} flows")
        with progress_lock:
            gpu_status[gpu_id] = "completed"
    except Exception as e:
        print(f"GPU {gpu_id} initialization error: {e}")
        with progress_lock:
            result_queue.put((gpu_id, []))
            gpu_status[gpu_id] = "error"

def progress_monitor_fixed(result_queue, total_batches, start_time, gpu_status, gpu_ids):
    """Fixed progress monitor for tracking processing status"""
    completed_batches = 0
    processed_flows = []
    last_progress_time = time.time()
    
    print(f"Starting to monitor {total_batches} batch processing progress...")
    
    while completed_batches < total_batches:
        try:
            # Increased timeout to 60 seconds
            gpu_id, results = result_queue.get(timeout=60)
            processed_flows.extend(results)
            completed_batches += 1
            elapsed_time = time.time() - start_time
            current_time = time.time()
            
            print(f"Progress: {completed_batches}/{total_batches} batches completed "
                  f"({completed_batches/total_batches*100:.1f}%), "
                  f"processed {len(processed_flows)} flows, "
                  f"elapsed {elapsed_time:.1f} seconds")
            
            # Check GPU status every 30 seconds
            if current_time - last_progress_time > 30:
                print("GPU status check:")
                for gpu_id in gpu_ids:
                    status = gpu_status.get(gpu_id, "unknown")
                    print(f"  GPU {gpu_id}: {status}")
                last_progress_time = current_time
                
        except queue.Empty:
            print(f"Progress monitor: Waiting for results timeout, completed {completed_batches}/{total_batches} batches")
            
            # Check GPU status
            print("GPU status check:")
            for gpu_id in gpu_ids:
                status = gpu_status.get(gpu_id, "unknown")
                print(f"  GPU {gpu_id}: {status}")
            
            # If all GPUs are completed but there are still incomplete batches, there might be a queue issue
            all_completed = all(status in ["completed", "error"] for status in gpu_status.values())
            if all_completed and completed_batches < total_batches:
                print(f"Warning: All GPUs completed, but {total_batches - completed_batches} batches still incomplete")
                print("Attempting to get remaining results from queue...")
                
                # Try to get remaining results
                remaining_attempts = 10
                while remaining_attempts > 0 and completed_batches < total_batches:
                    try:
                        gpu_id, results = result_queue.get_nowait()
                        processed_flows.extend(results)
                        completed_batches += 1
                        print(f"Got remaining results: GPU {gpu_id}, {len(results)} flows")
                    except queue.Empty:
                        break
                    remaining_attempts -= 1
                
                if completed_batches < total_batches:
                    print(f"Still {total_batches - completed_batches} batches incomplete, data loss possible")
            
            break
    
    print(f"Progress monitoring completed, collected {len(processed_flows)} flows total")
    return processed_flows

def preprocess_flow_data_multi_gpu_with_time_fixed(args):
    """Stream TSV (contiguous packets per flow), multi-GPU encode, write JSONL incrementally."""
    print("=== Starting Fixed Multi-GPU Packet-level Encoding Preprocessing (with Temporal Features) ===")
    print(
        "TSV 流式读取：默认同一 flow 的包在文件中连续（相同 category 与 flow_order 的相邻行）。"
        "若不同 flow 的行交错，请先按 category、flow_order、relative_time 排序 TSV。"
    )

    gpu_ids = get_available_gpus(args)
    if not gpu_ids:
        print("Error: No available GPUs")
        return

    available_gpus = len(gpu_ids)
    print(f"Will use {available_gpus} GPUs: {gpu_ids}")

    flow_batch_size = max(1, int(getattr(args, "flow_batch_size", 32)))
    max_k = int(getattr(args, "max_samples_per_category", 0) or 0)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    reservoirs = defaultdict(list)
    seen_count = defaultdict(int)
    category_seen_raw = defaultdict(int)

    def emit_flows_for_encode():
        """Yield (flow_key, packets); reservoir mode fills after one full TSV pass."""
        if max_k > 0:
            print(f"Reservoir sampling: at most {max_k} raw flows per category (single pass over TSV).")
            for flow_key, packets in iter_contiguous_flows_from_tsv(args.input_file, args.min_flow_length):
                cat = packets[0]['category']
                category_seen_raw[cat] += 1
                reservoir_maybe_keep_flow(cat, flow_key, packets, reservoirs, seen_count, max_k, rng)
            for cat, bucket in reservoirs.items():
                print(f"  Category {cat}: reservoir kept {len(bucket)} flows (seen {seen_count[cat]} total)")
            for item in iter_reservoir_flows_flat(reservoirs):
                yield item
        else:
            for flow_key, packets in iter_contiguous_flows_from_tsv(args.input_file, args.min_flow_length):
                cat = packets[0]['category']
                category_seen_raw[cat] += 1
                yield flow_key, packets

    flow_queue = queue.Queue(maxsize=max(32, flow_batch_size * available_gpus * 4))
    result_queue = queue.Queue()
    progress_lock = Lock()
    stats_lock = Lock()
    gpu_status = {}
    stats = {'written': 0, 'invalid': 0, 'category_counts': {}, 'writer_error': None}

    writer_thread = Thread(
        target=result_writer_worker,
        args=(result_queue, args.output_file, stats, stats_lock, progress_lock),
    )
    writer_thread.start()

    threads = []
    for gpu_id in gpu_ids:
        thread = Thread(
            target=gpu_worker_stable,
            args=(gpu_id, flow_queue, result_queue, args, progress_lock, gpu_status),
        )
        threads.append(thread)
        thread.start()
        print(f"Started GPU {gpu_id} worker thread")

    start_time = time.time()
    current_batch = []
    total_batches = 0
    flows_queued = 0

    def flush_batch():
        nonlocal total_batches, current_batch
        if not current_batch:
            return
        flow_queue.put(current_batch[:])
        total_batches += 1
        current_batch = []

    try:
        for flow_key, packets in emit_flows_for_encode():
            flows_queued += 1
            current_batch.append((flow_key, packets))
            if len(current_batch) >= flow_batch_size:
                flush_batch()
        flush_batch()

        print(f"Stream scan: {flows_queued} flows queued for encoding in {total_batches} batches.")
        if category_seen_raw:
            print("Raw valid flows seen per category (before reservoir / encoding):")
            for cat in sorted(category_seen_raw.keys()):
                print(f"  Category {cat}: {category_seen_raw[cat]} flows")

        if total_batches == 0:
            print("Error: No flows meeting minimum length requirement (or empty input).")
            for _ in range(available_gpus):
                flow_queue.put(None)
            for thread in threads:
                thread.join()
            result_queue.put(None)
            writer_thread.join()
            return

        for _ in range(available_gpus):
            flow_queue.put(None)

        print("Waiting for all GPU worker threads to complete...")
        for i, thread in enumerate(threads):
            thread.join()
            print(f"GPU {gpu_ids[i]} worker thread completed")

        result_queue.put(None)
        writer_thread.join()

    except Exception as e:
        print(f"Preprocessing error: {e}")
        for _ in range(available_gpus):
            try:
                flow_queue.put_nowait(None)
            except queue.Full:
                pass
        for thread in threads:
            thread.join(timeout=5)
        try:
            result_queue.put_nowait(None)
        except Exception:
            pass
        writer_thread.join(timeout=30)
        raise

    end_time = time.time()

    print("Final GPU status check:")
    with progress_lock:
        for gid in gpu_ids:
            status = gpu_status.get(gid, "unknown")
            print(f"  GPU {gid}: {status}")

    with stats_lock:
        written = stats['written']
        invalid = stats['invalid']
        category_counts = stats['category_counts']
        werr = stats['writer_error']

    if werr is not None:
        print(f"Writer finished with error: {werr}")

    if written == 0 and total_batches > 0:
        print("No flows written via multi-GPU pipeline; trying single-GPU streaming fallback...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            w, inv_fb, cc = stream_fallback_single_gpu_write(
                args,
                gpu_ids[0],
                args.input_file,
                args.output_file,
                args.min_flow_length,
                flow_batch_size,
                max_k,
                args.seed,
            )
            with stats_lock:
                stats['written'] = w
                stats['invalid'] = inv_fb
                stats['category_counts'] = dict(cc)
        except Exception as e:
            print(f"Fallback strategy failed: {e}")

    with stats_lock:
        written = stats.get('written', written)
        invalid = stats.get('invalid', invalid)
        category_counts = stats.get('category_counts', category_counts)

    if written == 0:
        print("Error: No valid flows written to output.")
        return

    print(f"Data saved to: {args.output_file} ({written} flows, {invalid} invalid skipped in writer)")
    print("\nFinal category distribution (written):")
    for category, count in sorted(category_counts.items()):
        print(f"Category {category}: {count} flows")

    print("\n=== Processing Summary Report ===")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    print(f"Valid flow count (written): {written}")
    print(f"Category count: {len(category_counts)}")
    if category_counts:
        print(f"Average flows per category: {written / len(category_counts):.1f}")
    print(f"GPU usage: {available_gpus} GPUs")

    print("\nGPU Status Summary:")
    with progress_lock:
        completed_gpus = sum(1 for status in gpu_status.values() if status == "completed")
        error_gpus = sum(1 for status in gpu_status.values() if status == "error")
        print(f"Successfully completed: {completed_gpus} GPUs")
        print(f"Processing errors: {error_gpus} GPUs")
        print(f"Total GPUs: {available_gpus}")

    print("=== Processing Completed ===")

def get_available_gpus(args):
    """
    Parse and validate specified GPU IDs
    """
    total_gpus = torch.cuda.device_count()
    print(f"System detected {total_gpus} GPUs")
    
    if not total_gpus:
        print("Error: No available GPUs detected")
        return []
    
    if args.gpu_ids is None:
        # Use all GPUs
        gpu_ids = list(range(total_gpus))
        print(f"No GPU specified, will use all {total_gpus} GPUs: {gpu_ids}")
        return gpu_ids
    
    try:
        # Parse specified GPU IDs
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        
        # Validate GPU IDs
        valid_gpus = []
        invalid_gpus = []
        
        for gpu_id in gpu_ids:
            if 0 <= gpu_id < total_gpus:
                valid_gpus.append(gpu_id)
            else:
                invalid_gpus.append(gpu_id)
        
        if invalid_gpus:
            print(f"Warning: Following GPU IDs are invalid: {invalid_gpus} (valid range: 0-{total_gpus-1})")
        
        if not valid_gpus:
            print(f"Error: No valid GPU IDs, will use all GPUs: {list(range(total_gpus))}")
            return list(range(total_gpus))
        
        print(f"Will use specified GPUs: {valid_gpus}")
        return valid_gpus
        
    except ValueError as e:
        print(f"Error: Invalid GPU ID format '{args.gpu_ids}', will use all GPUs")
        return list(range(total_gpus))

def split_encoded_dataset(input_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, output_dir=None, low_memory=False):
    """Split encoded dataset into train/validation/test sets (stratified by category).
    input_file: JSONL file, one encoded flow per line (dict with category, src, time_intervals, etc.).
    output_dir: if set, write *_train.json, *_val.json, *_test.json under this dir; else same dir as input_file.
    low_memory: if True, use two-pass streaming to avoid loading full dataset into RAM (slower but bounded memory).
    """
    random.seed(seed)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    out_dir = output_dir if output_dir else os.path.dirname(os.path.abspath(input_file))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    prefix = os.path.join(out_dir, base_name)

    if low_memory:
        _split_encoded_dataset_streaming(
            input_file, train_ratio, val_ratio, test_ratio, seed, prefix,
        )
        return

    # In-memory path: load all, then split (original behavior)
    flows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            flows.append(json.loads(line))
    category_flows = defaultdict(list)
    for flow in flows:
        category_flows[flow['category']].append(flow)
    train_flows = []
    val_flows = []
    test_flows = []
    for category, category_data in category_flows.items():
        random.shuffle(category_data)
        n = len(category_data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        train_flows.extend(category_data[:train_end])
        val_flows.extend(category_data[train_end:val_end])
        test_flows.extend(category_data[val_end:])
    with open(f"{prefix}_train.json", 'w', encoding='utf-8') as f:
        for flow in train_flows:
            f.write(json.dumps(to_py(flow), ensure_ascii=False) + '\n')
    with open(f"{prefix}_val.json", 'w', encoding='utf-8') as f:
        for flow in val_flows:
            f.write(json.dumps(to_py(flow), ensure_ascii=False) + '\n')
    with open(f"{prefix}_test.json", 'w', encoding='utf-8') as f:
        for flow in test_flows:
            f.write(json.dumps(to_py(flow), ensure_ascii=False) + '\n')
    print(f"Dataset splitting completed:")
    print(f"Training set: {len(train_flows)} flows")
    print(f"Validation set: {len(val_flows)} flows")
    print(f"Test set: {len(test_flows)} flows")


def _split_encoded_dataset_streaming(input_file, train_ratio, val_ratio, test_ratio, seed, prefix):
    """Two-pass streaming split: O(1) flow data in memory, stratified by category.
    Pass 1: read each line only to get category, build per-category indices and assign train/val/test.
    Pass 2: re-open file, for each line decide split from precomputed assignment and write to corresponding file.
    """
    random.seed(seed)
    line_categories = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                line_categories.append(obj.get('category'))
            except (json.JSONDecodeError, TypeError):
                line_categories.append(None)
    n_total = len(line_categories)
    if n_total == 0:
        print("No valid lines in input file.")
        return
    category_indices = defaultdict(list)
    for i, cat in enumerate(line_categories):
        if cat is not None:
            category_indices[cat].append(i)
    assignment = [None] * n_total
    for category, indices in category_indices.items():
        random.shuffle(indices)
        n = len(indices)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        for idx, i in enumerate(indices):
            assignment[i] = 0 if idx < train_end else (1 if idx < val_end else 2)
    train_count = val_count = test_count = 0
    out_files = [
        open(f"{prefix}_train.json", 'w', encoding='utf-8'),
        open(f"{prefix}_val.json", 'w', encoding='utf-8'),
        open(f"{prefix}_test.json", 'w', encoding='utf-8'),
    ]
    try:
        line_idx = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line_idx >= n_total or assignment[line_idx] is None:
                    line_idx += 1
                    continue
                w = assignment[line_idx]
                flow = json.loads(line)
                out_files[w].write(json.dumps(to_py(flow), ensure_ascii=False) + '\n')
                if w == 0:
                    train_count += 1
                elif w == 1:
                    val_count += 1
                else:
                    test_count += 1
                line_idx += 1
        print(f"Dataset splitting completed (streaming):")
        print(f"Training set: {train_count} flows")
        print(f"Validation set: {val_count} flows")
        print(f"Test set: {test_count} flows")
    finally:
        for fp in out_files:
            fp.close()


def main():
    """Main function for multi-GPU flow-level data preprocessing with temporal features"""
    parser = argparse.ArgumentParser(description="Fixed Multi-GPU Flow-level Data Preprocessing (with Temporal Features)")
    parser.add_argument("--input_file", type=str, required=True, help="Input file path (TSV for encoding; JSONL for --split_only)")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path (not needed for --split_only)")
    parser.add_argument("--packet_pretrained_model_path", type=str, default=None, help="Packet-level pretrained model path (not needed for --split_only)")
    parser.add_argument(
        "--random_init_packet_encoder",
        action="store_true",
        help="Do not load packet-level pretrained weights; initialize one deterministic encoder shared across GPUs",
    )
    parser.add_argument(
        "--packet_encoder_seed",
        type=int,
        default=42,
        help="Initialization seed used with --random_init_packet_encoder",
    )
    parser.add_argument("--config_path", type=str, default="/home/sunshine/Projects/ET-BERT-main/models/bert/base_config.json", help="Model configuration file path")
    parser.add_argument("--max_flow_length", type=int, default=50, help="Maximum flow length (packet count)")
    parser.add_argument("--min_flow_length", type=int, default=3, help="Minimum flow length (packet count)")
    parser.add_argument("--gpu_ids", type=str, default=None,
                       help="Specify GPU IDs to use, e.g. '0,1,2' or '0', default uses all available GPUs")
    model_opts(parser)
    optimization_opts(parser)
    training_opts(parser)
    parser.add_argument("--vocab_path", default=None, type=str, help="Path of the vocabulary file.")
    parser.add_argument("--spm_model_path", default=None, type=str, help="Path of the sentence piece model.")
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert", help="Specify the tokenizer.")
    parser.add_argument("--split", action='store_true', help="After encoding, split output into train/val/test")
    parser.add_argument("--split_only", action='store_true',
                        help="Only split existing encoded JSONL into train/val/test; input_file must be JSONL (one flow per line with category, src, etc.)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for split files (train/val/test); default: same as input file directory")
    parser.add_argument("--low_memory", action="store_true",
                        help="Use streaming split (two-pass) to avoid loading full JSONL into RAM; use when dataset is large")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument("--max_samples_per_category", type=int, default=0, help="Maximum flows to select per category, 0 means no limit")
    parser.add_argument(
        "--flow_batch_size",
        type=int,
        default=32,
        help="Number of flows per GPU task batch (streaming mode; smaller uses less peak RAM in queues)",
    )
    args = parser.parse_args()

    # Set default seed to 42
    if not hasattr(args, 'seed') or args.seed == 7:
        args.seed = 42

    # 仅分割模式：已有编码好的 JSONL，只做 train/val/test 划分，无需 GPU 与预训练模型
    if args.split_only:
        if not os.path.exists(args.input_file):
            print(f"Error: Input file does not exist: {args.input_file}")
            sys.exit(1)
        print("=== Split only mode: splitting encoded JSONL into train/val/test ===")
        try:
            split_encoded_dataset(
                args.input_file,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
                output_dir=args.output_dir,
                low_memory=getattr(args, "low_memory", False),
            )
        except Exception as e:
            print(f"Dataset splitting failed: {e}")
            sys.exit(1)
        return

    # Full encoding requires an output plus exactly one packet-encoder source.
    if not args.output_file:
        print("Error: --output_file is required when not using --split_only")
        sys.exit(1)
    if bool(args.packet_pretrained_model_path) == bool(args.random_init_packet_encoder):
        print(
            "Error: specify exactly one of --packet_pretrained_model_path or "
            "--random_init_packet_encoder"
        )
        sys.exit(1)

    output_parent = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(output_parent, exist_ok=True)

    if args.config_path:
        load_hyperparam(args)
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    preprocess_flow_data_multi_gpu_with_time_fixed(args)

    if args.split:
        if os.path.exists(args.output_file) and os.path.getsize(args.output_file) > 0:
            print(f"Starting dataset splitting: {args.output_file}")
            try:
                split_encoded_dataset(
                    args.output_file,
                    args.train_ratio,
                    args.val_ratio,
                    args.test_ratio,
                    args.seed,
                    output_dir=args.output_dir,
                    low_memory=getattr(args, "low_memory", False),
                )
            except Exception as e:
                print(f"Dataset splitting failed: {e}")
        else:
            print(f"Warning: Output file {args.output_file} does not exist or is empty, skipping dataset splitting")

if __name__ == "__main__":
    main()
