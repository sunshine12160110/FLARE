#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow-level BERT Fine-tuning (Random Initialization, Gated PLD + Content Fusion)

- 包长/方向序列：使用 rm_128_pld_corpus_vocab.txt 词表做 Embedding（与 BERT-ps dilength 一致）
- PLD 先映射到 hidden_size（默认 768），再与 768 维内容在包粒度做门控融合
- 不使用时间间隔特征
- --feature_mode 可为 content / pld / content_pld，便于针对不同数据集选择特征组合
- 全参数随机初始化，端到端训练（不加载预训练权重）
"""

import os
import sys
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import fileinput
from uer.encoders import *
from uer.layers.layer_norm import LayerNorm
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts
import tqdm
import pickle

#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

# ===================== PLD Vocab & Embedding =====================
PLD_MAX_ABS_LENGTH = 1500


def load_pld_vocab(vocab_path):
    """加载包长/方向词表（与 BERT-ps dilength 词表格式一致）。"""
    vocab = Vocab()
    vocab.load(vocab_path, is_quiet=False)
    return vocab


def build_pld_id_lookup(vocab, device):
    """
    构建长度 -> token id 查表（+l / -l），与 dataset_generation._build_dilength_token_string 一致。
    0 长度映射为 [UNK]；查表外长度裁剪到 PLD_MAX_ABS_LENGTH。
    """
    pad_id = vocab.w2i.get("[PAD]", PAD_ID)
    unk_id = vocab.w2i.get("[UNK]", 3)

    lookup_pos = torch.full((PLD_MAX_ABS_LENGTH + 1,), unk_id, dtype=torch.long, device=device)
    lookup_neg = torch.full((PLD_MAX_ABS_LENGTH + 1,), unk_id, dtype=torch.long, device=device)
    for length in range(PLD_MAX_ABS_LENGTH + 1):
        pos_token = "+{}".format(length)
        neg_token = "-{}".format(length)
        if pos_token in vocab.w2i:
            lookup_pos[length] = vocab.w2i[pos_token]
        if neg_token in vocab.w2i:
            lookup_neg[length] = vocab.w2i[neg_token]
    return pad_id, unk_id, lookup_pos, lookup_neg


def pld_values_to_token_ids(packet_length_directions, pad_id, unk_id, lookup_pos, lookup_neg, flow_lengths=None):
    """
    将 float 包长+方向序列转为词表 id。
    packet_length_directions: [B, L]，正=出向，负=入向，0=UNK。
    flow_lengths 非空时，超出有效包数的位置强制为 PAD。
    """
    pld = packet_length_directions
    batch_size, seq_len = pld.shape
    device = pld.device

    ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device=device)
    abs_len = pld.abs().long().clamp(0, PLD_MAX_ABS_LENGTH)

    pos_mask = pld > 0
    neg_mask = pld < 0
    zero_mask = pld == 0

    if pos_mask.any():
        ids[pos_mask] = lookup_pos[abs_len[pos_mask]]
    if neg_mask.any():
        ids[neg_mask] = lookup_neg[abs_len[neg_mask]]
    if zero_mask.any():
        ids[zero_mask] = unk_id

    if flow_lengths is not None:
        for i, length in enumerate(flow_lengths):
            actual = min(int(length.item()), seq_len)
            if actual < seq_len:
                ids[i, actual:] = pad_id

    return ids


class GatedPldContentEmbedding(nn.Module):
    """
    包粒度：PLD 与内容均对齐到 hidden_size 后做门控融合，再叠加位置/segment 编码。
    不使用时间间隔特征。
    feature_mode:
      - content: 仅内容（768 维包向量 -> hidden_size）
      - pld: 仅包长/方向序列（词表 Embedding -> hidden_size）
      - content_pld: 门控融合（gate * content + (1-gate) * pld），均在 hidden_size 维
    """

    def __init__(self, args, pld_vocab):
        super(GatedPldContentEmbedding, self).__init__()
        self.feature_mode = args.feature_mode
        self.use_content = self.feature_mode in ("content", "content_pld")
        self.use_pld = self.feature_mode in ("pld", "content_pld")
        self.use_gated_fusion = self.feature_mode == "content_pld"

        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.max_seq_length = args.max_seq_length
        self.hidden_size = args.hidden_size
        self.embedding_dim = getattr(args, "embedding_dim", args.packet_feature_dim)
        pld_dim = getattr(args, "pld_embedding_dim", 0) or args.hidden_size
        self.pld_embedding_dim = pld_dim

        pad_id, unk_id, lookup_pos, lookup_neg = build_pld_id_lookup(
            pld_vocab, device=torch.device("cpu")
        )
        self.register_buffer("pld_pad_id", torch.tensor(pad_id, dtype=torch.long))
        self.register_buffer("pld_unk_id", torch.tensor(unk_id, dtype=torch.long))
        self.register_buffer("pld_lookup_pos", lookup_pos)
        self.register_buffer("pld_lookup_neg", lookup_neg)

        if self.use_content:
            if self.embedding_dim != args.hidden_size:
                self.content_projection = nn.Linear(self.embedding_dim, args.hidden_size)
            else:
                self.content_projection = None

        if self.use_pld:
            pld_vocab_size = len(pld_vocab)
            if pld_dim != args.hidden_size:
                self.pld_embedding = nn.Embedding(pld_vocab_size, pld_dim, padding_idx=pad_id)
                self.pld_projection = nn.Linear(pld_dim, args.hidden_size)
            else:
                self.pld_embedding = nn.Embedding(pld_vocab_size, args.hidden_size, padding_idx=pad_id)
                self.pld_projection = None

        if self.use_gated_fusion:
            gate_input_dim = args.hidden_size * 2
            self.gate = nn.Sequential(
                nn.Linear(gate_input_dim, args.hidden_size),
                nn.Sigmoid(),
            )

        self.position_embedding = nn.Embedding(self.max_seq_length, args.hidden_size)
        self.segment_embedding = nn.Embedding(3, args.hidden_size)

        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.hidden_size)

    def _encode_content(self, packet_features):
        if self.content_projection is not None:
            return self.content_projection(packet_features)
        return packet_features

    def _encode_pld(self, packet_length_directions, flow_lengths=None):
        pld_ids = pld_values_to_token_ids(
            packet_length_directions,
            int(self.pld_pad_id.item()),
            int(self.pld_unk_id.item()),
            self.pld_lookup_pos,
            self.pld_lookup_neg,
            flow_lengths=flow_lengths,
        )
        pld_emb = self.pld_embedding(pld_ids)
        if self.pld_projection is not None:
            pld_emb = self.pld_projection(pld_emb)
        return pld_emb

    def forward(self, packet_features, seg, packet_length_directions=None, flow_lengths=None):
        if self.use_content and self.use_pld:
            content_emb = self._encode_content(packet_features)
            pld_emb = self._encode_pld(packet_length_directions, flow_lengths)
            concat_features = torch.cat([content_emb, pld_emb], dim=-1)
            gate_values = self.gate(concat_features)
            fused = gate_values * content_emb + (1.0 - gate_values) * pld_emb
        elif self.use_content:
            fused = self._encode_content(packet_features)
        elif self.use_pld:
            if packet_length_directions is None:
                raise ValueError("feature_mode=pld 需要 packet_length_directions")
            fused = self._encode_pld(packet_length_directions, flow_lengths)
        else:
            raise ValueError("无效的 feature_mode: {}".format(self.feature_mode))

        pos_emb = self.position_embedding(
            torch.arange(0, fused.size(1), device=fused.device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(fused.size(0), 1)
        )
        seg_emb = self.segment_embedding(seg)
        emb = fused + pos_emb + seg_emb

        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        return self.dropout(emb)

    def compute_gate_values(self, packet_features, packet_length_directions, flow_lengths=None):
        """
        计算门控值 [B, L, hidden_size]。
        gate 接近 1 表示更依赖 content；接近 0 表示更依赖 PLD。
        仅在 feature_mode=content_pld 时有效，否则返回 None。
        """
        if not self.use_gated_fusion:
            return None
        content_emb = self._encode_content(packet_features)
        pld_emb = self._encode_pld(packet_length_directions, flow_lengths)
        concat_features = torch.cat([content_emb, pld_emb], dim=-1)
        return self.gate(concat_features)


# ===================== Model Definition =====================
class FlowLevelClassifier(nn.Module):
    """流级分类器：可配置 content / pld / content_pld 特征，包粒度门控融合。"""

    def __init__(self, args, pld_vocab):
        super(FlowLevelClassifier, self).__init__()

        self.packet_feature_dim = args.packet_feature_dim
        self.hidden_size = args.hidden_size
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.feature_mode = args.feature_mode

        self.embedding = GatedPldContentEmbedding(args, pld_vocab)
        self.flow_encoder = str2encoder[args.encoder](args)
        self.encoder_type = args.encoder

        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)

    def _pool_encoder_output(self, output, flow_lengths):
        batch_size, max_flow_length, _ = output.shape

        if flow_lengths is not None:
            pad_mask = torch.zeros(
                batch_size, max_flow_length, dtype=torch.bool, device=output.device
            )
            for i, length in enumerate(flow_lengths):
                actual_length = min(int(length.item()), max_flow_length)
                pad_mask[i, :actual_length] = True

            if self.pooling == "mean":
                mask_f = pad_mask.unsqueeze(-1).float()
                return (output * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
            if self.pooling == "max":
                masked = output.masked_fill(~pad_mask.unsqueeze(-1), float("-inf"))
                return torch.max(masked, dim=1)[0]
            if self.pooling == "last":
                last_idx = pad_mask.long().sum(dim=1).clamp(min=1) - 1
                return output[torch.arange(batch_size, device=output.device), last_idx, :]
            return output[:, 0, :]

        if self.pooling == "mean":
            return torch.mean(output, dim=1)
        if self.pooling == "max":
            return torch.max(output, dim=1)[0]
        if self.pooling == "last":
            return output[:, -1, :]
        return output[:, 0, :]

    def forward(self, packet_features, time_intervals, packet_sequences, tgt=None, flow_lengths=None):
        """
        packet_features: [B, L, D] 内容向量
        time_intervals: 保留接口以兼容数据管线，本模型不使用
        packet_sequences: [B, L, 1] 包长+方向 float
        """
        batch_size, max_flow_length, _ = packet_features.shape
        _ = time_intervals  # 不使用时间特征

        if self.feature_mode in ("content", "content_pld"):
            if not check_tensor_validity(packet_features, "packet_features"):
                raise ValueError("Input data contains NAN or INF values")

        seg = torch.zeros(batch_size, max_flow_length, dtype=torch.long, device=packet_features.device)
        pld = packet_sequences.squeeze(-1) if packet_sequences is not None else None

        emb = self.embedding(
            packet_features=packet_features,
            seg=seg,
            packet_length_directions=pld,
            flow_lengths=flow_lengths,
        )

        if not check_tensor_validity(emb, "embedding_output"):
            raise ValueError("Embedding output contains NAN or INF values")

        if self.encoder_type == "transformer":
            if flow_lengths is not None:
                enc_seg = torch.zeros(batch_size, max_flow_length, dtype=torch.long, device=emb.device)
                for i, length in enumerate(flow_lengths):
                    actual_length = min(int(length.item()), max_flow_length)
                    enc_seg[i, :actual_length] = 1
            else:
                enc_seg = torch.ones(batch_size, max_flow_length, dtype=torch.long, device=emb.device)
            output = self.flow_encoder(emb, enc_seg)
        else:
            enc_seg = torch.zeros(batch_size, max_flow_length, dtype=torch.long, device=emb.device)
            output = self.flow_encoder(emb, enc_seg)

        if not check_tensor_validity(output, "encoder_output"):
            raise ValueError("Encoder output contains NAN or INF values")

        pooled_output = self._pool_encoder_output(output, flow_lengths)

        if not check_tensor_validity(pooled_output, "pooled_output"):
            raise ValueError("Pooling output contains NAN or INF values")

        hidden = torch.tanh(self.output_layer_1(pooled_output))
        logits = self.output_layer_2(self.dropout(hidden))

        if tgt is not None:
            cls_loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            if not check_tensor_validity(cls_loss, "classification_loss"):
                print("Warning: Classification loss contains NAN or INF")
            return cls_loss, logits
        return None, logits

# ========== Utility Functions ==========
def check_tensor_validity(tensor, name="tensor"):
    """Check if tensor contains NAN or INF values"""
    if torch.isnan(tensor).any():
        print(f"Warning: {name} contains NAN values")
        return False
    if torch.isinf(tensor).any():
        print(f"Warning: {name} contains INF values")
        return False
    return True

def calculate_f1_score(confusion_matrix):
    """Calculate F1 score"""
    # Calculate precision, recall, and F1 score for each category
    precision = torch.zeros(confusion_matrix.size(0))
    recall = torch.zeros(confusion_matrix.size(0))
    f1_scores = torch.zeros(confusion_matrix.size(0))
    
    for i in range(confusion_matrix.size(0)):
        # True Positive
        tp = confusion_matrix[i, i]
        # False Positive - predicted as i but actually not i
        fp = confusion_matrix[:, i].sum() - tp
        # False Negative - actually i but predicted as not i
        fn = confusion_matrix[i, :].sum() - tp
        
        # Calculate precision and recall
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate F1 score
        f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    # Calculate macro averages
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1_scores.mean()
    
    # Calculate weighted averages
    class_counts = confusion_matrix.sum(dim=1)
    total_samples = class_counts.sum()
    weighted_precision = (precision * class_counts).sum() / total_samples if total_samples > 0 else 0
    weighted_recall = (recall * class_counts).sum() / total_samples if total_samples > 0 else 0
    weighted_f1 = (f1_scores * class_counts).sum() / total_samples if total_samples > 0 else 0
    
    return precision, recall, f1_scores, macro_precision, macro_recall, macro_f1, weighted_precision, weighted_recall, weighted_f1

# ===================== Parameter Initialization Functions =====================
def initialize_parameters_randomly(model, init_std=0.02):
    """
    Initialize all model parameters randomly.
    This function ensures complete random initialization for baseline comparison.
    
    Args:
        model: The model to initialize
        init_std: Standard deviation for normal distribution initialization (default: 0.02)
    
    Returns:
        total_params: Total number of parameters
        initialized_params: Number of initialized parameters
    """
    print("\n" + "=" * 60)
    print("Random Parameter Initialization")
    print("=" * 60)
    print(f"Initialization method: Normal distribution (std={init_std})")
    print("⚠️  ALL PARAMETERS ARE RANDOMLY INITIALIZED - NO PRETRAINED MODEL LOADED")
    print("=" * 60)
    
    total_params = 0
    initialized_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        # Initialize parameters (skip gamma and beta for LayerNorm)
        if "gamma" not in name and "beta" not in name:
            param.data.normal_(0, init_std)
            initialized_params += param_count
        else:
            # LayerNorm parameters: gamma=1, beta=0
            if "gamma" in name:
                param.data.fill_(1.0)
            elif "beta" in name:
                param.data.fill_(0.0)
            initialized_params += param_count
    
    print(f"Total parameters: {total_params:,}")
    print(f"Initialized parameters: {initialized_params:,}")
    print("=" * 60 + "\n")
    
    return total_params, initialized_params

def verify_all_parameters_trainable(model):
    """Verify that all parameters are trainable (not frozen)"""
    print("Verifying parameter trainability...")
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)
    
    print(f"Trainable parameters: {len(trainable_params)}")
    print(f"Frozen parameters: {len(frozen_params)}")
    
    if len(frozen_params) > 0:
        print(f"WARNING: Found {len(frozen_params)} frozen parameters:")
        for name in frozen_params[:10]:
            print(f"  - {name}")
        if len(frozen_params) > 10:
            print(f"  ... and {len(frozen_params) - 10} more")
        print("All parameters should be trainable for random initialization baseline!")
        return False
    
    if len(trainable_params) == 0:
        print("ERROR: No trainable parameters found! Model cannot be trained.")
        return False
    
    print("✓ All parameters are trainable")
    return True

# ===================== Data Loading =====================

def resolve_encoded_dataset_paths(path):
    """Resolve a JSONL path or a newline-delimited ``.manifest`` of JSONL shards."""
    if not path.endswith(".manifest"):
        return [path]

    manifest_dir = os.path.dirname(os.path.abspath(path))
    shard_paths = []
    with open(path, mode="r", encoding="utf-8") as manifest:
        for raw_line in manifest:
            shard_path = raw_line.strip()
            if not shard_path or shard_path.startswith("#"):
                continue
            if not os.path.isabs(shard_path):
                shard_path = os.path.join(manifest_dir, shard_path)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError("encoded dataset shard does not exist: {}".format(shard_path))
            shard_paths.append(shard_path)
    if not shard_paths:
        raise ValueError("encoded dataset manifest is empty: {}".format(path))
    return shard_paths


def count_labels_num(path):
    """Count labels without decoding the very large packet embedding arrays."""
    labels_set = set()
    with fileinput.input(
        files=resolve_encoded_dataset_paths(path),
        openhook=fileinput.hook_encoded("utf-8"),
    ) as f:
        for line in f:
            marker = '"category":'
            marker_start = line.find(marker)
            if marker_start < 0:
                raise ValueError("missing category in encoded dataset")
            value_start = marker_start + len(marker)
            value_end = line.find(',', value_start)
            if value_end < 0:
                value_end = line.find('}', value_start)
            labels_set.add(int(line[value_start:value_end].strip()))
    return len(labels_set)

def read_encoded_flow_dataset(args, path):
    """Read flow-level data; PLD-only JSONL may omit src and time_intervals."""
    dataset = []
    
    print(f"Reading encoded dataset: {path}")
    
    with fileinput.input(
        files=resolve_encoded_dataset_paths(path),
        openhook=fileinput.hook_encoded("utf-8"),
    ) as f:
        for line_id, line in enumerate(f):
            try:
                flow_data = json.loads(line.strip())
                
                # Handle packet_length_directions: should be float values (positive=outgoing, negative=incoming)
                # This format is consistent with pretraining stage
                if 'packet_length_directions' in flow_data:
                    pkt_length_dirs = flow_data['packet_length_directions']
                    # Convert to float array (consistent with pretraining format)
                    packet_sequences = []
                    for pld in pkt_length_dirs:
                        if isinstance(pld, str):
                            # Handle string format (should not happen if preprocessing is correct)
                            try:
                                if '/' in pld:
                                    parts = pld.split('/')
                                    length = float(parts[0])
                                    direction = int(parts[1]) if len(parts) > 1 else 1
                                    val = length if direction > 0 else -length
                                else:
                                    val = float(pld)
                            except (ValueError, AttributeError):
                                val = 0.0
                        else:
                            # Should be float (positive=outgoing, negative=incoming)
                            val = float(pld) if pld else 0.0
                        packet_sequences.append(val)
                    packet_sequences = np.array(packet_sequences, dtype=np.float32).reshape(-1, 1)
                elif 'packet_sequences' in flow_data:
                    # Backward compatibility: convert int to float
                    packet_sequences = np.array(flow_data['packet_sequences'], dtype=np.float32).reshape(-1, 1)
                else:
                    raise ValueError("missing packet_length_directions")

                sequence_length = packet_sequences.shape[0]
                if args.feature_mode == "pld":
                    # Sequence-only training never consumes content embeddings.
                    # Avoid materializing a 128x768 NumPy array for every flow.
                    src = np.zeros((sequence_length, 1), dtype=np.float32)
                elif 'src' in flow_data:
                    src = np.array(flow_data['src'], dtype=np.float32)
                    if src.ndim != 2:
                        raise ValueError("src must be a 2D packet-feature array")
                else:
                    raise ValueError("feature_mode={} requires src".format(args.feature_mode))

                if 'time_intervals' in flow_data:
                    time_intervals = np.array(
                        flow_data['time_intervals'], dtype=np.float32
                    ).reshape(-1, 1)
                else:
                    # Time is intentionally disabled by FlowLevelClassifier.
                    time_intervals = np.zeros((sequence_length, 1), dtype=np.float32)
                
                category = flow_data['category']
                packet_count = min(int(flow_data['packet_count']), sequence_length, args.max_flow_length)
                if packet_count <= 0:
                    raise ValueError("packet_count must be positive")
                
                # Pad to fixed length
                max_flow_length = args.max_flow_length
                if src.shape[0] < max_flow_length:
                    # Zero padding
                    pad_len = max_flow_length - src.shape[0]
                    src = np.vstack([src, np.zeros((pad_len, src.shape[1]))])
                    time_intervals = np.vstack([time_intervals, np.zeros((pad_len, 1))])
                    packet_sequences = np.vstack([packet_sequences, np.zeros((pad_len, 1), dtype=np.float32)])
                else:
                    # Truncate
                    src = src[:max_flow_length]
                    time_intervals = time_intervals[:max_flow_length]
                    packet_sequences = packet_sequences[:max_flow_length]
                
                dataset.append((src, time_intervals, packet_sequences, category, packet_count))
                
            except Exception as e:
                print(f"Error parsing line {line_id}: {e}")
                continue
    
    print(f"Reading completed, total {len(dataset)} flows")
    return dataset

def batch_loader(batch_size, src, time_intervals, packet_sequences, tgt, flow_lengths=None):
    """Batch data loader"""
    instances_num = src.shape[0]
    for i in range(instances_num // batch_size):
        src_batch = torch.FloatTensor(src[i * batch_size : (i + 1) * batch_size])
        time_batch = torch.FloatTensor(time_intervals[i * batch_size : (i + 1) * batch_size])
        # packet_sequences is now float (packet_length_directions), not int
        seq_batch = torch.FloatTensor(packet_sequences[i * batch_size : (i + 1) * batch_size])
        tgt_batch = torch.LongTensor(tgt[i * batch_size : (i + 1) * batch_size])
        
        if flow_lengths is not None:
            flow_lengths_batch = torch.LongTensor(flow_lengths[i * batch_size : (i + 1) * batch_size])
        else:
            flow_lengths_batch = None
            
        yield src_batch, time_batch, seq_batch, tgt_batch, flow_lengths_batch
    
    # Handle the last incomplete batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = torch.FloatTensor(src[instances_num // batch_size * batch_size :])
        time_batch = torch.FloatTensor(time_intervals[instances_num // batch_size * batch_size :])
        # packet_sequences is now float (packet_length_directions), not int
        seq_batch = torch.FloatTensor(packet_sequences[instances_num // batch_size * batch_size :])
        tgt_batch = torch.LongTensor(tgt[instances_num // batch_size * batch_size :])
        
        if flow_lengths is not None:
            flow_lengths_batch = torch.LongTensor(flow_lengths[instances_num // batch_size * batch_size :])
        else:
            flow_lengths_batch = None
            
        yield src_batch, time_batch, seq_batch, tgt_batch, flow_lengths_batch

# === Checkpoint and Resume Training Related ===
def train_model(args, model, optimizer, scheduler, src_batch, time_batch, seq_batch, tgt_batch, flow_lengths_batch=None):
    """Improved training model function with gradient clipping"""
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    time_batch = time_batch.to(args.device)
    seq_batch = seq_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    if flow_lengths_batch is not None:
        flow_lengths_batch = flow_lengths_batch.to(args.device)

    loss, _ = model(src_batch, time_batch, seq_batch, tgt_batch, flow_lengths_batch)
    
    # Multi-GPU processing: handle DataParallel loss first
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)
    
    # Check loss value (now loss is scalar)
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Warning: Loss value is {loss.item()}, skipping this batch")
        return loss

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    # Gradient clipping to prevent gradient explosion
    # Clip gradients for all parameters (all are trainable in random init mode)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if trainable_params:
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

    optimizer.step()
    scheduler.step()

    return loss


def get_base_model(model):
    """获取 DataParallel 包装下的原始模型。"""
    return model.module if isinstance(model, nn.DataParallel) else model


def _write_gate_param_artifacts(weight, bias, output_dir, source_model_path=None):
    """将 gate 的 weight/bias 写入标准输出文件。"""
    os.makedirs(output_dir, exist_ok=True)

    gate_state = {
        "0.weight": torch.from_numpy(weight),
        "0.bias": torch.from_numpy(bias),
    }
    gate_pt_path = os.path.join(output_dir, "gate_network_params.pt")
    torch.save(gate_state, gate_pt_path)

    np.save(os.path.join(output_dir, "gate_linear_weight.npy"), weight)
    np.save(os.path.join(output_dir, "gate_linear_bias.npy"), bias)

    summary = {
        "source_model_path": source_model_path,
        "gate_formula": "fused = gate * content + (1 - gate) * pld",
        "gate_interpretation": {
            "gate_near_1": "更依赖 content 模态",
            "gate_near_0": "更依赖 PLD 模态",
        },
        "weight_shape": list(weight.shape),
        "bias_shape": list(bias.shape),
        "weight_stats": {
            "mean": float(weight.mean()),
            "std": float(weight.std()),
            "min": float(weight.min()),
            "max": float(weight.max()),
        },
        "bias_stats": {
            "mean": float(bias.mean()),
            "std": float(bias.std()),
            "min": float(bias.min()),
            "max": float(bias.max()),
        },
        "bias_sigmoid_mean": float(1.0 / (1.0 + np.exp(-bias)).mean()),
        "per_dimension_bias_sigmoid": (1.0 / (1.0 + np.exp(-bias))).tolist(),
    }
    summary_path = os.path.join(output_dir, "gate_network_params_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"门控网络 state_dict 已保存: {gate_pt_path}")
    print(f"门控网络参数摘要已保存: {summary_path}")
    print(
        "  Linear weight shape: {}, bias shape: {}".format(
            summary["weight_shape"], summary["bias_shape"]
        )
    )
    print(
        "  bias 经 Sigmoid 后的均值 (无输入时的先验 content 权重): {:.4f}".format(
            summary["bias_sigmoid_mean"]
        )
    )
    return summary


def save_gate_network_params(model, output_dir):
    """
    导出训练后的门控网络可学习参数（Linear 权重与偏置）。
    gate 输出经 Sigmoid，融合公式: fused = gate * content + (1 - gate) * pld
    """
    embedding = get_base_model(model).embedding
    if not embedding.use_gated_fusion:
        print("feature_mode != content_pld，跳过门控网络参数导出")
        return

    gate_state = embedding.gate.state_dict()
    linear = embedding.gate[0]
    weight = linear.weight.detach().cpu().numpy()
    bias = linear.bias.detach().cpu().numpy()
    _write_gate_param_artifacts(weight, bias, output_dir)


def collect_and_save_gate_statistics(args, dataset, tag="test"):
    """
    在指定数据集上统计门控值，衡量模型对不同模态的自适应偏好。
    输出整体/逐维/逐类别的 content vs PLD 权重统计。
    """
    embedding = get_base_model(args.model).embedding
    if not embedding.use_gated_fusion:
        print("feature_mode != content_pld，跳过门控值统计")
        return

    src = np.stack([sample[0] for sample in dataset])
    time_intervals = np.stack([sample[1] for sample in dataset])
    packet_sequences = np.stack([sample[2] for sample in dataset])
    tgt = np.array([sample[3] for sample in dataset])
    flow_lengths = np.array([sample[4] for sample in dataset])

    hidden_size = args.hidden_size
    sum_gate = np.zeros(hidden_size, dtype=np.float64)
    sum_gate_sq = np.zeros(hidden_size, dtype=np.float64)
    total_valid_tokens = 0
    flow_mean_gates = []

    per_class_sum = {i: np.zeros(hidden_size, dtype=np.float64) for i in range(args.labels_num)}
    per_class_token_count = {i: 0 for i in range(args.labels_num)}
    per_class_flow_gates = {i: [] for i in range(args.labels_num)}

    args.model.eval()
    batch_size = args.batch_size

    for src_batch, time_batch, seq_batch, tgt_batch, flow_lengths_batch in batch_loader(
        batch_size, src, time_intervals, packet_sequences, tgt, flow_lengths
    ):
        src_batch = src_batch.to(args.device)
        seq_batch = seq_batch.to(args.device)
        if flow_lengths_batch is not None:
            flow_lengths_batch = flow_lengths_batch.to(args.device)

        pld = seq_batch.squeeze(-1)
        with torch.no_grad():
            gate_values = embedding.compute_gate_values(
                src_batch, pld, flow_lengths_batch
            )

        gate_np = gate_values.cpu().numpy()
        tgt_np = tgt_batch.numpy()
        batch_len = gate_np.shape[0]
        seq_len = gate_np.shape[1]

        for i in range(batch_len):
            if flow_lengths_batch is not None:
                length = min(int(flow_lengths_batch[i].item()), seq_len)
            else:
                length = seq_len
            if length <= 0:
                continue

            valid_gate = gate_np[i, :length, :]
            label = int(tgt_np[i])
            flow_mean = float(valid_gate.mean())

            sum_gate += valid_gate.sum(axis=0)
            sum_gate_sq += (valid_gate ** 2).sum(axis=0)
            total_valid_tokens += length
            flow_mean_gates.append(flow_mean)

            per_class_sum[label] += valid_gate.sum(axis=0)
            per_class_token_count[label] += length
            per_class_flow_gates[label].append(flow_mean)

    if total_valid_tokens == 0:
        print("无有效 token，跳过门控值统计")
        return

    mean_per_dim = sum_gate / total_valid_tokens
    var_per_dim = np.maximum(sum_gate_sq / total_valid_tokens - mean_per_dim ** 2, 0.0)
    std_per_dim = np.sqrt(var_per_dim)

    overall = {
        "dataset_tag": tag,
        "num_flows": len(flow_mean_gates),
        "num_valid_tokens": int(total_valid_tokens),
        "mean_content_weight": float(mean_per_dim.mean()),
        "mean_pld_weight": float((1.0 - mean_per_dim).mean()),
        "flow_level_content_weight": {
            "mean": float(np.mean(flow_mean_gates)),
            "std": float(np.std(flow_mean_gates)),
            "min": float(np.min(flow_mean_gates)),
            "max": float(np.max(flow_mean_gates)),
        },
        "per_dimension": {
            "mean_content_weight": mean_per_dim.tolist(),
            "std_content_weight": std_per_dim.tolist(),
            "mean_pld_weight": (1.0 - mean_per_dim).tolist(),
        },
    }

    per_class = {}
    for cls_id in range(args.labels_num):
        token_count = per_class_token_count[cls_id]
        flow_gates = per_class_flow_gates[cls_id]
        if token_count == 0:
            per_class[str(cls_id)] = {
                "num_flows": 0,
                "num_valid_tokens": 0,
                "mean_content_weight": None,
                "mean_pld_weight": None,
            }
            continue
        cls_mean_per_dim = per_class_sum[cls_id] / token_count
        per_class[str(cls_id)] = {
            "num_flows": len(flow_gates),
            "num_valid_tokens": int(token_count),
            "mean_content_weight": float(cls_mean_per_dim.mean()),
            "mean_pld_weight": float((1.0 - cls_mean_per_dim).mean()),
            "flow_level_content_weight_mean": float(np.mean(flow_gates)),
        }

    prefix = "gate_statistics_{}".format(tag)
    overall_path = os.path.join(args.output_dir, "{}_overall.json".format(prefix))
    per_class_path = os.path.join(args.output_dir, "{}_per_class.json".format(prefix))
    per_dim_path = os.path.join(args.output_dir, "{}_per_dim.npy".format(prefix))

    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)
    with open(per_class_path, "w", encoding="utf-8") as f:
        json.dump(per_class, f, indent=2, ensure_ascii=False)
    np.save(per_dim_path, mean_per_dim)

    print("\n" + "=" * 60)
    print("门控网络自适应统计 [{}]".format(tag))
    print("=" * 60)
    print("  平均 content 权重 (gate):     {:.4f}".format(overall["mean_content_weight"]))
    print("  平均 PLD 权重 (1-gate):       {:.4f}".format(overall["mean_pld_weight"]))
    print(
        "  流级 content 权重 (均值±标准差): {:.4f} ± {:.4f}".format(
            overall["flow_level_content_weight"]["mean"],
            overall["flow_level_content_weight"]["std"],
        )
    )
    print("  逐类别 content 权重:")
    for cls_id in range(args.labels_num):
        cls_stats = per_class[str(cls_id)]
        if cls_stats["mean_content_weight"] is not None:
            print(
                "    Class {}: content={:.4f}, pld={:.4f} ({} flows)".format(
                    cls_id,
                    cls_stats["mean_content_weight"],
                    cls_stats["mean_pld_weight"],
                    cls_stats["num_flows"],
                )
            )
    print("  已保存: {}, {}, {}".format(overall_path, per_class_path, per_dim_path))
    print("=" * 60 + "\n")


def export_final_gate_analysis(args, model):
    """训练结束后导出门控网络参数及在验证/测试集上的门控统计。"""
    if args.feature_mode != "content_pld":
        return

    print("\n" + "=" * 60)
    print("导出门控网络参数与模态自适应统计...")
    print("=" * 60)

    save_gate_network_params(model, args.output_dir)

    stats_path = args.test_path if args.test_path is not None else args.dev_path
    stats_tag = "test" if args.test_path is not None else "dev"
    if stats_path is not None:
        collect_and_save_gate_statistics(
            args, read_encoded_flow_dataset(args, stats_path), tag=stats_tag
        )
    else:
        print("未指定 test_path 或 dev_path，跳过门控值统计")


def evaluate(args, dataset, print_confusion_matrix=False, output_tag=None):
    """Evaluate model"""
    # Prepare data
    src = np.stack([sample[0] for sample in dataset])
    time_intervals = np.stack([sample[1] for sample in dataset])
    packet_sequences = np.stack([sample[2] for sample in dataset])
    tgt = np.array([sample[3] for sample in dataset])
    flow_lengths = np.array([sample[4] for sample in dataset])

    batch_size = args.batch_size

    correct = 0
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    args.model.eval()

    for src_batch, time_batch, seq_batch, tgt_batch, flow_lengths_batch in batch_loader(batch_size, src, time_intervals, packet_sequences, tgt, flow_lengths):
        src_batch = src_batch.to(args.device)
        time_batch = time_batch.to(args.device)
        seq_batch = seq_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        if flow_lengths_batch is not None:
            flow_lengths_batch = flow_lengths_batch.to(args.device)
        
        with torch.no_grad():
            _, logits = args.model(src_batch, time_batch, seq_batch, tgt_batch, flow_lengths_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()

    accuracy = correct / len(dataset)
    
    precision, recall, f1_scores, macro_precision, macro_recall, macro_f1, weighted_precision, weighted_recall, weighted_f1 = calculate_f1_score(confusion)
    
    if print_confusion_matrix:
        print("Confusion Matrix:")
        print(confusion)
        
        # Save confusion matrix
        cf_array = confusion.numpy()
        confusion_name = "confusion_matrix{}.txt".format(
            "_" + output_tag if output_tag else ""
        )
        with open(os.path.join(args.output_dir, confusion_name), 'w') as f:
            for cf_a in cf_array:
                f.write(str(cf_a) + '\n')

    # Print overall metrics
    print("=" * 60)
    print("Overall Evaluation Results:")
    print("=" * 60)
    print("Accuracy (Correct/Total): {:.4f} ({}/{})".format(accuracy, correct, len(dataset)))
    print("Overall Metrics (Macro Average):")
    print("  ACC (Accuracy):        {:.4f}".format(accuracy))
    print("  PR  (Precision):       {:.4f}".format(macro_precision))
    print("  RC  (Recall):          {:.4f}".format(macro_recall))
    print("  F1  (F1 Score):        {:.4f}".format(macro_f1))
    print("Weighted Metrics:")
    print("  PR  (Precision):       {:.4f}".format(weighted_precision))
    print("  RC  (Recall):          {:.4f}".format(weighted_recall))
    print("  F1  (F1 Score):        {:.4f}".format(weighted_f1))
    print()
    
    # Print detailed metrics for each category
    print("Detailed Metrics by Category:")
    print("-" * 60)
    print(f"{'Category':<8} {'Samples':<8} {'PR':<8} {'RC':<8} {'F1':<8}")
    print("-" * 60)
    
    for i in range(args.labels_num):
        class_count = confusion[:, i].sum().item()  # Correction: use actual class i sample count
        print(f"{i:<8} {class_count:<8} {precision[i]:<8.4f} {recall[i]:<8.4f} {f1_scores[i]:<8.4f}")
    
    print("-" * 60)
    
    # Save detailed results
    with open(f"{args.output_dir}/evaluation_results.txt", 'a') as f:
        f.write("=" * 60 + "\n")
        f.write("Overall Evaluation Results:\n")
        f.write("=" * 60 + "\n")
        f.write("Accuracy (Correct/Total): {:.4f} ({}/{})\n".format(accuracy, correct, len(dataset)))
        f.write("Overall Metrics (Macro Average):\n")
        f.write("  ACC: {:.4f}\n".format(accuracy))
        f.write("  PR:  {:.4f}\n".format(macro_precision))
        f.write("  RC:  {:.4f}\n".format(macro_recall))
        f.write("  F1:  {:.4f}\n".format(macro_f1))
        f.write("Weighted Metrics:\n")
        f.write("  PR:  {:.4f}\n".format(weighted_precision))
        f.write("  RC:  {:.4f}\n".format(weighted_recall))
        f.write("  F1:  {:.4f}\n".format(weighted_f1))
        f.write("\n")
        
        f.write("Detailed Metrics by Category:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Category':<8} {'Samples':<8} {'PR':<8} {'RC':<8} {'F1':<8}\n")
        f.write("-" * 60 + "\n")
        
        for i in range(args.labels_num):
            class_count = confusion[:, i].sum().item()  # Correction: use actual class i sample count
            f.write(f"{i:<8} {class_count:<8} {precision[i]:<8.4f} {recall[i]:<8.4f} {f1_scores[i]:<8.4f}\n")
        
        f.write("-" * 60 + "\n\n")
    
    # Save detailed metrics to JSON file for subsequent analysis
    detailed_metrics = {
        'overall': {
            'accuracy': float(accuracy),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'weighted_precision': float(weighted_precision),
            'weighted_recall': float(weighted_recall),
            'weighted_f1': float(weighted_f1),
            'total_samples': int(len(dataset)),
            'correct_predictions': int(correct)
        },
        'per_class': {}
    }
    
    for i in range(args.labels_num):
        class_count = confusion[:, i].sum().item()
        detailed_metrics['per_class'][str(i)] = {
            'sample_count': int(class_count),  # Ensure it's Python int
            'precision': float(precision[i].item()),  # Ensure it's Python float
            'recall': float(recall[i].item()),  # Ensure it's Python float
            'f1_score': float(f1_scores[i].item())  # Ensure it's Python float
        }
    
    metrics_name = "detailed_metrics{}.json".format(
        "_" + output_tag if output_tag else ""
    )
    with open(os.path.join(args.output_dir, metrics_name), 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    
    return accuracy, confusion, macro_precision, macro_recall, macro_f1, weighted_f1, precision, recall, f1_scores

def load_checkpoint_for_resume(args, model, optimizer, scheduler):
    """Load checkpoint to resume training"""
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Resuming training from checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        
        # Load model state
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Return training state
        start_epoch = checkpoint.get('epoch', 0)
        best_result = checkpoint.get('best_result', 0.0)
        global_step = checkpoint.get('global_step', 0)

        # Load early stopping state (compatible with old checkpoints)
        early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
        best_metric_value = checkpoint.get('best_metric_value', best_result)
        best_epoch = checkpoint.get('best_epoch', start_epoch)
        
        print(f"Resumed training state: epoch={start_epoch}, best_result={best_result:.4f}, global_step={global_step}")
        if 'early_stopping_counter' in checkpoint:
            print(f"Resumed early stopping state: counter={early_stopping_counter}, "
                  f"best_metric={best_metric_value:.4f} at epoch {best_epoch}")
        return start_epoch, best_result, global_step, early_stopping_counter, best_metric_value, best_epoch
    else:
        if args.warm_start_model_path:
            print("No checkpoint file found; continuing from warm-start weights with fresh optimizer state")
            return (
                0,
                args.warm_start_best_metric,
                0,
                0,
                args.warm_start_best_metric,
                0,
            )
        print("No checkpoint file found, starting training from scratch")
        return 0, 0.0, 0, 0, 0.0, 0

def save_checkpoint(args, model, optimizer, scheduler, epoch, best_result, global_step,
                    early_stopping_counter=0, best_metric_value=0.0, best_epoch=0):
    """Save checkpoint with quantity limit and automatic cleanup"""
    checkpoint = {
        'epoch': epoch,
        'best_result': best_result,
        'global_step': global_step,
        'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'early_stopping_counter': early_stopping_counter,
        'best_metric_value': best_metric_value,
        'best_epoch': best_epoch,
    }
    
    # Generate checkpoint filename
    checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Clean up old checkpoint files
    if hasattr(args, 'max_checkpoints') and args.max_checkpoints > 0:
        cleanup_old_checkpoints(args.output_dir, args.max_checkpoints)

def cleanup_old_checkpoints(output_dir, max_checkpoints):
    """Clean up old checkpoint files, keep only the latest few"""
    import glob
    import re
    
    # Find all checkpoint files
    checkpoint_pattern = os.path.join(output_dir, 'checkpoint_epoch_*.pt')
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if len(checkpoint_files) <= max_checkpoints:
        return
    
    # Sort by epoch (extract epoch number)
    def extract_epoch(filename):
        match = re.search(r'checkpoint_epoch_(\d+)\.pt', filename)
        return int(match.group(1)) if match else 0
    
    checkpoint_files.sort(key=extract_epoch, reverse=True)  # Latest first
    
    # Delete excess old files
    files_to_delete = checkpoint_files[max_checkpoints:]
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted old checkpoint: {os.path.basename(file_path)}")
        except OSError as e:
            print(f"Failed to delete file {file_path}: {e}")
    
    print(f"Checkpoint cleanup completed, keeping the latest {max_checkpoints} files")

def build_optimizer(args, model):
    """Build optimizer and scheduler - for ALL parameters (end-to-end training)"""
    # Get ALL parameters (all are trainable in random init mode)
    param_optimizer = list(model.named_parameters())
    
    if len(param_optimizer) == 0:
        raise ValueError("No parameters found! Cannot build optimizer.")
    
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type")

    # Flow-level specific parameters
    parser.add_argument("--max_flow_length", type=int, default=128,
                        help="Maximum flow length (packet count)")
    parser.add_argument("--packet_feature_dim", type=int, default=768,
                        help="Packet-level feature dimension")
    
    # PLD 词表 + 门控融合特征配置（不使用时间特征）
    parser.add_argument(
        "--pld_vocab_path",
        type=str,
        default="/home/sunshine/Baseline/BERT-ps/models/rm_128_pld_corpus_vocab.txt",
        help="包长/方向序列词表路径（dilength token，如 +64 / -128）",
    )
    parser.add_argument(
        "--feature_mode",
        choices=["content", "pld", "content_pld"],
        default="content_pld",
        help="下游任务使用的特征：仅内容 / 仅包长方向 / 门控融合二者",
    )
    parser.add_argument(
        "--pld_embedding_dim",
        type=int,
        default=0,
        help="PLD 词表 Embedding 维度；0 表示与 hidden_size 一致（默认直接 768 维后与内容门控融合）",
    )
    
    # Random initialization parameters
    parser.add_argument("--init_std", type=float, default=0.02,
                        help="Standard deviation for random parameter initialization (default: 0.02)")
    parser.add_argument(
        "--warm_start_model_path",
        type=str,
        default=None,
        help="Optional classifier state_dict used to continue training with a fresh optimizer.",
    )
    parser.add_argument(
        "--warm_start_best_metric",
        type=float,
        default=0.0,
        help="Validation metric of the warm-start model; retained until later validation improves.",
    )
    
    # Output directory parameters
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory path for saving logs, results, etc.")
    
    # GPU selection parameters
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7",
                        help="Specify GPU IDs to use, comma-separated")
    
    # Evaluation frequency parameters
    parser.add_argument("--eval_epochs", type=int, default=10,
                        help="Evaluation frequency (default every 10 epochs)")

    parser.add_argument(
        "--temporal_test_paths",
        nargs="*",
        default=[],
        help="Additional later-period JSONL test sets evaluated with the same best model.",
    )
    parser.add_argument(
        "--inference_only",
        action="store_true",
        help=(
            "Load --output_model_path as an already trained classifier and evaluate "
            "--test_path plus --temporal_test_paths without training."
        ),
    )
    
    # Early stopping parameters
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Stop training if the monitored metric does not improve for this many "
                             "consecutive evaluations (set <=0 to disable, default: 5)")
    parser.add_argument("--early_stopping_metric", type=str, default="accuracy",
                        choices=["accuracy", "macro_f1", "weighted_f1"],
                        help="Metric for early stopping (default: accuracy)")
    
    # Checkpoint parameters
    parser.add_argument("--checkpoint_path", type=str, default=None,
                    help="Checkpoint path for resuming training")
    parser.add_argument("--max_checkpoints", type=int, default=5,
                    help="Maximum number of checkpoints to save, exceeding this will delete the oldest")
    parser.add_argument("--save_checkpoint_freq", type=int, default=10,
                    help="Save checkpoint every N epochs (only during evaluation)")
    parser.add_argument("--save_best_only", action="store_true",
                    help="Whether to only save the best model checkpoint")

    args = parser.parse_args()

    # Set GPU
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"Set GPU usage: {args.gpu_ids}")

    # 内容向量任务不需要 BERT 文本词表；PLD 词表单独加载
    args.vocab_path = None
    args.spm_model_path = None
    args.tgt_vocab_path = None
    args.tgt_spm_model_path = None

    args = load_hyperparam(args)

    if getattr(args, "max_flow_length", None):
        args.max_seq_length = args.max_flow_length

    args.feature_mode = getattr(args, "feature_mode", "content_pld")
    pld_dim = getattr(args, "pld_embedding_dim", 0)
    args.pld_embedding_dim = args.hidden_size if (pld_dim is None or pld_dim <= 0) else pld_dim
    args.embedding_dim = getattr(args, "embedding_dim", args.packet_feature_dim)

    if not os.path.isfile(args.pld_vocab_path):
        raise FileNotFoundError("PLD 词表不存在: {}".format(args.pld_vocab_path))

    print("Gated PLD + Content embedding configuration:")
    print("  feature_mode: {}".format(args.feature_mode))
    print("  pld_vocab_path: {}".format(args.pld_vocab_path))
    print("  hidden_size: {}".format(args.hidden_size))
    print("  packet_feature_dim: {}".format(args.packet_feature_dim))
    print("  pld_embedding_dim: {}".format(args.pld_embedding_dim))
    if args.pld_embedding_dim == args.hidden_size:
        print("  pld_encode: token -> {}-dim (direct, no pre-gate projection)".format(args.hidden_size))
    else:
        print("  pld_encode: token -> {}-dim -> Linear -> {}-dim".format(
            args.pld_embedding_dim, args.hidden_size))
    print("  fusion: gate * content_{} + (1-gate) * pld_{}".format(
        args.hidden_size, args.hidden_size))
    print("  max_seq_length: {}".format(args.max_seq_length))
    print("  time_intervals: disabled")

    set_seed(args.seed)

    args.labels_num = count_labels_num(args.train_path)
    print("Number of labels: {}".format(args.labels_num))

    pld_vocab = load_pld_vocab(args.pld_vocab_path)
    model = FlowLevelClassifier(args, pld_vocab)

    if args.inference_only:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(args.device)
        if args.output_dir is None:
            args.output_dir = os.path.dirname(args.output_model_path)
        os.makedirs(args.output_dir, exist_ok=True)

        load_file_path = args.output_model_path
        if not load_file_path.endswith((".bin", ".pt")):
            load_file_path += ".bin"
        print("Inference-only mode: loading {}".format(load_file_path))
        loaded = torch.load(load_file_path, map_location=args.device)
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            loaded = loaded["model_state_dict"]
        if not isinstance(loaded, dict):
            raise TypeError("Unsupported model file: expected a state_dict mapping")
        if any(key.startswith("module.") for key in loaded):
            loaded = {
                key[len("module."):] if key.startswith("module.") else key: value
                for key, value in loaded.items()
            }
        model.load_state_dict(loaded, strict=True)
        model.eval()
        args.model = model

        if args.test_path is None:
            raise ValueError("--test_path is required in --inference_only mode")
        evaluation_paths = [args.test_path] + list(args.temporal_test_paths)
        temporal_summary = {}
        for evaluation_path in evaluation_paths:
            evaluation_tag = os.path.splitext(os.path.basename(evaluation_path))[0]
            print("\nEvaluating temporal test set: {}".format(evaluation_tag))
            evaluation_result = evaluate(
                args,
                read_encoded_flow_dataset(args, evaluation_path),
                True,
                output_tag=evaluation_tag,
            )
            accuracy, _, macro_precision, macro_recall, macro_f1 = evaluation_result[:5]
            temporal_summary[evaluation_tag] = {
                "path": evaluation_path,
                "accuracy": float(accuracy),
                "macro_precision": float(macro_precision),
                "macro_recall": float(macro_recall),
                "macro_f1": float(macro_f1),
            }

        baseline_tag = os.path.splitext(os.path.basename(args.test_path))[0]
        baseline_f1 = temporal_summary[baseline_tag]["macro_f1"]
        baseline_acc = temporal_summary[baseline_tag]["accuracy"]
        for metrics in temporal_summary.values():
            metrics["macro_f1_change_from_baseline"] = metrics["macro_f1"] - baseline_f1
            metrics["accuracy_change_from_baseline"] = metrics["accuracy"] - baseline_acc
        metrics_path = os.path.join(args.output_dir, "temporal_drift_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(temporal_summary, handle, indent=2)
        print("Inference-only temporal metrics written to {}".format(metrics_path))
        return

    # ===== KEY DIFFERENCE: Random initialization instead of loading pretrained model =====
    # IMPORTANT: This script does NOT load any pretrained model
    # All parameters are randomly initialized for baseline comparison
    print("\n" + "=" * 60)
    if args.warm_start_model_path:
        print("WARM-START CONTINUATION MODE")
        print("The classifier weights will be loaded from the explicit warm-start model.")
        print("Optimizer and scheduler state will be initialized afresh.")
    else:
        print("⚠️  RANDOM INITIALIZATION MODE")
        print("This script trains from scratch with random initialization.")
        print("No pretrained model will be loaded (even if --pretrained_model_path is provided).")
    print("All parameters will be trained end-to-end.")
    print("=" * 60 + "\n")
    
    # Initialize all parameters randomly (ignore pretrained_model_path if provided).
    # An explicit warm-start path may then replace those values for continuation runs.
    total_params, initialized_params = initialize_parameters_randomly(model, init_std=args.init_std)

    if args.warm_start_model_path:
        if not os.path.isfile(args.warm_start_model_path):
            raise FileNotFoundError(
                "Warm-start model does not exist: {}".format(args.warm_start_model_path)
            )
        warm_state = torch.load(args.warm_start_model_path, map_location="cpu")
        if isinstance(warm_state, dict) and "model_state_dict" in warm_state:
            warm_state = warm_state["model_state_dict"]
        if not isinstance(warm_state, dict):
            raise TypeError("Warm-start model must contain a state_dict mapping")
        if any(key.startswith("module.") for key in warm_state):
            warm_state = {
                key[len("module."):] if key.startswith("module.") else key: value
                for key, value in warm_state.items()
            }
        model.load_state_dict(warm_state, strict=True)
        print("Loaded warm-start classifier: {}".format(args.warm_start_model_path))
        print("Warm-start validation metric: {:.4f}".format(args.warm_start_best_metric))
    
    # Verify that all parameters are trainable (not frozen)
    if not verify_all_parameters_trainable(model):
        print("ERROR: Parameter trainability verification failed!")
        print("All parameters must be trainable for random initialization baseline.")
        return

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.output_model_path)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")
    
    print(f"Output directory: {args.output_dir}")

    # Record the exact optimizer update scope for reproducible parameter-count
    # reporting. Parameters with requires_grad=True are all passed to the
    # optimizer by build_optimizer; buffers are intentionally excluded.
    parameter_groups = {}
    parameter_tensors = []
    trainable_parameter_elements = 0
    frozen_parameter_elements = 0
    trainable_parameter_bytes = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        group = name.split(".", 1)[0]
        group_stats = parameter_groups.setdefault(
            group,
            {"total_elements": 0, "trainable_elements": 0, "frozen_elements": 0},
        )
        group_stats["total_elements"] += count
        if parameter.requires_grad:
            trainable_parameter_elements += count
            trainable_parameter_bytes += count * parameter.element_size()
            group_stats["trainable_elements"] += count
        else:
            frozen_parameter_elements += count
            group_stats["frozen_elements"] += count
        parameter_tensors.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "elements": count,
                "requires_grad": bool(parameter.requires_grad),
            }
        )
    parameter_scope = {
        "definition": (
            "Parameters with requires_grad=true constitute the optimizer update scope; "
            "non-parameter buffers are excluded."
        ),
        "total_parameter_elements": trainable_parameter_elements + frozen_parameter_elements,
        "trainable_parameter_elements": trainable_parameter_elements,
        "frozen_parameter_elements": frozen_parameter_elements,
        "trainable_parameter_bytes": trainable_parameter_bytes,
        "trainable_fraction": (
            trainable_parameter_elements /
            max(1, trainable_parameter_elements + frozen_parameter_elements)
        ),
        "groups": parameter_groups,
        "tensors": parameter_tensors,
    }
    with open(os.path.join(args.output_dir, "parameter_update_scope.json"), "w", encoding="utf-8") as handle:
        json.dump(parameter_scope, handle, indent=2)
    print(
        "Optimizer update scope: {:,}/{:,} parameter elements ({:.2%})".format(
            trainable_parameter_elements,
            trainable_parameter_elements + frozen_parameter_elements,
            parameter_scope["trainable_fraction"],
        )
    )
    
    # Save initialization information
    with open(f"{args.output_dir}/initialization_info.txt", 'w') as f:
        f.write("Continuation Initialization Information\n" if args.warm_start_model_path else "Random Initialization Information\n")
        f.write("=" * 60 + "\n")
        if args.warm_start_model_path:
            f.write("Initialization mode: Warm-start continuation\n")
            f.write(f"Warm-start model: {args.warm_start_model_path}\n")
            f.write(f"Warm-start best metric: {args.warm_start_best_metric}\n")
        else:
            f.write("Initialization mode: Random (no pretrained model)\n")
        f.write(f"Initialization std: {args.init_std}\n")
        f.write(f"feature_mode: {args.feature_mode}\n")
        f.write(f"pld_vocab_path: {args.pld_vocab_path}\n")
        f.write(f"pld_embedding_dim: {args.pld_embedding_dim}\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Initialized parameters: {initialized_params:,}\n")
        f.write(f"All parameters are trainable: Yes\n")
        f.write(f"Training mode: End-to-end (all layers)\n")
        f.write("time_intervals: disabled\n")
        f.write("\n")
        f.write("This is a baseline comparison experiment.\n")
        f.write("Compare results with pretrained model to evaluate pretraining effectiveness.\n")

    # Keep the incoming best model as a valid output candidate. If no later
    # validation improves on it, final test evaluation still uses this model.
    if args.warm_start_model_path:
        initial_output_path = args.output_model_path
        if not initial_output_path.endswith((".bin", ".pt")):
            initial_output_path += ".bin"
        if not os.path.exists(initial_output_path):
            save_model(model, initial_output_path)
            print("Saved warm-start candidate to: {}".format(initial_output_path))
        else:
            print("Preserving existing best candidate: {}".format(initial_output_path))

    # Training phase
    print("Start reading encoded training data...")
    trainset = read_encoded_flow_dataset(args, args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size
    
    # Prepare training data
    src = np.stack([example[0] for example in trainset])
    time_intervals = np.stack([example[1] for example in trainset])
    packet_sequences = np.stack([example[2] for example in trainset])
    tgt = np.array([example[3] for example in trainset])
    flow_lengths = np.array([example[4] for example in trainset])

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("Number of training samples:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex to use fp16 training")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        print("Training with {} GPUs".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    print("Start training, using {} GPUs".format(torch.cuda.device_count()))
    args.model = model

    # Check if need to resume training
    resume_result = load_checkpoint_for_resume(args, model, optimizer, scheduler)
    if len(resume_result) == 6:
        start_epoch, best_result, global_step, early_stopping_counter, best_metric_value, best_epoch = resume_result
    else:
        start_epoch, best_result, global_step = resume_result
        early_stopping_counter = 0
        best_metric_value = best_result
        best_epoch = start_epoch

    total_loss, result = 0.0, 0.0

    print("Start training...")
    print(f"Evaluation frequency: every {args.eval_epochs} epoch(s)")
    if args.early_stopping_patience > 0:
        print(f"Early stopping: patience={args.early_stopping_patience}, metric={args.early_stopping_metric}")
        if args.eval_epochs > 1:
            print(f"  (patience counts consecutive evaluations; each evaluation spans {args.eval_epochs} epoch(s))")
        if start_epoch > 0:
            print(f"Resumed early stopping counter: {early_stopping_counter}/{args.early_stopping_patience}")
    else:
        print("Early stopping: disabled")
    print("⚠️  RANDOM INITIALIZATION - All parameters trained end-to-end")

    for epoch in tqdm.tqdm(range(start_epoch + 1, args.epochs_num + 1)):
        model.train()
        for i, (src_batch, time_batch, seq_batch, tgt_batch, flow_lengths_batch) in enumerate(batch_loader(batch_size, src, time_intervals, packet_sequences, tgt, flow_lengths)):
            loss = train_model(args, model, optimizer, scheduler, src_batch, time_batch, seq_batch, tgt_batch, flow_lengths_batch)
            total_loss += loss.item()
            global_step += 1
            
            if (i + 1) % args.report_steps == 0:
                print("Epoch: {}, Training steps: {}, Average loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                # Save training log
                with open(f"{args.output_dir}/training_log.txt", 'a') as f:
                    f.write("Epoch: {}, Training steps: {}, Average loss: {:.3f}\n".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

        # Evaluate by epoch
        if epoch % args.eval_epochs == 0:
            print(f"\n=== Epoch {epoch} Evaluation ===")
            result = evaluate(args, read_encoded_flow_dataset(args, args.dev_path))
            accuracy, confusion, macro_precision, macro_recall, macro_f1, weighted_f1, precision, recall, f1_scores = result

            if args.early_stopping_metric == "accuracy":
                current_metric_value = accuracy
            elif args.early_stopping_metric == "macro_f1":
                current_metric_value = macro_f1
            else:
                current_metric_value = weighted_f1

            if current_metric_value > best_metric_value:
                improvement = current_metric_value - best_metric_value
                best_metric_value = current_metric_value
                best_epoch = epoch
                early_stopping_counter = 0

                if args.early_stopping_metric == "accuracy":
                    best_result = accuracy

                # Ensure output path is a file
                if not args.output_model_path.endswith('.bin') and not args.output_model_path.endswith('.pt'):
                    output_file_path = args.output_model_path + '.bin'
                else:
                    output_file_path = args.output_model_path
                save_model(model, output_file_path)
                print(f"✓ New best {args.early_stopping_metric}: {best_metric_value:.4f} (improvement: +{improvement:.4f})")
                print(f"  Best metrics - ACC: {accuracy:.4f}, Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")
            else:
                if args.early_stopping_patience > 0:
                    early_stopping_counter += 1
                    print(f"No improvement. Best {args.early_stopping_metric}: {best_metric_value:.4f} (at epoch {best_epoch})")
                    print(f"Early stopping counter: {early_stopping_counter}/{args.early_stopping_patience}")
            print(f"=== Evaluation completed ===\n")

            # Early stopping check
            if args.early_stopping_patience > 0 and early_stopping_counter >= args.early_stopping_patience:
                print("=" * 60)
                print("Early stopping triggered!")
                print(f"No improvement for {args.early_stopping_patience} consecutive evaluation(s).")
                print(f"Best {args.early_stopping_metric}: {best_metric_value:.4f} at epoch {best_epoch}")
                print(f"Stopping training at epoch {epoch}")
                print("=" * 60)
                break
            
            # Save checkpoint (decide whether to save based on configuration)
            should_save_checkpoint = False
            
            # Check save frequency
            if hasattr(args, 'save_checkpoint_freq') and epoch % args.save_checkpoint_freq == 0:
                should_save_checkpoint = True
            
            # Check if only save best model
            if hasattr(args, 'save_best_only') and args.save_best_only:
                should_save_checkpoint = should_save_checkpoint and (current_metric_value == best_metric_value)
            
            # If checkpoint saving is configured, save it
            if should_save_checkpoint:
                save_checkpoint(args, model, optimizer, scheduler, epoch, best_result, global_step,
                                  early_stopping_counter, best_metric_value, best_epoch)
            else:
                print(f"Skip checkpoint saving (epoch {epoch})")

    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)
    if args.early_stopping_patience > 0 and early_stopping_counter >= args.early_stopping_patience:
        print(f"Training stopped early: no improvement for {args.early_stopping_patience} consecutive evaluation(s).")
    else:
        print(f"Training completed all {args.epochs_num} epochs.")
    print(f"Best {args.early_stopping_metric}: {best_metric_value:.4f} at epoch {best_epoch}")
    print(f"Best model saved to: {args.output_model_path}")
    print("=" * 60 + "\n")

    # Test phase
    if args.test_path is not None:
        print("\n" + "=" * 60)
        print("Evaluating on test set...")
        print("=" * 60)
        # Ensure load path is a file
        if not args.output_model_path.endswith('.bin') and not args.output_model_path.endswith('.pt'):
            load_file_path = args.output_model_path + '.bin'
        else:
            load_file_path = args.output_model_path
        
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(load_file_path))
        else:
            model.load_state_dict(torch.load(load_file_path))
        test_tag = os.path.splitext(os.path.basename(args.test_path))[0]
        test_result = evaluate(
            args,
            read_encoded_flow_dataset(args, args.test_path),
            True,
            output_tag=test_tag,
        )
        test_acc, _, test_pr, test_rc, test_f1, _, test_precision, test_recall, test_f1_per_class = test_result

        print("\n" + "=" * 60)
        print("Final Test Set Metrics Summary:")
        print("=" * 60)
        print("  ACC (Accuracy):  {:.4f}".format(test_acc))
        print("  PR  (Precision): {:.4f}  [Macro]".format(test_pr))
        print("  RC  (Recall):    {:.4f}  [Macro]".format(test_rc))
        print("  F1  (F1 Score):  {:.4f}  [Macro]".format(test_f1))
        print()
        print("Per-Class Metrics (PR, RC, F1):")
        print("-" * 60)
        print(f"{'Class':<8} {'PR':<12} {'RC':<12} {'F1':<12}")
        print("-" * 60)
        for i in range(args.labels_num):
            print(f"{i:<8} {test_precision[i]:<12.4f} {test_recall[i]:<12.4f} {test_f1_per_class[i]:<12.4f}")
        print("=" * 60 + "\n")

        with open(f"{args.output_dir}/test_metrics_summary.txt", 'w') as f:
            f.write("Final Test Set Metrics Summary\n")
            f.write("=" * 60 + "\n")
            f.write("ACC: {:.4f}\n".format(test_acc))
            f.write("PR:  {:.4f}  [Macro]\n".format(test_pr))
            f.write("RC:  {:.4f}  [Macro]\n".format(test_rc))
            f.write("F1:  {:.4f}  [Macro]\n".format(test_f1))
            f.write("\nPer-Class (PR, RC, F1):\n")
            for i in range(args.labels_num):
                f.write("  Class {}: PR={:.4f}, RC={:.4f}, F1={:.4f}\n".format(
                    i, test_precision[i], test_recall[i], test_f1_per_class[i]))

        temporal_summary = {
            test_tag: {
                "path": args.test_path,
                "accuracy": float(test_acc),
                "macro_precision": float(test_pr),
                "macro_recall": float(test_rc),
                "macro_f1": float(test_f1),
            }
        }
        for temporal_path in args.temporal_test_paths:
            temporal_tag = os.path.splitext(os.path.basename(temporal_path))[0]
            print("\nEvaluating temporal test set: {}".format(temporal_tag))
            temporal_result = evaluate(
                args,
                read_encoded_flow_dataset(args, temporal_path),
                True,
                output_tag=temporal_tag,
            )
            temporal_acc, _, temporal_pr, temporal_rc, temporal_f1 = temporal_result[:5]
            temporal_summary[temporal_tag] = {
                "path": temporal_path,
                "accuracy": float(temporal_acc),
                "macro_precision": float(temporal_pr),
                "macro_recall": float(temporal_rc),
                "macro_f1": float(temporal_f1),
            }

        baseline_f1 = temporal_summary[test_tag]["macro_f1"]
        baseline_acc = temporal_summary[test_tag]["accuracy"]
        for metrics in temporal_summary.values():
            metrics["macro_f1_change_from_baseline"] = metrics["macro_f1"] - baseline_f1
            metrics["accuracy_change_from_baseline"] = metrics["accuracy"] - baseline_acc
        with open(os.path.join(args.output_dir, "temporal_drift_metrics.json"), "w") as f:
            json.dump(temporal_summary, f, indent=2)

        export_final_gate_analysis(args, model)
    else:
        # 无测试集时，加载最优模型后仍导出门控参数
        if os.path.exists(
            args.output_model_path
            if args.output_model_path.endswith(('.bin', '.pt'))
            else args.output_model_path + '.bin'
        ):
            load_file_path = (
                args.output_model_path
                if args.output_model_path.endswith(('.bin', '.pt'))
                else args.output_model_path + '.bin'
            )
            if torch.cuda.device_count() > 1:
                model.module.load_state_dict(torch.load(load_file_path))
            else:
                model.load_state_dict(torch.load(load_file_path))
        export_final_gate_analysis(args, model)

if __name__ == "__main__":
    main()

"""
================================================================================
Flow-level BERT Fine-tuning (Random Init + Gated PLD/Content)
================================================================================

特征模式 (--feature_mode):
  content      - 仅 768 维包内容向量（对齐 hidden_size）
  pld          - 仅包长/方向序列（词表 Embedding 直接为 hidden_size 维，默认 768）
  content_pld  - 二者在 hidden_size 维门控融合: gate*content + (1-gate)*pld

--pld_embedding_dim: 默认 0，表示 PLD 直接嵌入 hidden_size（768）；非 0 时先嵌入该维再 Linear 到 hidden_size

PLD token 规则（与 BERT-ps dilength 一致）:
  出向: +<len>  入向: -<len>  零长度: [UNK]  填充: [PAD]
  |len| 裁剪至 1500

数据 JSONL 字段（time_intervals 可保留但不会被使用）:
  src, packet_length_directions, category, packet_count

示例:
python rm_fine_temporal_classifier_random_init.py \\
    --train_path /path/to/train.jsonl \\
    --dev_path /path/to/dev.jsonl \\
    --test_path /path/to/test.jsonl \\
    --output_model_path /path/to/output/model.bin \\
    --output_dir /path/to/logs \\
    --config_path /path/to/config.json \\
    --pld_vocab_path /home/sunshine/Baseline/BERT-ps/models/rm_128_pld_corpus_vocab.txt \\
    --feature_mode content_pld \\
    --max_flow_length 128 \\
    --epochs_num 20 --batch_size 32 --learning_rate 1e-4
================================================================================
"""
