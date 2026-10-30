#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning Data Generation Script

This script is responsible for generating fine-tuning datasets from raw pcap files.
It includes data cleaning, file preprocessing, and flow generation for network traffic analysis.

"""

import numpy as np
import json
import os
import time
import xlrd
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from scipy.stats import skew,kurtosis
import sys
import csv
import copy
import tqdm
import random
import shutil
import argparse
import rm_flow_dataset_generation
import dataset_cleanning


# Data paths and generation parameters
pcap_path = "/mnt/nfs/sunshine/Datasets/visquic_subset_max1000/"  # Input pcap files directory
dataset_save_path = "/mnt/sdb/sunshine/MGPT/datasets/mix/"  # Output dataset directory
samples = [1000]  # Number of samples to generate per category
features = ["flow_order","relative_time","packet_length_direction","text_a"]  # Features to extract
dataset_level = "flow"  # Dataset level (flow-level analysis)
payload_packet = 128  # Limit packets parsed per flow
# Packet encoding modes:
# - "field": unchanged legacy IPv4 parser with fixed Ethernet payload offset
# - "field_ipv4_ipv6": opt-in IPv6 fields, legacy IPv4 payload behavior
# - "field_ipv4_ipv6_l2safe": corrected IPv4/IPv6 parser with link-layer-independent payload
# - "field_ipv4_ipv6_l2safe_app": additionally parse TLS records and QUIC headers,
#   retaining unparsed/encrypted bytes with the original bi-byte fallback
# - "bigram": bi-byte split (Baseline get_feature_packet)
packet_encoding_mode = "field"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate packet-field TSV data from class-organized PCAP flows."
    )
    parser.add_argument("--pcap-path", default=pcap_path)
    parser.add_argument(
        "--output-file",
        default=os.path.join(
            rm_flow_dataset_generation.fine_tuning_flowdataset_dir,
            rm_flow_dataset_generation.fine_tuning_flowdataset_name,
        ),
    )
    parser.add_argument("--samples-per-class", type=int, default=samples[0])
    parser.add_argument("--payload-packet", type=int, default=payload_packet)
    parser.add_argument("--payload-length", type=int, default=128)
    parser.add_argument(
        "--packet-encoding-mode",
        choices=[
            "field",
            "field_zero_length",
            "field_ipv4_ipv6",
            "field_ipv4_ipv6_l2safe",
            "field_ipv4_ipv6_l2safe_zero_length",
            "field_ipv4_ipv6_l2safe_app",
            "bigram",
        ],
        default=packet_encoding_mode,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output instead of appending to it.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    output_file = os.path.abspath(args.output_file)
    if os.path.exists(output_file):
        if not args.overwrite:
            raise FileExistsError(
                "Output already exists; pass --overwrite to replace it: {}".format(output_file)
            )
        os.remove(output_file)

    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    rm_flow_dataset_generation.fine_tuning_flowdataset_dir = output_dir + os.sep
    rm_flow_dataset_generation.fine_tuning_flowdataset_name = os.path.basename(output_file)

    random.seed(args.seed)
    np.random.seed(args.seed)
    rm_flow_dataset_generation.set_finetuning_output_buffering(True)
    try:
        rm_flow_dataset_generation.generation(
            args.pcap_path,
            [args.samples_per_class],
            features,
            splitcap=False,
            payload_length=args.payload_length,
            payload_packet=args.payload_packet,
            dataset_save_path=output_dir + os.sep,
            dataset_level=dataset_level,
            packet_encoding_mode=args.packet_encoding_mode,
        )
    finally:
        rm_flow_dataset_generation.close_finetuning_output()

    label_names = sorted(
        entry.name for entry in os.scandir(args.pcap_path) if entry.is_dir()
    )
    mapping_path = output_file + ".labels.json"
    with open(mapping_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"label_to_id": {name: index for index, name in enumerate(label_names)}},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print("Generated TSV: {}".format(output_file))
    print("Label mapping: {}".format(mapping_path))
