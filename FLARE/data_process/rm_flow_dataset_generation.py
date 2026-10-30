#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import sys
import copy
# import xlrd
import json
#import tqdm
from tqdm import tqdm
import shutil
import pickle
import random
import binascii
import operator
import numpy as np
import pandas as pd
import scapy.all as scapy
from scapy.all import rdpcap, TCP, UDP, IP, Raw
from scapy.layers.inet6 import IPv6
from scapy.layers.tls.all import TLSClientHello, TLSServerHello, TLS_Ext_ServerName, TLS_Ext_SupportedGroups
from scapy.layers.tls.record import TLS
from functools import partial, reduce
from flowcontainer.extractor import extract
import socket
import struct
import tempfile
import shutil
import subprocess
from application_field_parser import (
    build_flow_profile,
    encode_packet as encode_packet_application_fields,
    pieces_to_text,
)

word_dir = "/mnt/sdb/sunshine/corpora/SAMP1/"
word_name = "DoHMalious_128.txt"
error_log = "DoHMalious_128_errorlog.txt"
fine_tuning_flowdataset_dir = "/mnt/nfs/sunshine/MGPT/Fine-tuning/visquic_1000/"
fine_tuning_flowdataset_name = "visquic_1000.txt"
_buffer_finetuning_output = False
_finetuning_output_handle = None


def set_finetuning_output_buffering(enabled):
    """Enable buffered appends for an explicitly managed generation run."""
    global _buffer_finetuning_output
    _buffer_finetuning_output = bool(enabled)


def close_finetuning_output():
    """Flush and close the optional buffered fine-tuning output handle."""
    global _finetuning_output_handle
    if _finetuning_output_handle is not None:
        _finetuning_output_handle.flush()
        _finetuning_output_handle.close()
        _finetuning_output_handle = None


def _append_finetuning_line(line):
    """Append one line, preserving the historical open-per-line default."""
    global _finetuning_output_handle
    output_path = fine_tuning_flowdataset_dir + fine_tuning_flowdataset_name
    if not _buffer_finetuning_output:
        with open(output_path, 'a') as output:
            output.write(line)
        return
    if _finetuning_output_handle is None:
        _finetuning_output_handle = open(output_path, 'a', buffering=1024 * 1024)
    _finetuning_output_handle.write(line)

#Split Pcap
def split_cap(pcap_path, pcap_file, pcap_name, pcap_label='', dataset_level='flow'):
    split_pcap_path=os.path.join(pcap_path, "splitcap")
    if not os.path.exists(split_pcap_path):
        os.mkdir(split_pcap_path)
    if pcap_label != '':
        output_path = os.path.join(split_pcap_path, pcap_label)
    else:
        output_path = split_pcap_path
    if dataset_level == 'flow':
        cmd = f"wine /home/sunshine/Projects/ET-BERT-main/SplitCap.exe -r {pcap_file} -s session -o {output_path}"
    elif dataset_level == 'packet':
        cmd = f"wine /home/sunshine/Projects/ET-BERT-main/SplitCap.exe -r {pcap_file} -s packets 1 -o {output_path}"
    os.system(cmd)
    return output_path
    
    # # 原有的最终输出路径逻辑
    # split_pcap_path = os.path.join(pcap_path, "splitcap")
    # if not os.path.exists(split_pcap_path):
    #     os.mkdir(split_pcap_path)
    # if pcap_label != '':
    #     final_output_path = os.path.join(split_pcap_path, pcap_label)
    # else:
    #     final_output_path = split_pcap_path
    
    # # 确保最终输出目录存在
    # os.makedirs(final_output_path, exist_ok=True)
    
    # # 使用极简路径（在根目录下）
    # temp_dir = "/tmp/sc"
    # temp_input = "/tmp/sc/a.pcap"  # 极简文件名
    # temp_output = "/tmp/sc/o"     # 极简输出目录
    
    # try:
    #     # 清理并创建临时目录
    #     if os.path.exists(temp_dir):
    #         shutil.rmtree(temp_dir)
    #     os.makedirs(temp_dir, exist_ok=True)
    #     os.makedirs(temp_output, exist_ok=True)
        
    #     # 复制文件到极简路径
    #     shutil.copy2(pcap_file, temp_input)
        
    #     # 构造SplitCap命令（使用极简路径）
    #     if dataset_level == 'flow':
    #         cmd = f'wine /home/sunshine/Projects/ET-BERT-main/SplitCap.exe -r "{temp_input}" -s session -o "{temp_output}"'
    #     elif dataset_level == 'packet':
    #         cmd = f'wine /home/sunshine/Projects/ET-BERT-main/SplitCap.exe -r "{temp_input}" -s packets 1 -o "{temp_output}"'
        
    #     # 执行命令
    #     os.system(cmd)
        
    #     # 移动输出文件到最终位置
    #     moved_count = 0
    #     if os.path.exists(temp_output):
    #         for file in os.listdir(temp_output):
    #             if file.endswith('.pcap'):
    #                 src = os.path.join(temp_output, file)
    #                 dst = os.path.join(final_output_path, file)
    #                 shutil.move(src, dst)
    #                 moved_count += 1
        
    #     print(f"Moved {moved_count} output files to {final_output_path}")
        
    # except Exception as e:
    #     print(f"Error in split_cap: {e}")
        
    # finally:
    #     # 清理临时目录
    #     if os.path.exists(temp_dir):
    #         shutil.rmtree(temp_dir)
    
    # return final_output_path
    

# Check for private_ip
def is_private_ip(ip):
    private_ranges = [
        ('10.0.0.0', '10.255.255.255'),     # 10/8
        ('172.16.0.0', '172.31.255.255'),   # 172.16/12
        ('192.168.0.0', '192.168.255.255'), # 192.168/16
        ('100.64.0.0', '100.127.255.255'),  # 100.64/10 (CGN)
        ('169.254.0.0', '169.254.255.255')  # 169.254/16 (Link-local)
    ]
    
    # convert ip into int
    octets = ip.split('.')
    ip_int = (int(octets[0]) << 24) + (int(octets[1]) << 16) + \
             (int(octets[2]) << 8) + int(octets[3])
    
    # check whether private
    for start_range, end_range in private_ranges:
        start_octets = start_range.split('.')
        start_int = (int(start_octets[0]) << 24) + (int(start_octets[1]) << 16) + \
                   (int(start_octets[2]) << 8) + int(start_octets[3])
        
        end_octets = end_range.split('.')
        end_int = (int(end_octets[0]) << 24) + (int(end_octets[1]) << 16) + \
                 (int(end_octets[2]) << 8) + int(end_octets[3])
        
        if start_int <= ip_int <= end_int:
            return True
    
    return False

def determine_packet_directions(packets):
    """
    Determine packet directions based on the direction of the first packet.
    """
    directions = [0] * len(packets)
    
    # Obtain the first packet with IP layer
    first_ip_packet = None
    for packet in packets:
        if IP in packet:
            first_ip_packet = packet
            break
    
    # If no packet with IP, remark unknown.
    if not first_ip_packet:
        return directions
    
    # Obtain the first IP
    first_src_ip = first_ip_packet[IP].src
    first_dst_ip = first_ip_packet[IP].dst
    
    # Check TCP SYN for accurate direction.
    syn_packet = None
    for packet in packets:
        if IP in packet and TCP in packet and packet[TCP].flags & 0x02:  # SYN
            if not packet[TCP].flags & 0x10:  # make sure not SYN-ACK
                syn_packet = packet
                break
    
    if syn_packet:
        # SYN Packet usually starts from Client
        local_ip = syn_packet[IP].src
        remote_ip = syn_packet[IP].dst
        print(f"Use TCP SYN: local_ip={local_ip}, remote_ip={remote_ip}")
    else:
        # No SYN, use the fist packet IP
        src_is_private = is_private_ip(first_src_ip)
        dst_is_private = is_private_ip(first_dst_ip)
        
        if src_is_private and not dst_is_private:
            # from local to remote
            local_ip = first_src_ip
            remote_ip = first_dst_ip
            print(f"Determine direction by IP: local_ip={local_ip}, remote_ip={remote_ip}")
        elif not src_is_private and dst_is_private:
            # from remote to local
            local_ip = first_dst_ip
            remote_ip = first_src_ip
            print(f"Determine direction by IP: local_ip={local_ip}, remote_ip={remote_ip}")
        else:
            # the first packet is egress.
            local_ip = first_src_ip
            remote_ip = first_dst_ip
            print(f"Determine direction by the first packet: local_ip={local_ip}, remote_ip={remote_ip}")
    
    # Get the number of different direction
    in_count = 0
    out_count = 0
    unknown_count = 0
    
    for i, packet in enumerate(packets):
        direction = 0
        
        if IP in packet:
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            
            if src_ip == local_ip and dst_ip == remote_ip:
                direction = 1
                out_count += 1
            elif src_ip == remote_ip and dst_ip == local_ip:
                direction = -1
                in_count += 1
            else:
                unknown_count += 1
        else:
            unknown_count += 1
        
        directions[i] = direction
    
    total = len(packets)
    print(f"Determine directions finish: ")
    print(f"  ingress: {in_count} ({in_count/total*100:.1f}%)")
    print(f"  egress: {out_count} ({out_count/total*100:.1f}%)")
    print(f"  unknown: {unknown_count} ({unknown_count/total*100:.1f}%)")
    
    # Whether normal distribution:
    if in_count == 0 and out_count > 0:
        print("Warning: Only packets of egress.")
    elif out_count == 0 and in_count > 0:
        print("Warning: Only packets of ingress.")
    
    return directions


def _read_first_n_packets(label_pcap, max_packets):
    """
    Stream-read at most `max_packets` packets from a pcap.
    This avoids loading entire large files into memory.
    """
    packets = []
    if max_packets is None or max_packets <= 0:
        return packets
    try:
        with scapy.PcapReader(label_pcap) as reader:
            for packet in reader:
                packets.append(packet)
                if len(packets) >= max_packets:
                    break
    except Exception as e:
        print(f"读取PCAP文件出错: {e}")
        return []
    return packets

def cut(obj, sec):
    """Split hex string into chunks (from Baseline dataset_generation.py)."""
    result = [obj[i:i + sec] for i in range(0, len(obj), sec)]
    try:
        remanent_count = len(result[0]) % 4
    except Exception:
        remanent_count = 0
        print("cut datagram error!")
    if remanent_count != 0:
        result = [obj[i:i + sec + remanent_count] for i in range(0, len(obj), sec + remanent_count)]
    return result


def bigram_generation(packet_datagram, packet_len=64, flag=True):
    """Bi-byte tokenization (from Baseline dataset_generation.py get_feature_packet)."""
    result = ''
    generated_datagram = cut(packet_datagram, 1)
    token_count = 0
    for sub_string_index in range(len(generated_datagram)):
        if sub_string_index != (len(generated_datagram) - 1):
            token_count += 1
            if token_count > packet_len:
                break
            merge_word_bigram = (
                generated_datagram[sub_string_index] + generated_datagram[sub_string_index + 1]
            )
        else:
            break
        result += merge_word_bigram
        result += ' '
    return result


def encode_packet_bigram(packet, payload_len):
    """
    Encode a single packet via bi-byte splitting (Baseline get_feature_packet style).
    Skips the first 76 hex chars (link-layer header) then bigram-tokenizes.
    """
    packet_data = packet.copy()
    data = binascii.hexlify(bytes(packet_data))
    packet_string = data.decode()
    new_packet_string = packet_string[76:]
    return bigram_generation(new_packet_string, packet_len=payload_len, flag=True)


def cutPacket(packet_info_string, payload_len):
    non_space_count = 0
    result = []
    # 遍历字符串中的每个字符
    for char in packet_info_string:
        if char != ' ':  # 检查字符是否不是空格
            result.append(char)
            non_space_count += 1
        else:
            result.append(char)
        if non_space_count == payload_len:  # 如果已经收集了一定长度的非空格字符，则停止循环
            break

    # 将结果列表转换为字符串
    packet_txt = ''.join(result)
    return packet_txt


def splitto4(options_hex):
    options_split = ' '.join(options_hex[i:i + 4] for i in range(0, len(options_hex), 4))
    return options_split


def degrade_parsed_fields_to_bibytes(
    field_values,
    field_names,
    probability=0.0,
    rng=None,
    stats=None,
):
    """Probabilistically replace parsed fields with non-overlapping 2-byte tokens.

    Only fields wider than two bytes are eligible because shorter fields are
    already representable by a single bi-byte token.  ``rng`` is deliberately
    separate from the dataset sampling RNG, so changing the degradation rate
    cannot change which PCAPs are selected for an experiment.
    """
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("field degradation probability must be in [0, 1]")
    if probability == 0.0:
        return list(field_values)
    if rng is None:
        raise ValueError("an explicit RNG is required for field degradation")

    degraded = []
    for name, value in zip(field_names, field_values):
        # Parsed scalar fields contain no spaces. Options and payload are
        # handled separately and already use historical two-byte splitting.
        eligible = " " not in value and len(value) > 4
        if not eligible:
            degraded.append(value)
            continue

        if stats is not None:
            stats["eligible_fields"] = stats.get("eligible_fields", 0) + 1
            by_field = stats.setdefault("eligible_by_field", {})
            by_field[name] = by_field.get(name, 0) + 1

        if rng.random() < probability:
            value = splitto4(value)
            if stats is not None:
                stats["degraded_fields"] = stats.get("degraded_fields", 0) + 1
                by_field = stats.setdefault("degraded_by_field", {})
                by_field[name] = by_field.get(name, 0) + 1
        degraded.append(value)
    return degraded

def format_tcp_tls_fields(protocol, field_name, hex_value):
    """
    根据协议和字段名称决定是否进行双字节切分
    
    参数:
        protocol: 'TCP' 或 'TLS'
        field_name: 字段名称
        hex_value: 十六进制字符串值
        
    返回:
        处理后的字符串
    """
    # 不需要切分的TCP字段
    tcp_no_split_fields = {
        'sport', 'dport', 'flags', 'window', 'chksum', 'urgptr', 
        'options_type', 'options_length'
    }
    
    # 不需要切分的TLS字段
    tls_no_split_fields = {
        'type', 'version', 'length', 'msgtype', 'msglen', 'protocol_version',
        'sid_length', 'cipher_suite', 'comp_method', 'ext_type', 'ext_length',
        'cert_length', 'cert_type', 'ticket_lifetime', 'ticket_age_add'
    }
    
    # 检查是否为不需要切分的字段
    if (protocol == 'TCP' and field_name in tcp_no_split_fields) or \
       (protocol == 'TLS' and field_name in tls_no_split_fields):
        return hex_value
    
    # 对于其他字段，特别是较长的数据，进行双字节切分
    if len(hex_value) > 4:  # 如果长度超过4个字符(2字节)
        # 确保字符串长度是偶数
        if len(hex_value) % 2 != 0:
            hex_value = '0' + hex_value
            
        # 按4个字符(2字节)切分
        return ' '.join([hex_value[i:i+4] for i in range(0, len(hex_value), 4)])
    
    # 对于短字段，不进行切分
    return hex_value

def format_sni_extension(ext, tls_packet_info):
    """格式化SNI扩展"""
    try:
        if isinstance(ext.val, bytes):
            sni_hex = ext.val.hex()
            # SNI通常包含可读的域名，但仍按双字节切分以保持一致性
            formatted_sni = format_tcp_tls_fields('TLS', 'sni_data', sni_hex)
            tls_packet_info.append(formatted_sni)
    except Exception:
        pass

def format_supported_groups_extension(ext, tls_packet_info):
    """格式化supported_groups扩展"""
    try:
        if hasattr(ext, 'groups'):
            groups_list = []
            for group in ext.groups:
                if isinstance(group, int):
                    groups_list.append(f"{group:04x}")
            if groups_list:
                # 每个组是2字节，保持完整
                tls_packet_info.append(' '.join(groups_list))
        elif isinstance(ext.val, bytes):
            # 尝试从原始字节中提取
            groups_hex = ext.val.hex()
            # 第一个2字节是长度，之后每个组是2字节
            if len(groups_hex) >= 4:
                length_hex = groups_hex[:4]
                tls_packet_info.append(length_hex)
                
                groups_data = groups_hex[4:]
                # 每个组是2字节，所以按4个字符分组
                formatted_groups = ' '.join([groups_data[i:i+4] 
                                           for i in range(0, len(groups_data), 4)])
                tls_packet_info.append(formatted_groups)
    except Exception:
        pass

def format_ec_point_formats_extension(ext, tls_packet_info):
    """格式化ec_point_formats扩展"""
    try:
        if isinstance(ext.val, bytes):
            formats_hex = ext.val.hex()
            # 第一个字节是长度，之后每个格式是1字节
            if len(formats_hex) >= 2:
                length_hex = formats_hex[:2]
                tls_packet_info.append(length_hex)
                
                formats_data = formats_hex[2:]
                # 每个格式是1字节，所以按2个字符分组
                formatted_formats = ' '.join([formats_data[i:i+2] 
                                            for i in range(0, len(formats_data), 2)])
                tls_packet_info.append(formatted_formats)
    except Exception:
        pass

def format_signature_algorithms_extension(ext, tls_packet_info):
    """格式化signature_algorithms扩展"""
    try:
        if isinstance(ext.val, bytes):
            sig_algs_hex = ext.val.hex()
            # 第一个2字节是长度，之后每个算法是2字节
            if len(sig_algs_hex) >= 4:
                length_hex = sig_algs_hex[:4]
                tls_packet_info.append(length_hex)
                
                sig_algs_data = sig_algs_hex[4:]
                # 每个算法是2字节，所以按4个字符分组
                formatted_sig_algs = ' '.join([sig_algs_data[i:i+4] 
                                             for i in range(0, len(sig_algs_data), 4)])
                tls_packet_info.append(formatted_sig_algs)
    except Exception:
        pass

def format_supported_versions_extension(ext, tls_packet_info):
    """格式化supported_versions扩展 (TLS 1.3)"""
    try:
        if hasattr(ext, 'versions'):
            versions_list = []
            for version in ext.versions:
                if isinstance(version, int):
                    versions_list.append(f"{version:04x}")
            if versions_list:
                # 每个版本是2字节，保持完整
                tls_packet_info.append(' '.join(versions_list))
        elif isinstance(ext.val, bytes):
            # 尝试从原始字节中提取
            versions_hex = ext.val.hex()
            # ClientHello中，第一个字节是长度，之后每个版本是2字节
            # ServerHello中，直接是2字节的选定版本
            if len(versions_hex) >= 2:
                if len(versions_hex) == 4:  # 可能是ServerHello
                    tls_packet_info.append(versions_hex)
                else:  # 可能是ClientHello
                    length_hex = versions_hex[:2]
                    tls_packet_info.append(length_hex)
                    
                    versions_data = versions_hex[2:]
                    # 每个版本是2字节，所以按4个字符分组
                    formatted_versions = ' '.join([versions_data[i:i+4] 
                                                 for i in range(0, len(versions_data), 4)])
                    tls_packet_info.append(formatted_versions)
    except Exception:
        pass

def format_key_share_extension(ext, tls_packet_info):
    """格式化key_share扩展 (TLS 1.3)"""
    try:
        if isinstance(ext.val, bytes):
            key_share_hex = ext.val.hex()
            
            # ClientHello中，前2字节是长度，然后是(组+长度+密钥数据)重复
            # ServerHello中，是2字节组ID + 2字节长度 + 密钥数据
            
            # 简化处理：提取前4字节(可能是长度+组ID)，其余按双字节切分
            if len(key_share_hex) >= 4:
                header_hex = key_share_hex[:4]
                tls_packet_info.append(header_hex)
                
                if len(key_share_hex) > 4:
                    key_data = key_share_hex[4:]
                    # 对密钥数据应用双字节切分
                    formatted_key_data = format_tcp_tls_fields('TLS', 'key_share_data', key_data)
                    tls_packet_info.append(formatted_key_data)
    except Exception:
        pass

def extract_tls13_message_fields(msg, tls_packet_info):
    """从TLS 1.3特定消息中提取字段"""
    try:
        # 提取可能存在的任何字段
        for field_name, field_type in [
            ('cert_type', 'cert_type'),
            ('sig_alg', 'sig_alg'),
            ('sig', 'signature'),
            ('verify_data', 'verify_data'),
            ('ticket_lifetime', 'ticket_lifetime'),
            ('ticket_age_add', 'ticket_age_add')
        ]:
            if hasattr(msg, field_name):
                field_value = getattr(msg, field_name)
                if isinstance(field_value, bytes):
                    field_hex = field_value.hex()
                    formatted_field = format_tcp_tls_fields('TLS', field_type, field_hex)
                    tls_packet_info.append(formatted_field)
                elif isinstance(field_value, int):
                    field_hex = f"{field_value:x}"
                    formatted_field = format_tcp_tls_fields('TLS', field_type, field_hex)
                    tls_packet_info.append(formatted_field)
        
        # 提取消息体（如果有）
        if hasattr(msg, 'data') and isinstance(msg.data, bytes):
            data_hex = msg.data.hex()
            formatted_data = format_tcp_tls_fields('TLS', 'message_data', data_hex)
            tls_packet_info.append(formatted_data)
    except Exception:
        pass

def process_tls_extensions(msg, packet, tls_packet_info, is_tls13):
    """处理TLS扩展部分"""
    try:
        # 首先尝试从msg.ext直接提取扩展信息
        if hasattr(msg, 'ext') and msg.ext:
            for ext in msg.ext:
                if hasattr(ext, 'type'):
                    ext_type_hex = f"{ext.type:04x}"
                    tls_packet_info.append(ext_type_hex)
                    
                    # 扩展长度
                    if hasattr(ext, 'len'):
                        ext_len_hex = f"{ext.len:04x}"
                        tls_packet_info.append(ext_len_hex)
                    
                    # 扩展值
                    if hasattr(ext, 'val'):
                        # 根据扩展类型决定格式化方式
                        if ext.type == 0:  # SNI
                            format_sni_extension(ext, tls_packet_info)
                        elif ext.type == 10:  # supported_groups
                            format_supported_groups_extension(ext, tls_packet_info)
                        elif ext.type == 11:  # ec_point_formats
                            format_ec_point_formats_extension(ext, tls_packet_info)
                        elif ext.type == 13:  # signature_algorithms
                            format_signature_algorithms_extension(ext, tls_packet_info)
                        elif ext.type == 43 and is_tls13:  # supported_versions (TLS 1.3)
                            format_supported_versions_extension(ext, tls_packet_info)
                        elif ext.type == 51 and is_tls13:  # key_share (TLS 1.3)
                            format_key_share_extension(ext, tls_packet_info)
                        else:
                            # 通用扩展值处理
                            if isinstance(ext.val, bytes):
                                ext_val_hex = ext.val.hex()
                                formatted_val = format_tcp_tls_fields('TLS', 'ext_data', ext_val_hex)
                                tls_packet_info.append(formatted_val)
            return
        
        # 如果没有结构化的扩展信息，尝试从原始字节中提取
        if hasattr(msg, 'extlen') and msg.extlen is not None and msg.extlen > 0:
            try:
                # 尝试从数据包中提取扩展数据
                tls_header = bytes(packet[TLS])
                # 这里的逻辑需要针对特定握手消息类型调整
                # 简化处理：尝试获取扩展区域的数据
                if len(tls_header) > 5:  # TLS记录头为5字节
                    # 提取可能的扩展数据
                    # 注意：这种方法是近似的，可能不够准确
                    ext_data_size = min(msg.extlen, 512)  # 限制最大提取长度
                    # 扩展通常在握手消息的末尾
                    ext_data = tls_header[-ext_data_size:]
                    ext_hex = ext_data.hex()
                    formatted_ext = format_tcp_tls_fields('TLS', 'extensions', ext_hex)
                    tls_packet_info.append(formatted_ext)
            except Exception:
                pass
    except Exception:
        pass

# Extract payload
def extract_payload_from_ip(packet):
    if not packet.haslayer(IP):
        return b""
    try:
        raw_packet=bytes(packet)
        ip_layer=packet[IP]

        ip_total_len=ip_layer.len
        ip_header_len=ip_layer.ihl*4
        
        if TCP in packet:
            transport_header_len=packet[TCP].dataofs*4
        elif UDP in packet:
            transport_header_len=8
        else:
            return b""
        # Calculate payload length
        payload_len=ip_total_len-ip_header_len-transport_header_len
        if payload_len<=0:
            return b""
        
        # Locate the position of IP
        ip_start=14
        if ip_start+ip_total_len>len(raw_packet):
            return b""
        
        # Extract payload
        payload_start=ip_start+ip_header_len+transport_header_len
        payload_end=ip_start+ip_total_len

        return raw_packet[payload_start:payload_end]
    except:
        return b""


def extract_payload_from_ipv6(packet):
    """Extract TCP/UDP application payload from an IPv6 packet.

    This is an additive IPv6 path.  The original IPv4 helper above deliberately
    remains unchanged so existing ``packet_encoding_mode='field'`` experiments
    retain byte-for-byte preprocessing behavior.
    """
    if IPv6 not in packet:
        return b""
    try:
        if TCP in packet:
            return bytes(packet[TCP].payload)
        if UDP in packet:
            return bytes(packet[UDP].payload)
    except Exception:
        return b""
    return b""


def extract_payload_l2_agnostic(packet):
    """Extract TCP/UDP payload without assuming a link-layer header length.

    Scapy has already located the transport layer according to the PCAP
    linktype, so this works for Ethernet, raw-IP, Linux cooked captures, and
    IPv4/IPv6.  Reported protocol lengths are used to avoid including trailing
    padding when they are available.
    """
    try:
        if UDP in packet:
            payload = bytes(packet[UDP].payload)
            udp_len = int(packet[UDP].len)
            expected_len = udp_len - 8
            if expected_len <= 0:
                return b""
            return payload[:expected_len]

        if TCP in packet:
            payload = bytes(packet[TCP].payload)
            if IP in packet:
                expected_len = (
                    int(packet[IP].len)
                    - int(packet[IP].ihl) * 4
                    - int(packet[TCP].dataofs) * 4
                )
                if expected_len <= 0:
                    return b""
                return payload[:expected_len]
            return payload
    except Exception:
        return b""
    return b""


def determine_packet_directions_ipv4_ipv6(packets):
    """Add IPv6 directions while preserving legacy IPv4 directions exactly.

    Calling the historical IPv4 helper first is intentional: its private-IP
    fallback affects the sign when a capture has no client SYN.  Reusing it
    prevents the opt-in parser from changing existing IPv4 metadata.  IPv6 is
    inferred independently so mixed-address-family captures remain defined.
    """
    directions = determine_packet_directions(packets)
    first_packet = next((packet for packet in packets if IPv6 in packet), None)
    if first_packet is None:
        return directions

    local_ip = first_packet[IPv6].src
    remote_ip = first_packet[IPv6].dst
    for packet in packets:
        if IPv6 not in packet:
            continue
        src = packet[IPv6].src
        dst = packet[IPv6].dst
        if TCP in packet and packet[TCP].flags & 0x02 and not packet[TCP].flags & 0x10:
            local_ip, remote_ip = src, dst
            break

    for index, packet in enumerate(packets):
        if IPv6 not in packet:
            continue
        src = packet[IPv6].src
        dst = packet[IPv6].dst
        if src == local_ip and dst == remote_ip:
            directions[index] = 1
        elif src == remote_ip and dst == local_ip:
            directions[index] = -1
    return directions


def encode_packet_fields_ipv4_ipv6(
    packet,
    payload_len,
    l2safe_payload=False,
    zero_length_fields=False,
):
    """Encode one packet using the legacy IPv4 schema plus a new IPv6 schema.

    ``zero_length_fields`` is an opt-in ablation switch.  It replaces only
    protocol-header fields that directly encode a byte/header length while
    retaining their original token widths.  The real header lengths are still
    used internally to locate options and payload bytes correctly.
    """
    packet_info = []

    if IP in packet:
        ip_layer = packet[IP]
        ip_fields = [
            f"{ip_layer.version:01x}",
            "0" if zero_length_fields else f"{ip_layer.ihl:01x}",
            f"{ip_layer.tos:02X}",
            "0000" if zero_length_fields else f"{ip_layer.len:04x}",
            f"{ip_layer.id:04x}",
            f"{int(ip_layer.flags):01x}",
            f"{ip_layer.ttl:02x}",
            f"{ip_layer.proto:02x}",
            f"{ip_layer.chksum:04x}",
        ]
        if ip_layer.options and ip_layer.ihl * 4 > 20:
            ip_fields.append(splitto4(bytes(ip_layer)[20:ip_layer.ihl * 4].hex()))
        packet_info.extend(ip_fields)
    elif IPv6 in packet:
        ipv6_layer = packet[IPv6]
        # IPv6 fixed header: version, traffic class, flow label, payload
        # length, next-header identifier, and hop limit.  Addresses are omitted
        # to match the legacy IPv4 encoder's address-independent representation.
        packet_info.extend([
            f"{ipv6_layer.version:01x}",
            f"{ipv6_layer.tc:02x}",
            f"{ipv6_layer.fl:05x}",
            "0000" if zero_length_fields else f"{ipv6_layer.plen:04x}",
            f"{ipv6_layer.nh:02x}",
            f"{ipv6_layer.hlim:02x}",
        ])

    if TCP in packet:
        tcp_layer = packet[TCP]
        tcp_fields = [
            f"{tcp_layer.seq:08x}",
            f"{tcp_layer.ack:08x}",
            "0" if zero_length_fields else f"{tcp_layer.dataofs:01x}",
            f"{tcp_layer.reserved:02x}",
            f"{int(tcp_layer.flags):03x}",
            f"{tcp_layer.window:04x}",
        ]
        if tcp_layer.chksum is not None:
            tcp_fields.append(f"{tcp_layer.chksum:04x}")
        tcp_fields.append(f"{tcp_layer.urgptr:04x}")
        if tcp_layer.options and tcp_layer.dataofs * 4 > 20:
            tcp_fields.append(splitto4(bytes(tcp_layer)[20:tcp_layer.dataofs * 4].hex()))
        packet_info.extend(tcp_fields)
    elif UDP in packet:
        udp_layer = packet[UDP]
        udp_length = "0000" if zero_length_fields else f"{udp_layer.len:04x}"
        packet_info.extend([udp_length, f"{udp_layer.chksum:04x}"])

    if l2safe_payload:
        payload_data = extract_payload_l2_agnostic(packet)
    else:
        payload_data = (
            extract_payload_from_ip(packet)
            if IP in packet
            else extract_payload_from_ipv6(packet)
        )
    if payload_data:
        payload_hex = binascii.hexlify(payload_data).decode("utf-8")
        packet_info.append(" ".join(payload_hex[i:i + 4] for i in range(0, len(payload_hex), 4)))
    return cutPacket(" ".join(packet_info), payload_len)

# Generate pre-training corpara
def get_pretraining_corpara(label_pcap, flow_order, payload_len):
    # Packet Scapy
    try:
        all_packets = scapy.rdpcap(label_pcap)
        print(f"Read {len(all_packets)} packets from {label_pcap}")
        if len(all_packets)==0:
            print(f"No packets from {label_pcap}")
            message = f"No packets from {label_pcap}"
            with open(word_dir + error_log, 'a') as f:
                f.write(message + '\n')
            return 00
    except Exception as e:
        print(f"Read {label_pcap} fail! {e}")
        message = f"Read {label_pcap} fail! {e}"
        with open(word_dir + error_log, 'a') as f:
            f.write(message + '\n')
        return 0
    
    # Determine packet-length using the first packet
    packet_direction = determine_packet_directions(all_packets)
    packets = all_packets[:128]  # 限制每个流只解析前128个数据包
    packet_direction = packet_direction[:128]  # 同步限制方向数组
    
    # Packet Parsing
    for packet_num, packet in enumerate(packets):
        if packet_num >= len(packet_direction):
            continue
            
        transport_layer = None
        packet_info_string = ''
        
        # Get the packet direction
        direction = packet_direction[packet_num]
        
        # Get the packet length
        packet_length = len(packet)

        # Combine packet direction and length
        packet_length_dir = direction * packet_length

        # Get packet intervals
        if packet_num==0:
            time_interval=0.0
        else:
            time_interval=packet.time-pre_packet_time
        
        pre_packet_time = packet.time

        # Get the packet lay content
        ip_packet_info = []
        tcp_packet_info = []
        udp_packet_info = []
        packet_info_string = ''


        # IP layer
        if IP in packet:
            ip_layer = packet[IP]
            ip_packet_info.append(f"{ip_layer.version:01x}")
            ip_packet_info.append(f"{ip_layer.ihl:01x}")
            ip_packet_info.append(f"{ip_layer.tos:02X}")
            ip_packet_info.append(f"{ip_layer.len:04x}")
            ip_packet_info.append(f"{ip_layer.id:04x}")
            # ip_packet_info.append("0000")
            ip_packet_info.append(f"{int(ip_layer.flags):01x}")  # Convert FlagValue into int
            ip_packet_info.append(f"{ip_layer.ttl:02x}")
            ip_packet_info.append(f"{ip_layer.proto:02x}")
            ip_packet_info.append(f"{ip_layer.chksum:04x}")
            # ip_packet_info.append("0000")
            
            ip_packet_info_string = ' '.join(ip_packet_info)
            if ip_layer.options:
                ip_header = bytes(packet[IP])
                # TCP报头大小由数据偏移字段决定，单位是4字节
                ip_data_offset = (ip_header.ihl * 4)
                # 选项数据开始于TCP报头的第20字节
                if ip_data_offset > 20:
                    # 提取从第20字节开始到data_offset的字节
                    ip_options_data = ip_header[20:ip_data_offset]
                    options_hex = ip_options_data.hex()
                    ip_packet_info_string += ' ' + splitto4(options_hex)
            packet_info_string += ip_packet_info_string

        # TCP layer
        if TCP in packet:
            transport_layer = 'TCP'
            tcp_layer = packet[TCP]
           
            tcp_packet_info.append(f"{tcp_layer.seq:08x}")
            # tcp_packet_info.append("00000000")
            tcp_packet_info.append(f"{tcp_layer.ack:08x}")
            # tcp_packet_info.append("00000000")
            tcp_packet_info.append(f"{tcp_layer.dataofs:01x}")
            tcp_packet_info.append(f"{tcp_layer.reserved:02x}")
            tcp_packet_info.append(f"{int(tcp_layer.flags):03x}")
            tcp_packet_info.append(f"{tcp_layer.window:04x}")
            # tcp_packet_info.append("0000")
            if tcp_layer.chksum is not None:
                tcp_packet_info.append(f"{tcp_layer.chksum:04x}")
            tcp_packet_info.append(f"{tcp_layer.urgptr:04x}")
            tcp_packet_info_string = ' '.join(tcp_packet_info)

            if tcp_layer.options:
                tcp_header = bytes(packet[TCP])
                # TCP报头大小由数据偏移字段决定，单位是4字节
                tcp_data_offset = packet[TCP].dataofs * 4
                # 选项数据开始于TCP报头的第20字节
                if tcp_data_offset > 20:
                    # 提取从第20字节开始到data_offset的字节
                    tcp_options_data = tcp_header[20:tcp_data_offset]
                    # tcp_packet_info.append(f"{tcp_options_data.hex()}")
                    options_hex = tcp_options_data.hex()
                    tcp_packet_info_string += ' ' +splitto4(options_hex)
            packet_info_string += ' ' + tcp_packet_info_string

        # UDP layer
        elif UDP in packet:
            transport_layer = 'UDP'
            udp_layer = packet[UDP]

            udp_packet_info.append(f"{udp_layer.len:04x}")
            udp_packet_info.append(f"{udp_layer.chksum:04x}")
            udp_packet_info_string = ' '.join(udp_packet_info)
            packet_info_string += ' ' + udp_packet_info_string

        # Payload
        payload_data=extract_payload_from_ip(packet)
        if payload_data:
            payload_hex=binascii.hexlify(payload_data).decode('utf-8')
            payload_split=' '.join(payload_hex[i:i+4] for i in range(0, len(payload_hex), 4))
            packet_info_string+=' '+payload_split

        # Cut packet_info_string into fixed length.
        packet_txt = cutPacket(packet_info_string, payload_len)
        
        # Add to pre-training corpora.
        with open(word_dir + word_name, 'a') as f:
            f.write(f"{flow_order}\t{time_interval}\t{packet_length_dir}\t{packet_txt}\n")
    
    flow_order = flow_order+1
    return flow_order

def get_finetuning_flow_feature_text(
    label_pcap,
    category,
    flow_order,
    payload_len,
    payload_packet=128,
    validate_with_extract=False,
    zero_length_fields=False,
    field_degradation_probability=0.0,
    field_degradation_rng=None,
    field_degradation_stats=None,
):
    print(flow_order)

    # 检查文件是否为 PCAP 文件
    if not label_pcap.endswith('.pcap'):
        return -1
    
    # 添加文件存在性检查
    if not os.path.exists(label_pcap):
        print(f"文件不存在，跳过: {label_pcap}")
        return -1
    
    all_packets = _read_first_n_packets(label_pcap, payload_packet)
    if len(all_packets) == 0:
        print(f"读取PCAP文件出错或无可读包: {label_pcap}")
        return -1
    print(f"读取到 {len(all_packets)} 个数据包(已截断)，来自 {label_pcap}")
    
    # 2. 检查数据包数量
    if len(all_packets) < 3:
        print(f"preprocess flow {label_pcap} but this flow has less than 3 packets.")
        return -1
    
    # 3. 快速协议检查（默认不做 extract，可显著提速）
    has_l4_packet = any((TCP in p) or (UDP in p) for p in all_packets)
    if not has_l4_packet:
        print(f"preprocess flow {label_pcap} but this flow has no TCP/UDP packets.")
        return -1

    # 如需严格保持历史行为，可开启 extract 校验（较慢）。
    if validate_with_extract:
        feature_result = {}
        try:
            feature_result = extract(
                label_pcap,
                filter='tcp',
                extension=['tls.record.content_type', 'tls.record.opaque_type', 'tls.handshake.type']
            )
        except Exception as e:
            print(f"TCP特征提取失败: {label_pcap}, 错误: {e}")
        if not feature_result:
            try:
                feature_result = extract(label_pcap, filter='udp')
            except Exception as e:
                print(f"UDP特征提取失败: {label_pcap}, 错误: {e}")
                return -1
        for key, flow in feature_result.items():
            if len(flow.ip_lengths) < 3:
                print(f"preprocess flow {label_pcap} but this flow has less than 3 packets.")
                return -1
    
    # Determine packet-length using the first packet
    packet_direction = determine_packet_directions(all_packets)
    packets = all_packets
    packet_direction = packet_direction[:payload_packet]
    
    # Packet Parsing
    for packet_num, packet in enumerate(packets):
        if packet_num >= len(packet_direction):
            continue
            
        transport_layer = None
        packet_info_string = ''
        
        # Get the packet direction
        direction = packet_direction[packet_num]
        
        # Get the packet length
        packet_length = len(packet)

        # Combine packet direction and length
        packet_length_dir = direction * packet_length

        # Get packet intervals
        if packet_num==0:
            time_interval=0.0
        else:
            time_interval=packet.time-pre_packet_time
        
        pre_packet_time = packet.time

        # Get the packet lay content
        ip_packet_info = []
        tcp_packet_info = []
        udp_packet_info = []
        packet_info_string = ''


        # IP layer
        if IP in packet:
            ip_layer = packet[IP]
            ip_packet_info.append(f"{ip_layer.version:01x}")
            ip_packet_info.append("0" if zero_length_fields else f"{ip_layer.ihl:01x}")
            ip_packet_info.append(f"{ip_layer.tos:02X}")
            ip_packet_info.append("0000" if zero_length_fields else f"{ip_layer.len:04x}")
            ip_packet_info.append(f"{ip_layer.id:04x}")
            ip_packet_info.append(f"{int(ip_layer.flags):01x}")  # Convert FlagValue into int
            ip_packet_info.append(f"{ip_layer.ttl:02x}")
            ip_packet_info.append(f"{ip_layer.proto:02x}")
            ip_packet_info.append(f"{ip_layer.chksum:04x}")
            ip_packet_info = degrade_parsed_fields_to_bibytes(
                ip_packet_info,
                [
                    "ipv4.version", "ipv4.ihl", "ipv4.tos", "ipv4.total_length",
                    "ipv4.id", "ipv4.flags", "ipv4.ttl", "ipv4.protocol",
                    "ipv4.checksum",
                ],
                field_degradation_probability,
                field_degradation_rng,
                field_degradation_stats,
            )
            
            ip_packet_info_string = ' '.join(ip_packet_info)
            if ip_layer.options:
                ip_header = bytes(packet[IP])
                # TCP报头大小由数据偏移字段决定，单位是4字节
                ip_data_offset = (ip_header.ihl * 4)
                # 选项数据开始于TCP报头的第20字节
                if ip_data_offset > 20:
                    # 提取从第20字节开始到data_offset的字节
                    ip_options_data = ip_header[20:ip_data_offset]
                    options_hex = ip_options_data.hex()
                    ip_packet_info_string += ' ' + splitto4(options_hex)
            packet_info_string += ip_packet_info_string

        # TCP layer
        if TCP in packet:
            transport_layer = 'TCP'
            tcp_layer = packet[TCP]
           
            tcp_packet_info.append(f"{tcp_layer.seq:08x}")
            tcp_packet_info.append(f"{tcp_layer.ack:08x}")
            tcp_packet_info.append("0" if zero_length_fields else f"{tcp_layer.dataofs:01x}")
            tcp_packet_info.append(f"{tcp_layer.reserved:02x}")
            tcp_packet_info.append(f"{int(tcp_layer.flags):03x}")
            tcp_packet_info.append(f"{tcp_layer.window:04x}")
            if tcp_layer.chksum is not None:
                tcp_packet_info.append(f"{tcp_layer.chksum:04x}")
            tcp_packet_info.append(f"{tcp_layer.urgptr:04x}")
            tcp_field_names = [
                "tcp.sequence", "tcp.acknowledgment", "tcp.data_offset",
                "tcp.reserved", "tcp.flags", "tcp.window",
            ]
            if tcp_layer.chksum is not None:
                tcp_field_names.append("tcp.checksum")
            tcp_field_names.append("tcp.urgent_pointer")
            tcp_packet_info = degrade_parsed_fields_to_bibytes(
                tcp_packet_info,
                tcp_field_names,
                field_degradation_probability,
                field_degradation_rng,
                field_degradation_stats,
            )
            tcp_packet_info_string = ' '.join(tcp_packet_info)

            if tcp_layer.options:
                tcp_header = bytes(packet[TCP])
                # TCP报头大小由数据偏移字段决定，单位是4字节
                tcp_data_offset = packet[TCP].dataofs * 4
                # 选项数据开始于TCP报头的第20字节
                if tcp_data_offset > 20:
                    # 提取从第20字节开始到data_offset的字节
                    tcp_options_data = tcp_header[20:tcp_data_offset]
                    # tcp_packet_info.append(f"{tcp_options_data.hex()}")
                    options_hex = tcp_options_data.hex()
                    tcp_packet_info_string += ' ' +splitto4(options_hex)
            packet_info_string += ' ' + tcp_packet_info_string

        # UDP layer
        elif UDP in packet:
            transport_layer = 'UDP'
            udp_layer = packet[UDP]

            udp_packet_info.append("0000" if zero_length_fields else f"{udp_layer.len:04x}")
            udp_packet_info.append(f"{udp_layer.chksum:04x}")
            udp_packet_info = degrade_parsed_fields_to_bibytes(
                udp_packet_info,
                ["udp.length", "udp.checksum"],
                field_degradation_probability,
                field_degradation_rng,
                field_degradation_stats,
            )
            udp_packet_info_string = ' '.join(udp_packet_info)
            packet_info_string += ' ' + udp_packet_info_string

        # Payload
        payload_data=extract_payload_from_ip(packet)
        if payload_data:
            payload_hex=binascii.hexlify(payload_data).decode('utf-8')
            payload_split=' '.join(payload_hex[i:i+4] for i in range(0, len(payload_hex), 4))
            packet_info_string+=' '+payload_split

        # Cut packet_info_string into fixed length.
        packet_txt = cutPacket(packet_info_string, payload_len)

        _append_finetuning_line(
            f"{category}\t{flow_order}\t{time_interval}\t{packet_length_dir}\t{packet_txt}\n"
        )
    return 0


def get_finetuning_flow_feature_text_zero_length(
    label_pcap,
    category,
    flow_order,
    payload_len,
    payload_packet=128,
    validate_with_extract=False,
):
    """Legacy IPv4/TCP/UDP encoder with direct length fields zeroed."""
    return get_finetuning_flow_feature_text(
        label_pcap,
        category,
        flow_order,
        payload_len,
        payload_packet=payload_packet,
        validate_with_extract=validate_with_extract,
        zero_length_fields=True,
    )


def get_finetuning_flow_feature_text_ipv4_ipv6(
    label_pcap,
    category,
    flow_order,
    payload_len,
    payload_packet=128,
    validate_with_extract=False,
    l2safe_payload=False,
    zero_length_fields=False,
):
    """Opt-in field encoder that adds IPv6 without changing the legacy path."""
    print(flow_order)
    if not label_pcap.endswith('.pcap') or not os.path.exists(label_pcap):
        return -1

    all_packets = _read_first_n_packets(label_pcap, payload_packet)
    if len(all_packets) < 3:
        return -1
    if not any((TCP in packet) or (UDP in packet) for packet in all_packets):
        return -1

    # Retain the optional historical flowcontainer validation behavior.
    if validate_with_extract:
        feature_result = {}
        try:
            feature_result = extract(
                label_pcap,
                filter='tcp',
                extension=['tls.record.content_type', 'tls.record.opaque_type', 'tls.handshake.type'],
            )
        except Exception:
            pass
        if not feature_result:
            try:
                feature_result = extract(label_pcap, filter='udp')
            except Exception:
                return -1
        if any(len(flow.ip_lengths) < 3 for flow in feature_result.values()):
            return -1

    packet_directions = determine_packet_directions_ipv4_ipv6(all_packets)
    previous_time = None
    for packet_num, packet in enumerate(all_packets):
        direction = packet_directions[packet_num]
        packet_length_direction = direction * len(packet)
        if previous_time is None:
            time_interval = 0.0
        else:
            time_interval = packet.time - previous_time
        previous_time = packet.time

        packet_txt = encode_packet_fields_ipv4_ipv6(
            packet,
            payload_len,
            l2safe_payload=l2safe_payload,
            zero_length_fields=zero_length_fields,
        )
        _append_finetuning_line(
            f"{category}\t{flow_order}\t{time_interval}\t"
            f"{packet_length_direction}\t{packet_txt}\n"
        )
    return 0


def get_finetuning_flow_feature_text_ipv4_ipv6_l2safe(
    label_pcap,
    category,
    flow_order,
    payload_len,
    payload_packet=128,
    validate_with_extract=False,
):
    """Opt-in IPv4/IPv6 encoder with link-layer-independent payload parsing."""
    return get_finetuning_flow_feature_text_ipv4_ipv6(
        label_pcap,
        category,
        flow_order,
        payload_len,
        payload_packet=payload_packet,
        validate_with_extract=validate_with_extract,
        l2safe_payload=True,
    )


def get_finetuning_flow_feature_text_ipv4_ipv6_l2safe_zero_length(
    label_pcap,
    category,
    flow_order,
    payload_len,
    payload_packet=128,
    validate_with_extract=False,
):
    """Corrected IPv4/IPv6 encoder with direct length fields zeroed."""
    return get_finetuning_flow_feature_text_ipv4_ipv6(
        label_pcap,
        category,
        flow_order,
        payload_len,
        payload_packet=payload_packet,
        validate_with_extract=validate_with_extract,
        l2safe_payload=True,
        zero_length_fields=True,
    )


def get_finetuning_flow_feature_text_application(
    label_pcap,
    category,
    flow_order,
    payload_len,
    payload_packet=128,
    validate_with_extract=False,
):
    """Opt-in IPv4/IPv6/TCP/UDP plus TLS/QUIC field encoder.

    TLS/QUIC bytes that cannot be represented as fields are retained with the
    historical two-byte fallback.  Existing encoders remain unchanged.
    """
    print(flow_order)
    if not label_pcap.endswith('.pcap') or not os.path.exists(label_pcap):
        return -1

    all_packets = _read_first_n_packets(label_pcap, payload_packet)
    if len(all_packets) < 3:
        return -1
    if not any((TCP in packet) or (UDP in packet) for packet in all_packets):
        return -1

    if validate_with_extract:
        feature_result = {}
        try:
            feature_result = extract(
                label_pcap,
                filter='tcp',
                extension=['tls.record.content_type', 'tls.record.opaque_type', 'tls.handshake.type'],
            )
        except Exception:
            pass
        if not feature_result:
            try:
                feature_result = extract(label_pcap, filter='udp')
            except Exception:
                return -1
        if any(len(flow.ip_lengths) < 3 for flow in feature_result.values()):
            return -1

    packet_directions = determine_packet_directions_ipv4_ipv6(all_packets)
    profile = build_flow_profile(all_packets)
    previous_time = None
    for packet_num, packet in enumerate(all_packets):
        direction = packet_directions[packet_num]
        packet_length_direction = direction * len(packet)
        time_interval = 0.0 if previous_time is None else packet.time - previous_time
        previous_time = packet.time

        pieces, _ = encode_packet_application_fields(packet, profile, payload_len)
        packet_txt = pieces_to_text(pieces)
        with open(fine_tuning_flowdataset_dir + fine_tuning_flowdataset_name, 'a') as output:
            output.write(
                f"{category}\t{flow_order}\t{time_interval}\t"
                f"{packet_length_direction}\t{packet_txt}\n"
            )
    return 0


def get_finetuning_flow_feature_bigram(label_pcap, category, flow_order, payload_len, payload_packet=128, validate_with_extract=False):
    """
    Flow-level fine-tuning feature extraction with bi-byte packet encoding.
    Same flow metadata as get_finetuning_flow_feature_text; per-packet text_a uses
    Baseline dataset_generation.get_feature_packet style (skip 76 hex chars + bigram).
    """
    print(flow_order)

    if not label_pcap.endswith('.pcap'):
        return -1

    if not os.path.exists(label_pcap):
        print(f"文件不存在，跳过: {label_pcap}")
        return -1

    all_packets = _read_first_n_packets(label_pcap, payload_packet)
    if len(all_packets) == 0:
        print(f"读取PCAP文件出错或无可读包: {label_pcap}")
        return -1
    print(f"读取到 {len(all_packets)} 个数据包(已截断)，来自 {label_pcap}")

    if len(all_packets) < 3:
        print(f"preprocess flow {label_pcap} but this flow has less than 3 packets.")
        return -1

    has_l4_packet = any((TCP in p) or (UDP in p) for p in all_packets)
    if not has_l4_packet:
        print(f"preprocess flow {label_pcap} but this flow has no TCP/UDP packets.")
        return -1

    if validate_with_extract:
        feature_result = {}
        try:
            feature_result = extract(
                label_pcap,
                filter='tcp',
                extension=['tls.record.content_type', 'tls.record.opaque_type', 'tls.handshake.type']
            )
        except Exception as e:
            print(f"TCP特征提取失败: {label_pcap}, 错误: {e}")
        if not feature_result:
            try:
                feature_result = extract(label_pcap, filter='udp')
            except Exception as e:
                print(f"UDP特征提取失败: {label_pcap}, 错误: {e}")
                return -1
        for key, flow in feature_result.items():
            if len(flow.ip_lengths) < 3:
                print(f"preprocess flow {label_pcap} but this flow has less than 3 packets.")
                return -1

    packet_direction = determine_packet_directions(all_packets)
    packets = all_packets
    packet_direction = packet_direction[:payload_packet]

    pre_packet_time = None
    for packet_num, packet in enumerate(packets):
        if packet_num >= len(packet_direction):
            continue

        direction = packet_direction[packet_num]
        packet_length = len(packet)
        packet_length_dir = direction * packet_length

        if packet_num == 0:
            time_interval = 0.0
        else:
            time_interval = packet.time - pre_packet_time
        pre_packet_time = packet.time

        packet_txt = encode_packet_bigram(packet, payload_len)

        with open(fine_tuning_flowdataset_dir + fine_tuning_flowdataset_name, 'a') as f:
            f.write(f"{category}\t{flow_order}\t{time_interval}\t{packet_length_dir}\t{packet_txt}\n")
    return 0

# def get_finetuning_packet_text(label_pcap, payload_len):
#     # 检查文件是否为 PCAP 文件
#     if not label_pcap.endswith('.pcap'):
#         print(f"跳过非 PCAP 文件: {label_pcap}")
#         return []
    
#     feature_data = []

#     packets = scapy.rdpcap(label_pcap)
   
#     for packet_num, packet in enumerate(packets):
#         # 获取数据包内容信息
#         ip_packet_info = []
#         tcp_packet_info = []
#         udp_packet_info = []
#         tls_packet_info = []
#         packet_info_string = ''


#         # 检查数据包是否包含IP层
#         if IP in packet:
#             # 获取IP层
#             ip_layer = packet[IP]
#             # 添加原始字节形式的字段到packet_info列表
#             ip_packet_info.append(f"{ip_layer.version:01x}")
#             ip_packet_info.append(f"{ip_layer.ihl:01x}")
#             ip_packet_info.append(f"{ip_layer.tos:02X}")
#             ip_packet_info.append(f"{ip_layer.len:04x}")
#             ip_packet_info.append(f"{ip_layer.id:04x}")
#             ip_packet_info.append(f"{int(ip_layer.flags):01x}")  # 将FlagValue转换为整数后格式化,这种方式存在问题
#             ip_packet_info.append(f"{ip_layer.frag:04x}")
#             ip_packet_info.append(f"{ip_layer.ttl:02x}")
#             ip_packet_info.append(f"{ip_layer.proto:02x}")
#             ip_packet_info.append(f"{ip_layer.chksum:04x}")
#             # 对src和dst进行十六进制转换
#             # src_hex = binascii.hexlify(inet_aton(ip_layer.src)).decode()
#             # dst_hex = binascii.hexlify(inet_aton(ip_layer.dst)).decode()
#             # packet_info.append(f"src hex:{src_hex}[FIE]")
#             # packet_info.append(f"dst hex:{dst_hex}[FIE]")
#             ip_packet_info_string = ' '.join(ip_packet_info)
#             if ip_layer.options:
#                 ip_header = bytes(packet[IP])
#                 # TCP报头大小由数据偏移字段决定，单位是4字节
#                 ip_data_offset = (ip_header.ihl * 4)
#                 # 选项数据开始于TCP报头的第20字节
#                 if ip_data_offset > 20:
#                     # 提取从第20字节开始到data_offset的字节
#                     ip_options_data = ip_header[20:ip_data_offset]
#                     # packet_info.append(f"{ip_options_data.hex()}")
#                     options_hex = ip_options_data.hex()
#                     ip_packet_info_string += ' ' + splitto4(options_hex)
#             packet_info_string += ip_packet_info_string

#         # 检查是否包括TCP头
#         if TCP in packet:
#             transport_layer = 'TCP'
#             # tl_layer_fields = packet[TCP].fields
#             tcp_layer = packet[TCP]
#             # packet_info.append(f"sport:{tcp_layer.sport:04x}[FIE]")
#             # packet_info.append(f"dport:{tcp_layer.dport:04x}[FIE]")
#             tcp_packet_info.append(f"{tcp_layer.seq:08x}")
#             tcp_packet_info.append(f"{tcp_layer.ack:08x}")
#             tcp_packet_info.append(f"{tcp_layer.dataofs:01x}")
#             tcp_packet_info.append(f"{tcp_layer.reserved:02x}")
#             tcp_packet_info.append(f"{int(tcp_layer.flags):03x}")
#             tcp_packet_info.append(f"{tcp_layer.window:04x}")
#             if tcp_layer.chksum is not None:
#                 tcp_packet_info.append(f"{tcp_layer.chksum:04x}")
#             tcp_packet_info.append(f"{tcp_layer.urgptr:04x}")
#             tcp_packet_info_string = ' '.join(tcp_packet_info)
#             if tcp_layer.options:
#                 tcp_header = bytes(packet[TCP])
#                 # TCP报头大小由数据偏移字段决定，单位是4字节
#                 tcp_data_offset = packet[TCP].dataofs * 4
#                 # 选项数据开始于TCP报头的第20字节
#                 if tcp_data_offset > 20:
#                     # 提取从第20字节开始到data_offset的字节
#                     tcp_options_data = tcp_header[20:tcp_data_offset]
#                     # tcp_packet_info.append(f"{tcp_options_data.hex()}")
#                     options_hex = tcp_options_data.hex()
#                     tcp_packet_info_string += ' ' +splitto4(options_hex)
#             packet_info_string += ' ' + tcp_packet_info_string

#         elif UDP in packet:
#             transport_layer = 'UDP'
#             udp_layer = packet[UDP]
#             # udp_packet_info.append(f"{udp_layer.sport:04x}")
#             # udp_packet_info.append(f"{udp_layer.dport:04x}")
#             udp_packet_info.append(f"{udp_layer.len:04x}")
#             udp_packet_info.append(f"{udp_layer.chksum:04x}")
#             udp_packet_info_string = ' '.join(udp_packet_info)
#             packet_info_string += ' ' + udp_packet_info_string

#         # TLS协议解析
#         if transport_layer:
#             if transport_layer == 'TCP' and packet.haslayer(TLS):
#                 tls_layer = packet[TLS]

#                 # 基本TLS字段
#                 if hasattr(tls_layer, 'type'):
#                     tls_packet_info.append(f"{tls_layer.type:02x}")
#                 if hasattr(tls_layer, 'version'):
#                     tls_packet_info.append(f"{tls_layer.version:04x}")
#                 if hasattr(tls_layer, 'len'):
#                     tls_packet_info.append(f"{tls_layer.len:04x}")

#                 # 检测是否可能是TLS 1.3
#                 is_tls13 = False

#                 #处理TLS握手消息
#                 if hasattr(tls_layer, 'msg') and tls_layer.msg:
#                     for msg in tls_layer.msg:
#                         # 通用握手消息
#                         if hasattr(msg, 'msgtype') and msg.msgtype != None:
#                             tls_packet_info.append(f"{msg.msgtype:02x}")
#                         if hasattr(msg, 'msglen') and msg.msglen != None:
#                             tls_packet_info.append(f"{msg.msglen:06x}")

#                         # ClientHello/ServerHello处理
#                         if isinstance(msg, TLSClientHello) or isinstance(msg, TLSServerHello):
#                             # TLS版本
#                             if hasattr(msg, 'version') and msg.version != None:
#                                 tls_packet_info.append(f"{msg.version:04x}")
                            
#                             # 随机字节，应用双字节划分
#                             if hasattr(msg, 'random_bytes') and msg.random_bytes != None:
#                                 if isinstance(msg.random_bytes, bytes):
#                                     random_bytes_hex=msg.random_bytes.hex()
#                                     tls_packet_info.append(f"{format_tcp_tls_fields('TLS', 'random_bytes', random_bytes_hex)}")
#                                 else:
#                                     print(f"Unexpected random_bytes type: {type(msg.random_bytes)}")
                            
#                             #会话ID长度
#                             if hasattr(msg, 'sidlen') and msg.sidlen != None:
#                                 tls_packet_info.append(f"{msg.sidlen:02x}")
                            
#                             #会话ID
#                             if hasattr(msg, 'sid') and msg.sid:
#                                 if isinstance(msg.sid, bytes):
#                                     sid_hex = msg.sid.hex()
#                                     tls_packet_info.append(f"{format_tcp_tls_fields('TLS', 'sid', sid_hex)}")
                                
#                             # 密码套件长度
#                             if hasattr(msg, 'cipherlen') and msg.cipherlen != None:
#                                 tls_packet_info.append(f"{msg.cipherslen:04x}")

#                             TLS13_SUITES = {1301, 1302, 1303, 1304, 1305}
#                             # 密码套件
#                             if hasattr(msg, 'ciphers') and msg.ciphers is not None:
#                                 cipher_hex_list = []
#                                 for cipher in msg.ciphers:
#                                     cipher_value = None
#                                     cipher_hex = None
#                                     if isinstance(cipher, bytes) and len(cipher)==2:
#                                         cipher_value=int.from_bytes(cipher,'big')
#                                     elif isinstance(cipher, int):
#                                         cipher_value = cipher
#                                     elif isinstance(cipher, str) and cipher.startswith(('0x','0X')):
#                                         try:
#                                             cipher_value = int(cipher, 16)
#                                         except ValueError:
#                                             continue
#                                     else:
#                                         try:
#                                             cipher_value = int(cipher)
#                                         except(TypeError,ValueError):
#                                             continue
                                    
#                                     if cipher_value is not None:
#                                         cipher_hex = f"{cipher_value:04x}"
#                                         cipher_hex_list.append(cipher_hex)
#                                         if cipher_value in TLS13_SUITES:
#                                             is_tls13 = True

#                                     # if hasattr(cipher, bytes):
#                                     #     cipher_hex = cipher.hex()
#                                     #     cipher_hex_list.append(cipher_hex)

#                                     #     #检测是否为TLS1.3密码套件
#                                     #     if len(cipher_hex)>=4 and cipher_hex.startswith("13"):
#                                     #         is_tls13 = True
#                                     # else:
#                                     #     try:
#                                     #         cipher_value = f"{cipher:04x}"
#                                     #         cipher_hex_list.append(cipher_value)

#                                     #         #检测是否为TLS1.3密码套件
#                                     #         if cipher_value.startswith("13"):
#                                     #             is_tls13 = True
#                                     #     except(TypeError, ValueError):
#                                     #         pass
#                                 #使用空格分隔不同的密码套件
#                                 if cipher_hex_list:
#                                     formatted_ciphers = ' '.join(cipher_hex_list)
#                                     tls_packet_info.append(formatted_ciphers)
                            
#                             # 压缩方法长度
#                             if hasattr(msg, 'complen') and msg.complen is not None:
#                                 tls_packet_info.append(f"{msg.complen:02x}")

#                             if hasattr(msg, 'comp') and msg.comp is not None:
#                                 if isinstance(msg.comp, (list,tuple)):
#                                     if msg.comp == [0]:
#                                         tls_packet_info.append("00")
#                                     else:
#                                         comp_hex = ' '.join(f"{c:02x}" for c in msg.comp if isinstance(c,int))
#                                         if comp_hex:
#                                             tls_packet_info.append(comp_hex)
#                                 elif isinstance(msg.comp, int):
#                                     tls_packet_info.append(f"{msg.comp:02x}")

#                             # 扩展长度
#                             if hasattr(msg, 'extlen') and msg.extlen is not None:
#                                 tls_packet_info.append(f"{msg.extlen:04x}")

#                             # 处理扩展
#                             process_tls_extensions(msg, packet, tls_packet_info, is_tls13)

#                         # 处理TLS1.3特有消息类型
#                         elif hasattr(msg, 'msgtype'):
#                             # EncryptedExtensions (8), Certificate (11), CertificateVerify (15),  Finished (20), NewSessionTicket (4), KeyUpdate (24)
#                             if msg.msgtype in [4, 8, 11, 15, 20, 24]:
#                                 #版本信息
#                                 if hasattr(msg, 'version') and msg.version is not None:
#                                     tls_packet_info.append(f"{msg.version:04x}")
                                
#                                 #证书长度
#                                 if msg.msgtype == 11 and hasattr(msg, 'cert_len'):
#                                     tls_packet_info.append(f"{msg.cert_len:06x}")

#                                 # 提取其他可能的字段
#                                 extract_tls13_message_fields(msg,tls_packet_info)
#                 tls_packet_info_string = ' '.join(tls_packet_info)    
#                 packet_info_string += ' ' + tls_packet_info_string
#                 tls_data_processed = True
#             else:
#                 tls_data_processed = False
#         # 检查数据包中是否有负载 (Raw Layer)
#         if packet.haslayer('Raw') and tls_data_processed == False:
#             # 获取负载数据
#             payload = packet['Raw'].load
#             # 将负载数据转换为十六进制形式
#             payload_hex = binascii.hexlify(payload).decode('utf-8')
#             # 打印或存储转换后的十六进制数据
#             # print(payload_hex)
#             payload_split = ' '.join(payload_hex[i:i + 4] for i in range(0, len(payload_hex), 4))
#             packet_info_string += ' ' + payload_split
        
#         # 截取一定长度的数据包数据
#         packet_txt = cutPacket(packet_info_string, payload_len)
#         feature_data.append(packet_txt)
#     return feature_data

def pretrain_dataset_generation(pcap_path):
    
    output_split_path = pcap_path
    # Split Pcap into Session-flow
    # if not os.path.exists(pcap_path + "splitcap"):
    #     print("Begin to split pcap as session flows.")
    #     for _p, _d, files in os.walk(pcap_path):
    #         for file in files:
    #             # add file filter
    #             if not file.endswith('.pcap') or file.startswith('._'):
    #                 print(f"Skip not pcap: {file}")
    #                 continue
    #             full_path = os.path.join(_p, file)
    #             # obtain relative path
    #             rel_path = os.path.relpath(full_path, pcap_path)
    #             # obtain the first as label
    #             first_dir = rel_path.split(os.sep)[0]
    #             print(first_dir)
    #             split_cap(output_split_path, full_path, file, pcap_label=first_dir, dataset_level='flow')
    # output_split_path=os.path.join(pcap_path, "splitcap")

    # Generate flow_level text dataset.
    print("Begin to generate flow_level text dataset.")
    firstline1 = 'flow_order'
    firstline2 = 'relative_time'
    firstline3 = 'packet_length_direction'
    firstline4 = 'text_a'
    flow_order = 1
    with open(word_dir + word_name, 'a') as f:
        f.write(f"{firstline1}\t{firstline2}\t{firstline3}\t{firstline4}\n")
    for _p, _d, files in os.walk(output_split_path):
        for file in files:
            full_path = os.path.join(_p, file)
            print(full_path)
            flow_order = get_pretraining_corpara(full_path, flow_order, payload_len=128)
    return 0


def size_format(size):
    # 'KB'
    file_size = '%.3f' % float(size / 1000)
    return file_size

# def generation(pcap_path, samples, features, splitcap=True, payload_length=128, payload_packet=128, dataset_save_path="/home/sunshine/Projects/ET-BERT-main/results/CipherSpectrum/chacha/H1/", dataset_level="flow", packet_encoding_mode="field"):
def generation(
    pcap_path,
    samples,
    features,
    splitcap=True,
    payload_length=128,
    payload_packet=128,
    dataset_save_path="/home/sunshine/Projects/ET-BERT-main/results/CipherSpectrum/chacha/H1/",
    dataset_level="flow",
    packet_encoding_mode="bigram",
    field_degradation_probability=0.0,
    field_degradation_seed=42,
    field_degradation_stats=None,
):
    dataset = {}
    label_name_list = []
    session_pcap_path = {}

    # 1. 获取类别标签名称
    for parent, dirs, files in os.walk(pcap_path):
        if not label_name_list:
            label_name_list.extend(sorted(dirs))
            print(label_name_list)
        for dir in label_name_list:
            for p, dd, ff in os.walk(os.path.join(parent, dir)):
                if splitcap:
                    print("Start Splitting")
                    for file in ff:
                        session_path = split_cap(pcap_path, os.path.join(p, file), file.split(".")[-2], dir, dataset_level=dataset_level)
                    session_pcap_path[dir] = os.path.join(pcap_path, "splitcap", dir)
                else:
                    session_pcap_path[dir] = os.path.join(pcap_path, dir)
                    print(os.path.join(pcap_path, dir))
        print("finish")
        break

    # 2. 为每个标签创建对应的id(序号)
    label_id = {label_name_list[index]: index for index in range(len(label_name_list))}
    r_file_record = []
    print("\nBegin to generate features.")

    # 3. splitcap和小流删除逻辑
    # if splitcap == False:
    #     for key in session_pcap_path.keys():
    #         if dataset_level == "flow":
    #             for p, d, f in os.walk(session_pcap_path[key]):
    #                 for file in f:
    #                     if file.endswith('.pcap'):
    #                         file_path = os.path.join(p, file)
    #                         file_size = float(size_format(os.path.getsize(file_path)))
    #                         if file_size < 5:
    #                             os.remove(file_path)
    #                             print(f"remove sample: {file_path} for its size is less than 5 KB.")
            # elif dataset_level == "packet":
            #     for p, d, f in os.walk(session_pcap_path[key]):
            #         for file in f:
            #             if file.endswith('.pcap'):
            #                 file_path = os.path.join(p, file)
            #                 if not os.path.getsize(file_path):
            #                     print(f"current pcap {file_path} is 0KB and delete")
            #                     os.remove(file_path)
            #                 else:
            #                     try:
            #                         current_packet = scapy.rdpcap(file_path)
            #                         file_size = float(size_format(os.path.getsize(file_path)))
            #                         if 'TCP' in str(current_packet.res):
            #                             if file_size < 0.14:
            #                                 os.remove(file_path)
            #                                 print(f"remove TCP sample: {file_path} for its size is less than 0.14KB.")
            #                         elif 'UDP' in str(current_packet.res):
            #                             if file_size < 0.1:
            #                                 os.remove(file_path)
            #                                 print(f"remove UDP sample: {file_path} for its size is less than 0.1KB.")
            #                     except Exception as e:
            #                         print("error in data_generation 611: scapy read pcap and analyse error")
            #                         os.remove(file_path)
            #                         print(f"remove packet sample: {file_path} for reading error.")

    # 4. 特征提取与数据组织
    print("Begin to generate flow dataset.")
    if packet_encoding_mode == "bigram":
        flow_feature_fn = get_finetuning_flow_feature_bigram
        print("Packet encoding mode: bigram (bi-byte split, Baseline get_feature_packet style)")
    elif packet_encoding_mode == "field":
        flow_feature_fn = get_finetuning_flow_feature_text
        print("Packet encoding mode: field (IP/TCP/UDP field parsing)")
    elif packet_encoding_mode == "field_degraded":
        if not 0.0 <= float(field_degradation_probability) <= 1.0:
            raise ValueError("field_degradation_probability must be in [0, 1]")
        degradation_rng = random.Random(field_degradation_seed)
        if field_degradation_stats is None:
            field_degradation_stats = {}
        field_degradation_stats.update({
            "configured_probability": float(field_degradation_probability),
            "seed": int(field_degradation_seed),
            "eligible_fields": 0,
            "degraded_fields": 0,
            "eligible_by_field": {},
            "degraded_by_field": {},
        })
        flow_feature_fn = partial(
            get_finetuning_flow_feature_text,
            field_degradation_probability=field_degradation_probability,
            field_degradation_rng=degradation_rng,
            field_degradation_stats=field_degradation_stats,
        )
        print(
            "Packet encoding mode: field_degraded "
            f"(probability={field_degradation_probability}, seed={field_degradation_seed}; "
            "length fields retained)"
        )
    elif packet_encoding_mode == "field_zero_length":
        flow_feature_fn = get_finetuning_flow_feature_text_zero_length
        print("Packet encoding mode: field_zero_length (legacy fields; direct length fields are zero)")
    elif packet_encoding_mode == "field_ipv4_ipv6":
        flow_feature_fn = get_finetuning_flow_feature_text_ipv4_ipv6
        print("Packet encoding mode: field_ipv4_ipv6 (legacy IPv4 plus opt-in IPv6 parsing)")
    elif packet_encoding_mode == "field_ipv4_ipv6_l2safe":
        flow_feature_fn = get_finetuning_flow_feature_text_ipv4_ipv6_l2safe
        print("Packet encoding mode: field_ipv4_ipv6_l2safe (IPv4/IPv6 plus link-layer-independent payload)")
    elif packet_encoding_mode == "field_ipv4_ipv6_l2safe_zero_length":
        flow_feature_fn = get_finetuning_flow_feature_text_ipv4_ipv6_l2safe_zero_length
        print(
            "Packet encoding mode: field_ipv4_ipv6_l2safe_zero_length "
            "(corrected IPv4/IPv6 fields; direct length fields are zero)"
        )
    elif packet_encoding_mode == "field_ipv4_ipv6_l2safe_app":
        flow_feature_fn = get_finetuning_flow_feature_text_application
        print("Packet encoding mode: field_ipv4_ipv6_l2safe_app (plus TLS record / QUIC header fields)")
    else:
        raise ValueError(
            f"Unsupported packet_encoding_mode: {packet_encoding_mode}. "
            "Choose 'field', 'field_degraded', 'field_zero_length', 'field_ipv4_ipv6', "
            "'field_ipv4_ipv6_l2safe', 'field_ipv4_ipv6_l2safe_zero_length', "
            "'field_ipv4_ipv6_l2safe_app', or 'bigram'."
        )
    firstline0 = "category"
    firstline1 = 'flow_order'
    firstline2 = 'relative_time'
    firstline3 = 'packet_length_direction'
    firstline4 = 'text_a'
    
    with open(fine_tuning_flowdataset_dir + fine_tuning_flowdataset_name, 'a') as f:
        f.write(f"{firstline0}\t{firstline1}\t{firstline2}\t{firstline3}\t{firstline4}\n")
    for key in tqdm(session_pcap_path.keys()):
        print(key)
        if label_id[key] not in dataset:
            if dataset_level == "flow":
                dataset[label_id[key]] = {
                    "samples": 0,
                    "flow_order": {},
                    "relative_time": {},
                    "packet_length_direction": {},
                    "text_a": {}
                }
            # elif dataset_level == "packet":
            #     dataset[label_id[key]] = {
            #         "samples": 0,
            #         "text_a": {}
            #     }
        # 采样逻辑
        target_all_files = [os.path.join(x[0], y) for x in [(p, f) for p, d, f in os.walk(session_pcap_path[key])] for y in x[1] if y.endswith('.pcap')]
        sample_size = min(samples[0] if isinstance(samples, list) else samples, len(target_all_files))
        if sample_size == 0:
            print(f"[Warning] 类别 {key} 没有可用样本，跳过。")
            continue
        r_files = random.sample(target_all_files, sample_size)
        flow_order = 0
        cate=label_id[key]
        for r_f in r_files:
            print(r_f)
            # 添加文件存在性检查
            if not os.path.exists(r_f):
                print(f"文件不存在，跳过: {r_f}")
                continue
                
            try:
                if dataset_level == "flow":
                    isflow = flow_feature_fn(
                        r_f,
                        cate,
                        flow_order,
                        payload_len=payload_length,
                        payload_packet=payload_packet,
                    )
                    if isflow != -1:
                        flow_order+=1
                # elif dataset_level == "packet":
                #     feature_data = get_finetuning_packet_text(r_f, payload_len=payload_length)
                else:
                    print(f"[Error] 不支持的数据集级别: {dataset_level}")
                    continue
                    
                if isflow == -1:
                    print(f"[Warning] 样本 {r_f} 特征提取失败，跳过。")
                    continue
            except Exception as e:
                print(f"[Error] 处理文件 {r_f} 时发生错误: {e}")
                continue
    #         r_file_record.append(r_f)
    #         dataset[label_id[key]]["samples"] += 1
    #         sample_idx = str(dataset[label_id[key]]["samples"])
    #         if dataset_level == "flow":
    #             # 保证feature_data长度和字段顺序一致
    #             dataset[label_id[key]]["flow_order"][sample_idx] = feature_data[0]
    #             dataset[label_id[key]]["relative_time"][sample_idx] = feature_data[1]
    #             dataset[label_id[key]]["packet_sequence"][sample_idx] = feature_data[2]
    #             dataset[label_id[key]]["text_a"][sample_idx] = feature_data[3]
    #         elif dataset_level == "packet":
    #             dataset[label_id[key]]["text_a"][sample_idx] = feature_data[0]

    # # 5. 统计信息
    # all_data_number = 0
    # for index in range(len(label_name_list)):
    #     label = label_name_list[index]
    #     print(f"{label_id[label]}\t{label}\t{dataset[label_id[label]]['samples']}")
    #     all_data_number += dataset[label_id[label]]["samples"]
    # print(f"all\t{all_data_number}")

    # # 6. 写入文件
    # os.makedirs(dataset_save_path, exist_ok=True)
    # with open(os.path.join(dataset_save_path, "picked_file_record"), "w") as p_f:
    #     for i in r_file_record:
    #         p_f.write(i)
    #         p_f.write("\n")
    # with open(os.path.join(dataset_save_path, "dataset.json"), "w") as f:
    #     json.dump(dataset, fp=f, ensure_ascii=False)

    # # 7. 生成X, Y
    # X, Y = obtain_data(pcap_path, samples, features, dataset_save_path, json_data=dataset)
    # return X, Y

# def read_data_from_json(json_data, features, samples):
#     X, Y = [], []
#     ablation_flag = 0
#     for feature_index in range(len(features)):
#         x = []
#         label_count = 0
#         for label in json_data.keys():
#             sample_num = json_data[label]["samples"]
#             if X == []:
#                 if not ablation_flag:
#                     y = [label] * sample_num
#                     Y.append(y)
#                 else:
#                     if sample_num > 1500:
#                         y = [label] * 1500
#                     else:
#                         y = [label] * sample_num
#                     Y.append(y)
#             if samples[label_count] < sample_num:
#                 x_label = []
#                 for sample_index in random.sample(list(json_data[label][features[feature_index]].keys()), 1500):
#                     x_label.append(json_data[label][features[feature_index]][sample_index])
#                 x.append(x_label)
#             else:
#                 x_label = []
#                 for sample_index in json_data[label][features[feature_index]].keys():
#                     x_label.append(json_data[label][features[feature_index]][sample_index])
#                 x.append(x_label)
#             label_count += 1
#         X.append(x)
#     return X, Y


# def obtain_data(pcap_path, samples, features, dataset_save_path, json_data=None):
#     if json_data:
#         X, Y = read_data_from_json(json_data, features, samples)
#     else:
#         print("read dataset from json file.")
#         with open(dataset_save_path + "dataset.json", "r") as f:
#             dataset = json.load(f)
#         X, Y = read_data_from_json(dataset, features, samples)

#     for index in range(len(X)):
#         if len(X[index]) != len(Y):
#             print("data and labels are not properly associated.")
#             print("x:%s\ty:%s" % (len(X[index]), len(Y)))
#             return -1
#     return X, Y

if __name__ == '__main__':
    pcap_path = "/mnt/nfs/sunshine/Datasets/cstnet-threshold/0.2/splitcap/"
    pretrain_dataset_generation(pcap_path)

