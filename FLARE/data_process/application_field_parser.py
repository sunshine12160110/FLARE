#!/usr/bin/env python3
"""Application-aware packet field encoding with auditable token provenance.

This module is deliberately additive.  It does not change the legacy encoders
in ``rm_flow_dataset_generation.py``.  The new encoder preserves their
IPv4/IPv6/TCP/UDP schema, then parses TLS record headers or QUIC headers when
the flow can be identified.  Bytes which cannot be represented by a parsed
field are retained with the historical two-byte fallback.

Every emitted token carries two audit labels:

* ``layer``: ipv4, ipv6, tcp, udp, tls, quic, or payload;
* ``source``: field or fallback.

The labels are audit metadata only and are not written to model input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6


TLS_CONTENT_TYPES = {20, 21, 22, 23, 24}
TLS_VERSIONS = {0x0300, 0x0301, 0x0302, 0x0303, 0x0304}


@dataclass(frozen=True)
class TokenPiece:
    value: str
    layer: str
    source: str  # "field" or "fallback"
    field_name: str = ""


@dataclass
class ParseResult:
    pieces: List[TokenPiece]
    success: bool
    reason: str
    header_bytes: int = 0


@dataclass
class FlowProfile:
    application: Optional[str] = None
    evidence: str = "none"
    quic_variant: Optional[str] = None
    # QUIC short headers contain a destination CID whose length is learned
    # from the peer's long-header source CID in the same flow.
    quic_dcid_lengths: Dict[Tuple[str, int], int] = field(default_factory=dict)


def _piece(value: Any, layer: str, name: str, source: str = "field") -> TokenPiece:
    return TokenPiece(str(value), layer, source, name)


def _fallback(data: bytes, layer: str) -> List[TokenPiece]:
    hex_data = data.hex()
    return [
        _piece(hex_data[index:index + 4], layer, "raw", "fallback")
        for index in range(0, len(hex_data), 4)
        if hex_data[index:index + 4]
    ]


def truncate_pieces(pieces: Sequence[TokenPiece], hex_char_budget: int) -> List[TokenPiece]:
    """Reproduce ``cutPacket`` while retaining token provenance.

    ``cutPacket`` limits non-space characters and can cut the last token in
    half.  Keeping that partial token is necessary for exact model-visible
    token accounting.
    """
    remaining = max(0, int(hex_char_budget))
    visible: List[TokenPiece] = []
    for item in pieces:
        if remaining <= 0:
            break
        value = item.value[:remaining]
        if value:
            visible.append(TokenPiece(value, item.layer, item.source, item.field_name))
            remaining -= len(value)
    return visible


def pieces_to_text(pieces: Sequence[TokenPiece]) -> str:
    return " ".join(item.value for item in pieces)


def transport_payload(packet: Any) -> bytes:
    """Return application payload using Scapy's decoded L4 boundary."""
    try:
        if TCP in packet:
            payload = bytes(packet[TCP].payload)
            if IP in packet:
                expected = int(packet[IP].len) - int(packet[IP].ihl) * 4 - int(packet[TCP].dataofs) * 4
                return payload[:max(0, expected)]
            return payload
        if UDP in packet:
            payload = bytes(packet[UDP].payload)
            expected = int(packet[UDP].len) - 8
            return payload[:max(0, expected)]
    except Exception:
        return b""
    return b""


def endpoint(packet: Any, source: bool) -> Optional[Tuple[str, int]]:
    try:
        address = packet[IP].src if source and IP in packet else packet[IP].dst if IP in packet else packet[IPv6].src if source and IPv6 in packet else packet[IPv6].dst
        port = packet[UDP].sport if source else packet[UDP].dport
        return str(address), int(port)
    except Exception:
        return None


def looks_like_tls_record(payload: bytes) -> bool:
    if len(payload) < 5:
        return False
    content_type = payload[0]
    version = int.from_bytes(payload[1:3], "big")
    length = int.from_bytes(payload[3:5], "big")
    # RFC 8446 permits 2^14 bytes plus expansion.  A generous 18432-byte
    # ceiling rejects random payload without excluding legal records.
    return content_type in TLS_CONTENT_TYPES and version in TLS_VERSIONS and 0 <= length <= 18432


def parse_tls_payload(payload: bytes, expected: bool = True) -> ParseResult:
    if not payload:
        return ParseResult([], False, "tls_no_payload")
    if not looks_like_tls_record(payload):
        return ParseResult(_fallback(payload, "tls"), False, "tls_header_not_at_payload_start")

    pieces: List[TokenPiece] = []
    offset = 0
    parsed_headers = 0
    while offset + 5 <= len(payload) and looks_like_tls_record(payload[offset:]):
        content_type = payload[offset]
        version = int.from_bytes(payload[offset + 1:offset + 3], "big")
        record_length = int.from_bytes(payload[offset + 3:offset + 5], "big")
        pieces.extend([
            _piece(f"{content_type:02x}", "tls", "record.content_type"),
            _piece(f"{version:04x}", "tls", "record.version"),
            _piece(f"{record_length:04x}", "tls", "record.length"),
        ])
        parsed_headers += 5
        body_start = offset + 5
        available = min(record_length, len(payload) - body_start)

        # A complete plaintext handshake header is structurally meaningful.
        # Its body may contain variable or encrypted data and stays fallback.
        if content_type == 22 and available >= 4:
            handshake_type = payload[body_start]
            handshake_length = int.from_bytes(payload[body_start + 1:body_start + 4], "big")
            pieces.extend([
                _piece(f"{handshake_type:02x}", "tls", "handshake.type"),
                _piece(f"{handshake_length:06x}", "tls", "handshake.length"),
            ])
            pieces.extend(_fallback(payload[body_start + 4:body_start + available], "tls"))
        else:
            pieces.extend(_fallback(payload[body_start:body_start + available], "tls"))

        offset = body_start + available
        if available < record_length:
            # TCP segmentation: the header is still parsed successfully; the
            # record body merely continues in a later segment.
            break

    if offset < len(payload):
        pieces.extend(_fallback(payload[offset:], "tls"))
    return ParseResult(pieces, True, "success", parsed_headers)


def _read_varint(data: bytes, offset: int) -> Optional[Tuple[int, int, bytes]]:
    if offset >= len(data):
        return None
    width = 1 << (data[offset] >> 6)
    if offset + width > len(data):
        return None
    raw = data[offset:offset + width]
    value = int.from_bytes(raw, "big") & ((1 << (width * 8 - 2)) - 1)
    return value, offset + width, raw


def _supported_ietf_quic_version(version: int) -> bool:
    """Versions whose long-header layout is safe to interpret here.

    Version zero is Version Negotiation, 1 and 2 are standardized, and the
    ff0000xx range contains the commonly captured IETF draft versions.  An
    arbitrary four-byte value is not sufficient QUIC evidence because it
    creates false positives for protocols such as ZeroTier.
    """
    return version in {0, 0x00000001, 0x6B3343CF} or (version & 0xFFFFFF00) == 0xFF000000


def parse_gquic_header(payload: bytes) -> Optional[ParseResult]:
    """Parse the stable public flag/version prefix of legacy Google QUIC."""
    if len(payload) < 5:
        return None
    version = payload[1:5]
    if version[:1] != b"Q" or not version[1:].isdigit():
        return None
    pieces = [
        _piece(f"{payload[0]:02x}", "quic", "gquic.public_flags"),
        _piece(version.hex(), "quic", "gquic.version"),
    ]
    pieces.extend(_fallback(payload[5:], "quic"))
    return ParseResult(pieces, True, "success", 5)


def parse_quic_long_header(payload: bytes) -> Optional[Tuple[ParseResult, int]]:
    """Parse QUIC long-header fields visible without removing header protection.

    Returns the parsed result and source-CID length.  Packet-number bytes are
    header-protected, so they are deliberately retained as fallback rather
    than reported as semantic fields.
    """
    if len(payload) < 7 or not (payload[0] & 0x80) or not (payload[0] & 0x40):
        return None
    version = int.from_bytes(payload[1:5], "big")
    if not _supported_ietf_quic_version(version):
        return None
    offset = 5
    dcid_len = payload[offset]
    offset += 1
    if dcid_len > 20 or offset + dcid_len + 1 > len(payload):
        return None
    dcid = payload[offset:offset + dcid_len]
    offset += dcid_len
    scid_len = payload[offset]
    offset += 1
    if scid_len > 20 or offset + scid_len > len(payload):
        return None
    scid = payload[offset:offset + scid_len]
    offset += scid_len
    packet_type = (payload[0] >> 4) & 0x03
    pieces = [
        _piece(f"{payload[0]:02x}", "quic", "long.flags"),
        _piece(f"{version:08x}", "quic", "version"),
        _piece(f"{dcid_len:02x}", "quic", "dcid.length"),
    ]
    if dcid:
        pieces.append(_piece(dcid.hex(), "quic", "dcid"))
    pieces.append(_piece(f"{scid_len:02x}", "quic", "scid.length"))
    if scid:
        pieces.append(_piece(scid.hex(), "quic", "scid"))

    # Version Negotiation has an unprotected sequence of offered versions.
    if version == 0:
        while offset + 4 <= len(payload):
            pieces.append(_piece(payload[offset:offset + 4].hex(), "quic", "supported.version"))
            offset += 4
        pieces.extend(_fallback(payload[offset:], "quic"))
        return ParseResult(pieces, True, "success", offset), scid_len

    # QUIC v1/v2 Initial packets have a token-length and token before Length.
    if packet_type == 0:
        token_info = _read_varint(payload, offset)
        if token_info is None:
            return ParseResult(_fallback(payload, "quic"), False, "quic_truncated_token_length"), scid_len
        token_len, offset, token_raw = token_info
        pieces.append(_piece(token_raw.hex(), "quic", "token.length"))
        if offset + token_len > len(payload):
            return ParseResult(_fallback(payload, "quic"), False, "quic_truncated_token"), scid_len
        if token_len:
            pieces.append(_piece(payload[offset:offset + token_len].hex(), "quic", "token"))
        offset += token_len
    elif packet_type == 3:  # Retry: remainder is token plus 16-byte integrity tag.
        if len(payload) - offset < 16:
            return ParseResult(_fallback(payload, "quic"), False, "quic_truncated_retry"), scid_len
        token = payload[offset:-16]
        if token:
            pieces.append(_piece(token.hex(), "quic", "retry.token"))
        pieces.append(_piece(payload[-16:].hex(), "quic", "retry.integrity_tag"))
        return ParseResult(pieces, True, "success", len(payload)), scid_len

    length_info = _read_varint(payload, offset)
    if length_info is None:
        return ParseResult(_fallback(payload, "quic"), False, "quic_truncated_length"), scid_len
    _, offset, length_raw = length_info
    pieces.append(_piece(length_raw.hex(), "quic", "packet.length"))
    pieces.extend(_fallback(payload[offset:], "quic"))
    return ParseResult(pieces, True, "success", offset), scid_len


def looks_like_quic_long_header(payload: bytes) -> bool:
    parsed = parse_quic_long_header(payload)
    return (parsed is not None and parsed[0].success) or parse_gquic_header(payload) is not None


def build_flow_profile(packets: Iterable[Any]) -> FlowProfile:
    profile = FlowProfile()
    tls_seen = False
    quic_seen = False
    tls_port_hint = False
    quic_port_hint = False
    for packet in packets:
        payload = transport_payload(packet)
        if TCP in packet and payload:
            if looks_like_tls_record(payload):
                tls_seen = True
            try:
                tls_port_hint = tls_port_hint or int(packet[TCP].sport) in {443, 853, 8443} or int(packet[TCP].dport) in {443, 853, 8443}
            except Exception:
                pass
        if UDP in packet and payload:
            parsed = parse_quic_long_header(payload)
            if parsed is not None and parsed[0].success:
                quic_seen = True
                profile.quic_variant = profile.quic_variant or "ietf"
                src = endpoint(packet, True)
                if src is not None:
                    profile.quic_dcid_lengths[src] = parsed[1]
            elif parse_gquic_header(payload) is not None:
                quic_seen = True
                profile.quic_variant = "gquic"
            try:
                quic_port_hint = quic_port_hint or (
                    (int(packet[UDP].sport) in {443, 784, 8853} or int(packet[UDP].dport) in {443, 784, 8853})
                    and bool(payload[0] & 0x40)
                )
            except Exception:
                pass
    # A split flow cannot be both TCP/TLS and UDP/QUIC.  TLS wins only if both
    # occur in an unusual multi-flow capture because its signature is stricter.
    if tls_seen:
        profile.application, profile.evidence = "tls", "record_signature"
    elif quic_seen:
        profile.application, profile.evidence = "quic", "long_header_signature"
    elif tls_port_hint:
        profile.application, profile.evidence = "tls", "well_known_port"
    elif quic_port_hint:
        profile.application, profile.evidence = "quic", "well_known_port_and_fixed_bit"
    return profile


def parse_quic_payload(
    payload: bytes,
    destination_cid_length: Optional[int],
    variant: Optional[str] = None,
) -> ParseResult:
    gquic = parse_gquic_header(payload)
    if gquic is not None:
        return gquic
    if variant == "gquic":
        # The public-header layout changed across legacy gQUIC versions.  If
        # this packet does not carry the explicit Qxxx version marker, retain
        # it losslessly instead of guessing field boundaries.
        return ParseResult(_fallback(payload, "quic"), False, "gquic_header_without_version_not_parsed")
    parsed = parse_quic_long_header(payload)
    if parsed is not None:
        return parsed[0]
    if not payload or payload[0] & 0x80 or not (payload[0] & 0x40):
        return ParseResult(_fallback(payload, "quic"), False, "quic_invalid_fixed_or_header_form")
    if destination_cid_length is None:
        return ParseResult(_fallback(payload, "quic"), False, "quic_short_dcid_length_unknown")
    offset = 1 + destination_cid_length
    if offset > len(payload):
        return ParseResult(_fallback(payload, "quic"), False, "quic_truncated_short_header")
    pieces = [_piece(f"{payload[0]:02x}", "quic", "short.flags")]
    if destination_cid_length:
        pieces.append(_piece(payload[1:offset].hex(), "quic", "dcid"))
    # Packet-number length and bytes are protected.  The remaining bytes are
    # therefore retained losslessly as fallback.
    pieces.extend(_fallback(payload[offset:], "quic"))
    return ParseResult(pieces, True, "success", offset)


def base_field_pieces(packet: Any) -> Tuple[List[TokenPiece], bool, str]:
    """Encode the legacy network/transport fields with provenance."""
    pieces: List[TokenPiece] = []
    try:
        if IP in packet:
            ip = packet[IP]
            for value, name in [
                (f"{ip.version:01x}", "version"), (f"{ip.ihl:01x}", "ihl"),
                (f"{ip.tos:02X}", "tos"), (f"{ip.len:04x}", "len"),
                (f"{ip.id:04x}", "id"), (f"{int(ip.flags):01x}", "flags"),
                (f"{ip.ttl:02x}", "ttl"), (f"{ip.proto:02x}", "proto"),
                (f"{ip.chksum:04x}", "chksum"),
            ]:
                pieces.append(_piece(value, "ipv4", name))
            if ip.options and int(ip.ihl) * 4 > 20:
                pieces.extend(_fallback(bytes(ip)[20:int(ip.ihl) * 4], "ipv4"))
        elif IPv6 in packet:
            ip = packet[IPv6]
            for value, name in [
                (f"{ip.version:01x}", "version"), (f"{ip.tc:02x}", "tc"),
                (f"{ip.fl:05x}", "flow_label"), (f"{ip.plen:04x}", "payload_len"),
                (f"{ip.nh:02x}", "next_header"), (f"{ip.hlim:02x}", "hop_limit"),
            ]:
                pieces.append(_piece(value, "ipv6", name))
        else:
            return pieces, False, "network_layer_not_parsed"

        if TCP in packet:
            tcp = packet[TCP]
            fields = [
                (f"{tcp.seq:08x}", "seq"), (f"{tcp.ack:08x}", "ack"),
                (f"{tcp.dataofs:01x}", "dataofs"), (f"{tcp.reserved:02x}", "reserved"),
                (f"{int(tcp.flags):03x}", "flags"), (f"{tcp.window:04x}", "window"),
            ]
            if tcp.chksum is not None:
                fields.append((f"{tcp.chksum:04x}", "chksum"))
            fields.append((f"{tcp.urgptr:04x}", "urgptr"))
            pieces.extend(_piece(value, "tcp", name) for value, name in fields)
            if tcp.options and int(tcp.dataofs) * 4 > 20:
                pieces.extend(_fallback(bytes(tcp)[20:int(tcp.dataofs) * 4], "tcp"))
        elif UDP in packet:
            udp = packet[UDP]
            pieces.extend([
                _piece(f"{udp.len:04x}", "udp", "len"),
                _piece(f"{udp.chksum:04x}", "udp", "chksum"),
            ])
        else:
            return pieces, False, "transport_layer_not_parsed"
    except Exception as exc:
        return pieces, False, f"base_field_error:{type(exc).__name__}"
    return pieces, True, "success"


def encode_packet(
    packet: Any,
    profile: FlowProfile,
    hex_char_budget: int = 128,
) -> Tuple[List[TokenPiece], Dict[str, Any]]:
    pieces, base_success, base_reason = base_field_pieces(packet)
    payload = transport_payload(packet)
    app_result: Optional[ParseResult] = None
    expected_layer = "tcp" if TCP in packet else "udp" if UDP in packet else "network"

    if payload and profile.application == "tls" and TCP in packet:
        expected_layer = "tls"
        app_result = parse_tls_payload(payload)
    elif payload and profile.application == "quic" and UDP in packet:
        expected_layer = "quic"
        dst = endpoint(packet, False)
        app_result = parse_quic_payload(
            payload,
            profile.quic_dcid_lengths.get(dst) if dst else None,
            profile.quic_variant,
        )
    elif payload:
        pieces.extend(_fallback(payload, "payload"))

    if app_result is not None:
        pieces.extend(app_result.pieces)
    visible = truncate_pieces(pieces, hex_char_budget)
    app_success = app_result.success if app_result is not None else True
    success = base_success and app_success
    reason = base_reason if not base_success else app_result.reason if app_result and not app_result.success else "success"
    return visible, {
        "success": success,
        "reason": reason,
        "expected_layer": expected_layer,
        "application": profile.application or "none",
        "application_evidence": profile.evidence,
        "payload_bytes": len(payload),
        "application_header_bytes": app_result.header_bytes if app_result else 0,
    }

