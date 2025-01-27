from pathlib import Path

from scapy.layers.dns import DNS
from scapy.layers.inet import TCP
from scapy.packet import Padding
from scapy.utils import PcapReader

# for ev status identification
PREFIX_EV_STA_ID = {
    # charging benign
    'evse-a-charging-benign': 0,
    # charing attack
    'evse-a-charging-aggressive-scan': 1,
    'evse-b-charging-aggressive-scan': 1,
    'evse-a-charging-icmp-flood': 1,
    'evse-b-charging-icmp-flood': 1,
    'evse-a-charging-icmp-fragmentation': 1,
    'evse-a-charging-os-fingerprinting': 1,
    'evse-b-charging-os-fingerprinting': 1,
    'evse-a-charging-portscan': 1,
    'evse-b-charging-port-scan': 1,
    'evse-a-charging-push-ack-flood': 1,
    'evse-b-charging-push-ack-flood': 1,
    'evse-a-charging-service-detection': 1,
    'evse-b-charging-service-detection-scan': 1,
    'evse-a-charging-slowLoris-scan': 1,
    'evse-a-charging-syn-flood': 1,
    'evse-b-charging-syn-flood': 1,
    'evse-a-charging-syn-stealth': 1,
    'evse-b-charging-syn-stealth': 1,
    'evse-a-charging-synonymous-ip': 1,
    'evse-b-charging-synonymous-ip-flood': 1,
    'evse-a-charging-tcp-flood': 1,
    'evse-b-charging-tcp-flood': 1,
    'evse-a-charging-udp-flood': 1,
    'evse-b-charging-udp-flood': 1,
    'evse-a-charging-vulnerability-scan': 1,
    'evse-b-charging-vulnerability-scan': 1,
    # idle benign
    'evse-a-idle-benign': 2,
    # idle attack
    'evse-a-idle-aggressive-scan': 3,
    'evse-b-idle-aggressive-scan': 3,
    'evse-b-idle-icmp-flood': 3,
    'evse-a-idle-icmp-fragmentation': 3,
    'evse-b-idle-icmp-fragmentation': 3,
    'evse-a-idle-os-fingerprinting': 3,
    'evse-b-idle-os-fingerprinting': 3,
    'evse-a-idle-portscan': 3,
    'evse-b-idle-port-scan': 3,
    'evse-b-idle-push-ack-flood': 3,
    'evse-a-idle-service-detection': 3,
    'evse-b-idle-service-detection': 3,
    'evse-a-idle-slowloris-scan': 3,
    'evse-a-idle-syn-flood': 3,
    'evse-b-idle-syn-flood': 3,
    'evse-a-idle-syn-stealth-scan': 3,
    'evse-b-idle-syn-stealth-scan': 3,
    'evse-a-idle-synonymous-ip': 3,
    'evse-b-idle-synonymous-ip-flood': 3,
    'evse-a-idle-tcp-flood': 3,
    'evse-b-idle-tcp-flood': 3,
    'evse-a-idle-udp-flood': 3,
    'evse-b-idle-udp-flood': 3,
    'evse-a-idle-vulnerability-scan': 3,
    'evse-b-idle-vulnerability-scan': 3,
}

ID_TO_EV = {
    0: "charging benign",
    1: "charging attack",
    2: "idle benign",
    3: "idle attack",
}

# for attack identification
PREFIX_TO_ATK_ID = {
    # charging benign
    'evse-a-charging-benign': 0,
    # idle benign
    'evse-a-idle-benign': 0,
    # charing attack
    'evse-a-charging-aggressive-scan': 1,
    'evse-b-charging-aggressive-scan': 1,
    'evse-a-charging-icmp-flood': 1,
    'evse-b-charging-icmp-flood': 1,
    'evse-a-charging-icmp-fragmentation': 1,
    'evse-a-charging-os-fingerprinting': 1,
    'evse-b-charging-os-fingerprinting': 1,
    'evse-a-charging-portscan': 1,
    'evse-b-charging-port-scan': 1,
    'evse-a-charging-push-ack-flood': 1,
    'evse-b-charging-push-ack-flood': 1,
    'evse-a-charging-service-detection': 1,
    'evse-b-charging-service-detection-scan': 1,
    'evse-a-charging-slowLoris-scan': 1,
    'evse-a-charging-syn-flood': 1,
    'evse-b-charging-syn-flood': 1,
    'evse-a-charging-syn-stealth': 1,
    'evse-b-charging-syn-stealth': 1,
    'evse-a-charging-synonymous-ip': 1,
    'evse-b-charging-synonymous-ip-flood': 1,
    'evse-a-charging-tcp-flood': 1,
    'evse-b-charging-tcp-flood': 1,
    'evse-a-charging-udp-flood': 1,
    'evse-b-charging-udp-flood': 1,
    'evse-a-charging-vulnerability-scan': 1,
    'evse-b-charging-vulnerability-scan': 1,
    # maliciousev attack
    'evse-b-maliciousev-aggressive-scan': 1,
    'evse-b-maliciousev-os-fingerprinting': 1,
    'evse-b-maliciousev-port-scan': 1,
    'evse-b-maliciousev-service-detection': 1,
    'evse-b-maliciousev-syn-stealth-scan': 1,
    'evse-b-maliciousev-vulnerability-scan': 1,
    # idle attack
    'evse-a-idle-aggressive-scan': 1,
    'evse-b-idle-aggressive-scan': 1,
    'evse-b-idle-icmp-flood': 1,
    'evse-a-idle-icmp-fragmentation': 1,
    'evse-b-idle-icmp-fragmentation': 1,
    'evse-a-idle-os-fingerprinting': 1,
    'evse-b-idle-os-fingerprinting': 1,
    'evse-a-idle-portscan': 1,
    'evse-b-idle-port-scan': 1,
    'evse-b-idle-push-ack-flood': 1,
    'evse-a-idle-service-detection': 1,
    'evse-b-idle-service-detection': 1,
    'evse-a-idle-slowloris-scan': 1,
    'evse-a-idle-syn-flood': 1,
    'evse-b-idle-syn-flood': 1,
    'evse-a-idle-syn-stealth-scan': 1,
    'evse-b-idle-syn-stealth-scan': 1,
    'evse-a-idle-synonymous-ip': 1,
    'evse-b-idle-synonymous-ip-flood': 1,
    'evse-a-idle-tcp-flood': 1,
    'evse-b-idle-tcp-flood': 1,
    'evse-a-idle-udp-flood': 1,
    'evse-b-idle-udp-flood': 1,
    'evse-a-idle-vulnerability-scan': 1,
    'evse-b-idle-vulnerability-scan': 1,
}

ID_TO_ATK = {
    0: "benign",
    1: "attack",
}

# for attack scenario identification
PREFIX_TO_SCENARIO_ID = {
    # benign
    'evse-a-charging-benign': 0,
    'evse-a-idle-benign': 0,
    # aggressive scan
    'evse-a-charging-aggressive-scan': 1,
    'evse-b-charging-aggressive-scan': 1,
    'evse-a-idle-aggressive-scan': 1,
    'evse-b-idle-aggressive-scan': 1,
    'evse-b-maliciousev-aggressive-scan': 1,
    # os fingerprinting
    'evse-a-charging-os-fingerprinting': 1,
    'evse-b-charging-os-fingerprinting': 1,
    'evse-a-idle-os-fingerprinting': 1,
    'evse-b-idle-os-fingerprinting': 1,
    'evse-b-maliciousev-os-fingerprinting': 1,
    # port scan
    'evse-a-charging-portscan': 1,
    'evse-b-charging-port-scan': 1,
    'evse-a-idle-portscan': 1,
    'evse-b-idle-port-scan': 1,
    'evse-b-maliciousev-port-scan': 1,
    # service detection
    'evse-a-charging-service-detection': 1,
    'evse-b-charging-service-detection-scan': 1,
    'evse-a-idle-service-detection': 1,
    'evse-b-idle-service-detection': 1,
    'evse-b-maliciousev-service-detection': 1,
    # slowloris scan
    'evse-a-charging-slowloris-scan': 1,
    'evse-a-idle-slowloris-scan': 1,
    # vulnerability scan
    'evse-a-charging-vulnerability-scan': 1,
    'evse-b-charging-vulnerability-scan': 1,
    'evse-a-idle-vulnerability-scan': 1,
    'evse-b-idle-vulnerability-scan': 1,
    'evse-b-maliciousev-vulnerability-scan': 1,
    # syn stealth scan
    'evse-a-charging-syn-stealth': 1,
    'evse-b-charging-syn-stealth': 1,
    'evse-a-idle-syn-stealth-scan': 1,
    'evse-b-idle-syn-stealth-scan': 1,
    'evse-b-maliciousev-syn-stealth-scan': 1,
    # icmp flood
    'evse-a-charging-icmp-flood': 2,
    'evse-b-charging-icmp-flood': 2,
    'evse-b-idle-icmp-flood': 2,
    # icmp fragmentation
    'evse-a-charging-icmp-fragmentation': 2,
    'evse-a-idle-icmp-fragmentation': 2,
    'evse-b-idle-icmp-fragmentation': 2,
    # syn flood
    'evse-a-charging-syn-flood': 2,
    'evse-b-charging-syn-flood': 2,
    'evse-a-idle-syn-flood': 2,
    'evse-b-idle-syn-flood': 2,
    # push ack flood
    'evse-a-charging-push-ack-flood': 2,
    'evse-b-charging-push-ack-flood': 2,
    'evse-b-idle-push-ack-flood': 2,
    # synonymous ip
    'evse-a-charging-synonymous-ip': 2,
    'evse-b-charging-synonymous-ip-flood': 2,
    'evse-a-idle-synonymous-ip': 2,
    'evse-b-idle-synonymous-ip-flood': 2,
    # tcp flood
    'evse-a-charging-tcp-flood': 2,
    'evse-b-charging-tcp-flood': 2,
    'evse-a-idle-tcp-flood': 2,
    'evse-b-idle-tcp-flood': 2,
    # udp flood
    'evse-a-charging-udp-flood': 2,
    'evse-b-charging-udp-flood': 2,
    'evse-a-idle-udp-flood': 2,
    'evse-b-idle-udp-flood': 2,
}

ID_TO_SCENARIO = {
    0: "Benign",
    1: "Recon",
    2: "DoS",
}

# for attack identification
PREFIX_TO_ATTACK_ID = {
    # benign
    'evse-a-charging-benign': 0,
    'evse-a-idle-benign': 0,
    # aggressive scan
    'evse-a-charging-aggressive-scan': 1,
    'evse-b-charging-aggressive-scan': 1,
    'evse-a-idle-aggressive-scan': 1,
    'evse-b-idle-aggressive-scan': 1,
    'evse-b-maliciousev-aggressive-scan': 1,
    # icmp flood
    'evse-a-charging-icmp-flood': 2,
    'evse-b-charging-icmp-flood': 2,
    'evse-b-idle-icmp-flood': 2,
    # icmp fragmentation
    'evse-a-charging-icmp-fragmentation': 3,
    'evse-a-idle-icmp-fragmentation': 3,
    'evse-b-idle-icmp-fragmentation': 3,
    # os fingerprinting
    'evse-a-charging-os-fingerprinting': 4,
    'evse-b-charging-os-fingerprinting': 4,
    'evse-a-idle-os-fingerprinting': 4,
    'evse-b-idle-os-fingerprinting': 4,
    'evse-b-maliciousev-os-fingerprinting': 4,
    # port scan
    'evse-a-charging-portscan': 5,
    'evse-b-charging-port-scan': 5,
    'evse-a-idle-portscan': 5,
    'evse-b-idle-port-scan': 5,
    'evse-b-maliciousev-port-scan': 5,
    # push ack flood
    'evse-a-charging-push-ack-flood': 6,
    'evse-b-charging-push-ack-flood': 6,
    'evse-b-idle-push-ack-flood': 6,
    # service detection
    'evse-a-charging-service-detection': 7,
    'evse-b-charging-service-detection-scan': 7,
    'evse-a-idle-service-detection': 7,
    'evse-b-idle-service-detection': 7,
    'evse-b-maliciousev-service-detection': 7,
    # slowloris scan
    'evse-a-charging-slowloris-scan': 8,
    'evse-a-idle-slowloris-scan': 8,
    # syn flood
    'evse-a-charging-syn-flood': 9,
    'evse-b-charging-syn-flood': 9,
    'evse-a-idle-syn-flood': 9,
    'evse-b-idle-syn-flood': 9,
    # syn stealth scan
    'evse-a-charging-syn-stealth': 10,
    'evse-b-charging-syn-stealth': 10,
    'evse-a-idle-syn-stealth-scan': 10,
    'evse-b-idle-syn-stealth-scan': 10,
    'evse-b-maliciousev-syn-stealth-scan': 10,
    # synonymous ip
    'evse-a-charging-synonymous-ip': 11,
    'evse-b-charging-synonymous-ip-flood': 11,
    'evse-a-idle-synonymous-ip': 11,
    'evse-b-idle-synonymous-ip-flood': 11,
    # tcp flood
    'evse-a-charging-tcp-flood': 12,
    'evse-b-charging-tcp-flood': 12,
    'evse-a-idle-tcp-flood': 12,
    'evse-b-idle-tcp-flood': 12,
    # udp flood
    'evse-a-charging-udp-flood': 13,
    'evse-b-charging-udp-flood': 13,
    'evse-a-idle-udp-flood': 13,
    'evse-b-idle-udp-flood': 13,
    # vulnerability scan
    'evse-a-charging-vulnerability-scan': 14,
    'evse-b-charging-vulnerability-scan': 14,
    'evse-a-idle-vulnerability-scan': 14,
    'evse-b-idle-vulnerability-scan': 14,
    'evse-b-maliciousev-vulnerability-scan': 14,
}

ID_TO_ATTACK = {
    0: "benign",
    1: "agressive scan",
    2: "icmp flood",
    3: "icmp fragmentation",
    4: "os fingerprinting",
    5: "port scan",
    6: "push ack flood",
    7: "service detection",
    8: "slowloris scan",
    9: "syn flood",
    10: "syn stealth scan",
    11: "synonymous ip",
    12: "tcp flood",
    13: "udp flood",
    14: "vulnerability scan",
}


def read_pcap(path: Path):
    packets = PcapReader(str(path))

    return packets


def should_omit_packet(packet):
    # SYN, ACK or FIN flags set to 1 and no payload
    if TCP in packet and (packet.flags & 0x13):
        # not payload or contains only padding
        layers = packet[TCP].payload.layers()
        if not layers or (Padding in layers and len(layers) == 1):
            return True

    # DNS segment
    if DNS in packet:
        return True

    return False
