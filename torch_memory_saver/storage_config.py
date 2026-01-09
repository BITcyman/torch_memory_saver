"""Storage backend configuration for TorchMemorySaver"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class StorageBackend(Enum):
    """Storage backend types"""
    CPU = "cpu"
    MOONCAKE = "mooncake"
    NVME = "nvme"


@dataclass
class MooncakeConfig:
    """Configuration for Mooncake distributed storage backend"""

    # Network configuration
    local_hostname: str = "127.0.0.1:0"
    metadata_server: str = "P2PHANDSHAKE"
    protocol: str = "tcp"
    master_server_addr: str = "127.0.0.1:50051"
    rdma_devices: str = ""

    # Memory configuration
    global_segment_size: int = 16 * 1024 * 1024  # 16MB
    local_buffer_size: int = 16 * 1024 * 1024    # 16MB

    # Replication configuration
    replica_num: int = 2
    with_soft_pin: bool = True
    prefer_alloc_in_same_node: bool = False

    def __post_init__(self):
        """Validate configuration"""
        if self.replica_num < 1:
            raise ValueError("replica_num must be at least 1")
        if self.global_segment_size <= 0:
            raise ValueError("global_segment_size must be positive")
        if self.local_buffer_size <= 0:
            raise ValueError("local_buffer_size must be positive")
