from .entrypoint import TorchMemorySaver
from .hooks.mode_preload import configure_subprocess
from .storage_config import StorageBackend, MooncakeConfig

# Global singleton
torch_memory_saver = TorchMemorySaver()

__all__ = [
    "TorchMemorySaver",
    "torch_memory_saver",
    "configure_subprocess",
    "StorageBackend",
    "MooncakeConfig",
]
