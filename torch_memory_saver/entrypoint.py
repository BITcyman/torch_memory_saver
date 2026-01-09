import ctypes

import numpy as np
import logging
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional
import torch

from .binary_wrapper import BinaryWrapper
from .hooks.base import HookUtilBase, HookMode
from .storage_config import StorageBackend, MooncakeConfig

logger = logging.getLogger(__name__)

_TAG_DEFAULT = "default"


class TorchMemorySaver:
    def __init__(self):
        self._impl_ctor_kwargs = {}
        self._impl: Optional[_TorchMemorySaverImpl] = None
        self._mooncake_config: Optional[MooncakeConfig] = None

    @contextmanager
    def region(
        self,
        tag: str = _TAG_DEFAULT,
        enable_cpu_backup: bool = False,
        storage_backend: Optional[StorageBackend] = None
    ):
        """Context manager for memory saving with optional tag and storage backend

        Args:
            tag: Tag for this memory region
            enable_cpu_backup: Enable backup (deprecated, use storage_backend parameter)
            storage_backend: Storage backend to use (CPU, MOONCAKE, or NVME)
                           If None, uses the globally configured backend
        """
        self._ensure_initialized()

        # Determine storage backend
        if storage_backend is None:
            # Backward compatibility: if enable_cpu_backup is True, use CPU backend
            storage_backend = StorageBackend.CPU if enable_cpu_backup else None

        # Configure storage backend if specified
        if storage_backend is not None:
            self._configure_storage_backend(storage_backend)

        with self._impl.region(tag=tag, enable_cpu_backup=enable_cpu_backup, storage_backend=storage_backend):
            yield

    @contextmanager
    def cuda_graph(
            self,
            cuda_graph, pool=None, stream=None, capture_error_mode='global',
            tag: str = _TAG_DEFAULT, enable_cpu_backup: bool = False,
    ):
        """Similar to `torch.cuda.graph`, but ensures memory in it to be pauseable."""
        self._ensure_initialized()
        with self._impl.cuda_graph(
                cuda_graph=cuda_graph,
                pool=pool, stream=stream, capture_error_mode=capture_error_mode,
                tag=tag, enable_cpu_backup=enable_cpu_backup,
        ):
            yield

    @contextmanager
    def disable(self):
        self._ensure_initialized()
        with self._impl.disable():
            yield

    def pause(self, tag: Optional[str] = None):
        """Pause memory for specific tag or all memory if tag is None"""
        self._ensure_initialized()
        self._impl.pause(tag=tag)

    def resume(self, tag: Optional[str] = None):
        """Resume memory for specific tag or all memory if tag is None"""
        self._ensure_initialized()
        self._impl.resume(tag=tag)

    # for compatibility
    @property
    def enabled(self):
        return True

    @property
    def hook_mode(self):
        raise AttributeError

    @hook_mode.setter
    def hook_mode(self, hook_mode: HookMode):
        assert self._impl_ctor_kwargs is not None, "Cannot configure after initialization"
        self._impl_ctor_kwargs["hook_mode"] = hook_mode

    @property
    def memory_margin_bytes(self):
        raise NotImplementedError("Only setter is supported")

    @memory_margin_bytes.setter
    def memory_margin_bytes(self, value: int):
        self._ensure_initialized()
        self._impl._binary_wrapper.cdll.set_memory_margin_bytes(value)

    def get_cpu_backup(self, x: torch.Tensor):
        self._ensure_initialized()
        return self._impl.get_cpu_backup(x)

    @property
    def mooncake_config(self) -> Optional[MooncakeConfig]:
        """Get Mooncake configuration"""
        return self._mooncake_config

    @mooncake_config.setter
    def mooncake_config(self, config: MooncakeConfig):
        """Set Mooncake configuration (must be set before first use of Mooncake backend)"""
        if self._impl is not None:
            logger.warning("Setting mooncake_config after initialization - applying to C++ layer")
            self._apply_mooncake_config(config)
        self._mooncake_config = config

    @property
    def storage_backend(self) -> StorageBackend:
        """Get current storage backend type"""
        self._ensure_initialized()
        backend_int = self._impl._binary_wrapper.cdll.tms_get_storage_backend_type()
        # Map C++ enum values to Python enum
        backend_map = {0: StorageBackend.CPU, 1: StorageBackend.MOONCAKE, 2: StorageBackend.NVME}
        return backend_map.get(backend_int, StorageBackend.CPU)

    @storage_backend.setter
    def storage_backend(self, backend: StorageBackend):
        """Set default storage backend type"""
        self._ensure_initialized()
        self._configure_storage_backend(backend)

    def _configure_storage_backend(self, backend: StorageBackend):
        """Configure storage backend in C++ layer"""
        self._ensure_initialized()

        # Set backend type
        self._impl._binary_wrapper.cdll.tms_set_storage_backend_type(backend.value.encode("utf-8"))

        # If Mooncake backend, ensure configuration is set
        if backend == StorageBackend.MOONCAKE:
            if self._mooncake_config is None:
                raise ValueError(
                    "Mooncake configuration must be set before using MOONCAKE storage backend. "
                    "Use torch_memory_saver.mooncake_config = MooncakeConfig(...)"
                )
            self._apply_mooncake_config(self._mooncake_config)

    def _apply_mooncake_config(self, config: MooncakeConfig):
        """Apply Mooncake configuration to C++ layer"""
        self._ensure_initialized()
        self._impl._binary_wrapper.cdll.tms_set_mooncake_config(
            config.local_hostname.encode("utf-8"),
            config.metadata_server.encode("utf-8"),
            config.protocol.encode("utf-8"),
            config.master_server_addr.encode("utf-8"),
            config.rdma_devices.encode("utf-8"),
            config.global_segment_size,
            config.local_buffer_size,
            config.replica_num,
            config.with_soft_pin,
            config.prefer_alloc_in_same_node
        )

    def _ensure_initialized(self):
        if self._impl is not None:
            return
        self._impl = _TorchMemorySaverImpl(**self._impl_ctor_kwargs)
        del self._impl_ctor_kwargs


class _TorchMemorySaverImpl:
    def __init__(self, hook_mode: HookMode = "preload"):
        self._hook_mode = hook_mode
        self._hook_util = HookUtilBase.create(hook_mode=hook_mode)
        self._binary_wrapper = BinaryWrapper(path_binary=self._hook_util.get_path_binary())
        self._mem_pools = defaultdict(lambda: torch.cuda.MemPool(allocator=self._hook_util.get_allocator()))
        _sanity_checks()

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool, storage_backend: Optional[StorageBackend]):
        # For hook_mode=preload, we need this b/c https://github.com/fzyzcjy/torch_memory_saver/pull/20#issuecomment-3047099047
        # (For hook_mode=torch we may not need it, but currently our primary usage is hook_mode=preload, thus we do this for simplicity)
        mem_pool = self._mem_pools[(tag, enable_cpu_backup, storage_backend)]
        with torch.cuda.use_mem_pool(mem_pool):
            with self._with_region_config(tag=tag, enable_cpu_backup=enable_cpu_backup):
                yield

    @contextmanager
    def cuda_graph(self, cuda_graph, pool, stream, capture_error_mode, tag: str, enable_cpu_backup: bool):
        assert self._hook_mode == "preload", "Only hook_mode=preload supports pauseable CUDA Graph currently"
        with torch.cuda.graph(cuda_graph, pool=pool, stream=stream, capture_error_mode=capture_error_mode):
            with self._with_region_config(tag=tag, enable_cpu_backup=enable_cpu_backup):
                yield

    @contextmanager
    def _with_region_config(self, tag: str, enable_cpu_backup: bool):
        assert not self._binary_wrapper.cdll.tms_get_interesting_region()
        original_enable_cpu_backup = self._binary_wrapper.cdll.tms_get_enable_cpu_backup()

        self._binary_wrapper.set_config(tag=tag, interesting_region=True, enable_cpu_backup=enable_cpu_backup)
        try:
            yield
        finally:
            assert self._binary_wrapper.cdll.tms_get_interesting_region()
            self._binary_wrapper.set_config(tag=_TAG_DEFAULT, interesting_region=False, enable_cpu_backup=original_enable_cpu_backup)

    @contextmanager
    def disable(self, dispose_mem_pool_after_use: bool = True):
        assert dispose_mem_pool_after_use, "Only dispose_mem_pool_after_use=true is supported now"
        assert self._binary_wrapper.cdll.tms_get_interesting_region(), "disable() should be called only when tms is active"

        self._binary_wrapper.cdll.tms_set_interesting_region(False)
        try:
            # We can either reuse the pool or delete it immediately, and we implement the latter currently since Slime uses it.
            # About why we need a pool: https://github.com/fzyzcjy/torch_memory_saver/pull/20#issuecomment-3047099047
            pool = torch.cuda.MemPool()
            with torch.cuda.use_mem_pool(pool):
                yield
            del pool
        finally:
            self._binary_wrapper.cdll.tms_set_interesting_region(True)

    def pause(self, tag: Optional[str]):
        tag_bytes = tag.encode("utf-8") if tag else None
        self._binary_wrapper.cdll.tms_pause(tag_bytes)

    def resume(self, tag: Optional[str]):
        tag_bytes = tag.encode("utf-8") if tag else None
        self._binary_wrapper.cdll.tms_resume(tag_bytes)

    def get_cpu_backup(self, x: torch.Tensor):
        assert x.is_cuda, f"{x.device=}"
        assert x.is_contiguous(), f"{x.shape=} {x.stride()=} {x.dtype=}"

        nbytes = x.nbytes
        gpu_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_uint8))
        cpu_ptr = self._binary_wrapper.cdll.tms_get_cpu_backup_pointer(gpu_ptr, nbytes)
        if not cpu_ptr:
            return None

        np_untyped = np.ctypeslib.as_array(cpu_ptr, shape=(nbytes,))
        assert np_untyped.dtype == np.uint8, f"{np_untyped.dtype=} {np_untyped.shape=}"

        ans_untyped = torch.from_numpy(np_untyped)
        ans = ans_untyped.view(x.dtype).view(x.shape)

        # For simplicity and safety
        ans = ans.clone()

        assert ans.device == torch.device("cpu"), f"{ans.device=}"
        assert ans.dtype == x.dtype, f"{ans.dtype=} {x.dtype=}"
        assert ans.shape == x.shape, f"{ans.shape=} {x.shape=}"
        assert ans.stride() == x.stride(), f"{ans.stride()=} {x.stride()=}"
        return ans

def _sanity_checks():
    if "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        raise RuntimeError(
            "TorchMemorySaver is disabled for the current process because expandable_segments is not supported yet."
        )
