#include "core.h"
#include "utils.h"
#include "macro.h"
#include "api_forwarder.h"
#include <sstream>
#include <iomanip>

#if defined(USE_ROCM)
#include "hardware_amd_support.h"
#endif

TorchMemorySaver::TorchMemorySaver() {}

TorchMemorySaver &TorchMemorySaver::instance() {
    static TorchMemorySaver instance;
    return instance;
}

// Storage backend management
void TorchMemorySaver::set_storage_backend_type(torch_memory_saver::StorageBackendType type) {
    std::lock_guard<std::mutex> lock(backend_mutex_);
    current_backend_type_ = type;
    std::cout << "[TorchMemorySaver] Storage backend type set to: " << static_cast<int>(type) << std::endl;
}

void TorchMemorySaver::set_mooncake_config(const torch_memory_saver::MooncakeConfig& config) {
    std::lock_guard<std::mutex> lock(backend_mutex_);
    mooncake_config_ = config;
    std::cout << "[TorchMemorySaver] Mooncake config updated" << std::endl;
}

torch_memory_saver::StorageBackendType TorchMemorySaver::get_current_backend_type() const {
    return current_backend_type_;
}

torch_memory_saver::StorageBackendInterface* TorchMemorySaver::get_storage_backend(
    torch_memory_saver::StorageBackendType type
) {
    std::lock_guard<std::mutex> lock(backend_mutex_);

    auto it = storage_backends_.find(type);
    if (it != storage_backends_.end()) {
        return it->second.get();
    }

    // Lazily create backend on first use
    std::unique_ptr<torch_memory_saver::StorageBackendInterface> backend;

    switch (type) {
        case torch_memory_saver::StorageBackendType::CPU_MEMORY:
            backend = std::make_unique<torch_memory_saver::CPUStorageBackend>();
            std::cout << "[TorchMemorySaver] Created CPU storage backend" << std::endl;
            break;

        case torch_memory_saver::StorageBackendType::MOONCAKE_STORE:
            backend = std::make_unique<torch_memory_saver::MooncakeStorageBackend>(mooncake_config_);
            std::cout << "[TorchMemorySaver] Created Mooncake storage backend" << std::endl;
            break;

        case torch_memory_saver::StorageBackendType::NVME_DISK:
            std::cerr << "[TorchMemorySaver] NVME_DISK backend not implemented yet" << std::endl;
            return nullptr;

        default:
            std::cerr << "[TorchMemorySaver] Unknown storage backend type: "
                      << static_cast<int>(type) << std::endl;
            return nullptr;
    }

    auto* backend_ptr = backend.get();
    storage_backends_[type] = std::move(backend);
    return backend_ptr;
}

std::string TorchMemorySaver::generate_object_key(void* ptr, const AllocationMetadata& metadata) {
    std::ostringstream oss;
    oss << "tms_" << metadata.tag << "_0x" << std::hex << std::setfill('0')
        << std::setw(16) << reinterpret_cast<uintptr_t>(ptr);
    return oss.str();
}

cudaError_t TorchMemorySaver::malloc(void **ptr, CUdevice device, size_t size, const std::string& tag, const bool enable_cpu_backup) {
    // Use current backend type, default to CPU if enable_cpu_backup is true
    torch_memory_saver::StorageBackendType backend_type = current_backend_type_;
    if (enable_cpu_backup && backend_type != torch_memory_saver::StorageBackendType::MOONCAKE_STORE) {
        backend_type = torch_memory_saver::StorageBackendType::CPU_MEMORY;
    }
    return malloc_with_backend(ptr, device, size, tag, backend_type);
}

cudaError_t TorchMemorySaver::malloc_with_backend(void **ptr, CUdevice device, size_t size, const std::string& tag,
                                                   torch_memory_saver::StorageBackendType backend_type) {
#if defined(USE_ROCM)
    // ROCm path - use legacy implementation for now
    bool enable_cpu_backup = (backend_type == torch_memory_saver::StorageBackendType::CPU_MEMORY);
    return ROCmHIPImplementation::rocm_malloc(ptr, device, size, tag, enable_cpu_backup, allocation_metadata_, allocator_metadata_mutex_);

#elif defined(USE_CUDA)
    const uint64_t memory_margin_bytes = memory_margin_bytes_.load();
    if (memory_margin_bytes > 0) {
        size_t free_bytes, total_bytes;
        CUDA_ERROR_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (memory_margin_bytes + size > free_bytes) {
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver::malloc return OOM since"
                << " memory_margin_bytes=" << memory_margin_bytes
                << " (alloc)size=" << size
                << " free_bytes=" << free_bytes
                << std::endl;
            return cudaErrorMemoryAllocation;
        }
    }

    CUmemGenericAllocationHandle allocHandle;

    cudaError_t ret = CUDAUtils::cu_mem_create(&allocHandle, size, device);
    if (ret != cudaSuccess) {
        return ret;
    }

    CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0));
    CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, size, 0, allocHandle, 0));
    CUDAUtils::cu_mem_set_access(*ptr, size, device);

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        bool enable_cpu_backup = (backend_type != torch_memory_saver::StorageBackendType::CPU_MEMORY) ||
                                (backend_type == torch_memory_saver::StorageBackendType::CPU_MEMORY);
        allocation_metadata_.emplace(
            *ptr,
            AllocationMetadata{size, device, tag, AllocationState::ACTIVE, enable_cpu_backup,
                              backend_type, nullptr, nullptr, allocHandle}
        );
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.malloc "
              << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
              << " allocHandle=" << allocHandle << " tag=" << tag
              << std::endl;
#endif

#else
    #error "USE_PLATFORM is not set"
#endif
    return cudaSuccess;
}

cudaError_t TorchMemorySaver::free(void *ptr) {
#if defined(USE_ROCM)
    return ROCmHIPImplementation::rocm_free(ptr, allocation_metadata_, allocator_metadata_mutex_);

#elif defined(USE_CUDA)
    AllocationMetadata metadata;
    {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
        if (allocation_metadata_.count(ptr) == 0) {
            return APIForwarder::call_real_cuda_free(ptr);
        }

        metadata = allocation_metadata_[ptr];
        allocation_metadata_.erase(ptr);
    }

    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
    CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
    CURESULT_CHECK(cuMemAddressFree((CUdeviceptr) ptr, metadata.size));

    if (nullptr != metadata.cpu_backup) {
        CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));
        metadata.cpu_backup = nullptr;
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.free "
              << " ptr=" << ptr << " metadata.size=" << metadata.size
              << " metadata.allocHandle=" << metadata.allocHandle << " tag=" << metadata.tag
              << std::endl;
#endif

#else
    #error "USE_PLATFORM is not set"
#endif
    return cudaSuccess;
}

void TorchMemorySaver::pause(const std::string& tag) {
#if defined(USE_ROCM)
    ROCmHIPImplementation::rocm_pause(tag, allocation_metadata_, allocator_metadata_mutex_);

#elif defined(USE_CUDA)
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata& metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        if (metadata.state != AllocationState::ACTIVE) {
            std::cerr << "[torch_memory_saver.cpp] Cannot pause allocation that is not active."
                      << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)ptr)
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }

        if (metadata.enable_cpu_backup) {
            // Use storage backend to backup data
            torch_memory_saver::StorageBackendInterface* backend = get_storage_backend(metadata.backend_type);
            if (backend == nullptr) {
                std::cerr << "[torch_memory_saver.cpp] Failed to get storage backend for type: "
                          << static_cast<int>(metadata.backend_type) << std::endl;
                exit(1);
            }

            std::string object_key = generate_object_key(ptr, metadata);
            cudaError_t err = backend->backup(ptr, metadata.size, object_key, &metadata.backup_handle);

            if (err != cudaSuccess) {
                std::cerr << "[torch_memory_saver.cpp] Failed to backup allocation: " << cudaGetErrorString(err) << std::endl;
                exit(1);
            }

            // For CPU backend, also set cpu_backup for backward compatibility
            if (metadata.backend_type == torch_memory_saver::StorageBackendType::CPU_MEMORY) {
                metadata.cpu_backup = metadata.backup_handle;
            }
        }

        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
        CURESULT_CHECK(cuMemRelease(metadata.allocHandle));

        metadata.state = AllocationState::PAUSED;

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause"
                  << " ptr=" << ptr << " metadata.size=" << metadata.size << " metadata.allocHandle="
                  << metadata.allocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << " backend_type=" << static_cast<int>(metadata.backend_type)
                  << std::endl;
#endif
    }
#else
    #error "USE_PLATFORM is not set"
#endif
}

void TorchMemorySaver::resume(const std::string& tag) {
#if defined(USE_ROCM)
    ROCmHIPImplementation::rocm_resume(tag, allocation_metadata_, allocator_metadata_mutex_);

#elif defined(USE_CUDA)
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata &metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        if (metadata.state != AllocationState::PAUSED) {
            std::cerr << "[torch_memory_saver.cpp] Cannot resume allocation that is not paused. "
                      << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)ptr)
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }

        CUmemGenericAllocationHandle newAllocHandle;
        CUDA_ERROR_CHECK(CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device));

        CURESULT_CHECK(cuMemMap((CUdeviceptr) ptr, metadata.size, 0, newAllocHandle, 0));

        CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);

        if (metadata.enable_cpu_backup) {
            // Use storage backend to restore data
            torch_memory_saver::StorageBackendInterface* backend = get_storage_backend(metadata.backend_type);
            if (backend == nullptr) {
                std::cerr << "[torch_memory_saver.cpp] Failed to get storage backend for type: "
                          << static_cast<int>(metadata.backend_type) << std::endl;
                exit(1);
            }

            SIMPLE_CHECK(metadata.backup_handle != nullptr, "backup_handle should not be nullptr");

            std::string object_key = generate_object_key(ptr, metadata);
            cudaError_t err = backend->restore(metadata.backup_handle, ptr, metadata.size, object_key);

            if (err != cudaSuccess) {
                std::cerr << "[torch_memory_saver.cpp] Failed to restore allocation: " << cudaGetErrorString(err) << std::endl;
                exit(1);
            }

            // Deallocate backup resources
            backend->deallocate(metadata.backup_handle);
            metadata.backup_handle = nullptr;
            metadata.cpu_backup = nullptr;
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.resume"
                  << " ptr=" << ptr << " metadata.size=" << metadata.size << " (old)metadata.allocHandle="
                  << metadata.allocHandle
                  << " (new)newAllocHandle=" << newAllocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << " backend_type=" << static_cast<int>(metadata.backend_type)
                  << std::endl;
#endif

        metadata.state = AllocationState::ACTIVE;
        metadata.allocHandle = newAllocHandle;
    }
#else
    #error "USE_PLATFORM is not set"
#endif
}

uint8_t* TorchMemorySaver::get_cpu_backup_pointer(const uint8_t* query_gpu_ptr, uint64_t query_size) {
#if defined(USE_ROCM)
    exit(1); // unsupported

#elif defined(USE_CUDA)
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        uint8_t *ptr = (uint8_t*) it->first;
        AllocationMetadata &metadata = it->second;

        if ((ptr <= query_gpu_ptr) && (query_gpu_ptr + query_size <= ptr + metadata.size)) {
            const size_t offset = query_gpu_ptr - ptr;
            if (metadata.state == AllocationState::ACTIVE) {
                return nullptr;
            } else {
                SIMPLE_CHECK(nullptr != metadata.backup_handle,
                    "get_cpu_backup_pointer: found paused allocation but backup_handle does not exist, do you forget to enable cpu backup");

                // Use storage backend to get CPU pointer
                torch_memory_saver::StorageBackendInterface* backend = get_storage_backend(metadata.backend_type);
                if (backend == nullptr) {
                    std::cerr << "[torch_memory_saver.cpp] Failed to get storage backend" << std::endl;
                    exit(1);
                }

                return backend->get_cpu_backup_pointer(metadata.backup_handle, offset);
            }
        }
    }

    std::cerr << "[torch_memory_saver.cpp] get_cpu_backup_pointer fail to find backup "
              << " query_gpu_ptr=" << query_gpu_ptr << " query_size=" << query_size
              << std::endl;
    exit(1);
#else
    #error "USE_PLATFORM is not set"
#endif
}
