#pragma once
#include <sys/types.h>
#include <stdio.h>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <string>
#include <memory>
#include "utils.h"
#include "macro.h"
#include "storage_backend_interface.h"
#include "cpu_storage_backend.h"
#include "mooncake_storage_backend.h"

#if defined(USE_ROCM)
#include "hardware_amd_support.h"
#endif

enum class AllocationState {
    // Memory is mapped and accessible
    ACTIVE,
    // Memory is unmapped and inaccessible
    PAUSED
};

struct AllocationMetadata {
    size_t size;
    CUdevice device;
    std::string tag;
    AllocationState state;
    bool enable_cpu_backup;

    // Storage backend support (NEW)
    torch_memory_saver::StorageBackendType backend_type;
    void* backup_handle;  // Backend-specific backup handle

    // Legacy field (kept for backward compatibility with old enable_cpu_backup behavior)
    // When backend_type == CPU_MEMORY, backup_handle points to the same memory as cpu_backup
    void* cpu_backup;

#if defined(USE_CUDA)
    CUmemGenericAllocationHandle allocHandle;
#elif defined(USE_ROCM)
    size_t aligned_size;
    std::vector<hipMemGenericAllocationHandle_t> allocHandles;
    std::vector<size_t> chunk_sizes;
#else
    #error "USE_PLATFORM is not set"
#endif
};

class TorchMemorySaver {
public:
    static TorchMemorySaver& instance();

    cudaError_t malloc(void** ptr, CUdevice device, size_t size, const std::string& tag, bool enable_cpu_backup);
    cudaError_t malloc_with_backend(void** ptr, CUdevice device, size_t size, const std::string& tag,
                                    torch_memory_saver::StorageBackendType backend_type);
    cudaError_t free(void* ptr);

    void pause(const std::string& tag);
    void resume(const std::string& tag);
    void set_memory_margin_bytes(uint64_t value) {
        memory_margin_bytes_.store(value);
    }
    uint8_t* get_cpu_backup_pointer(const uint8_t* query_gpu_ptr, uint64_t query_size);

    // Storage backend configuration
    void set_storage_backend_type(torch_memory_saver::StorageBackendType type);
    void set_mooncake_config(const torch_memory_saver::MooncakeConfig& config);
    torch_memory_saver::StorageBackendType get_current_backend_type() const;

private:
    TorchMemorySaver();
    ~TorchMemorySaver() = default;
    TorchMemorySaver(const TorchMemorySaver&) = delete;
    TorchMemorySaver& operator=(const TorchMemorySaver&) = delete;

    // Get or create storage backend
    torch_memory_saver::StorageBackendInterface* get_storage_backend(torch_memory_saver::StorageBackendType type);
    std::string generate_object_key(void* ptr, const AllocationMetadata& metadata);

    std::mutex allocator_metadata_mutex_;
    std::unordered_map<void*, AllocationMetadata> allocation_metadata_;
    std::atomic<uint64_t> memory_margin_bytes_ = 0;

    // Storage backend management
    std::unordered_map<torch_memory_saver::StorageBackendType,
                      std::unique_ptr<torch_memory_saver::StorageBackendInterface>> storage_backends_;
    torch_memory_saver::StorageBackendType current_backend_type_ = torch_memory_saver::StorageBackendType::CPU_MEMORY;
    torch_memory_saver::MooncakeConfig mooncake_config_;
    std::mutex backend_mutex_;
};
