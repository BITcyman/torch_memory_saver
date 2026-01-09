#pragma once

#include "storage_backend_interface.h"
#include <iostream>

namespace torch_memory_saver {

/**
 * CPU storage backend - uses pinned CPU memory for backup
 * This is the default implementation, equivalent to the original behavior
 */
class CPUStorageBackend : public StorageBackendInterface {
public:
    CPUStorageBackend() = default;
    ~CPUStorageBackend() override = default;

    cudaError_t backup(
        const void* gpu_ptr,
        size_t size,
        const std::string& key,
        void** backup_handle
    ) override;

    cudaError_t restore(
        void* backup_handle,
        void* gpu_ptr,
        size_t size,
        const std::string& key
    ) override;

    uint8_t* get_cpu_backup_pointer(
        void* backup_handle,
        size_t offset
    ) override;

    void deallocate(void* backup_handle) override;

    StorageBackendType get_type() const override {
        return StorageBackendType::CPU_MEMORY;
    }
};

} // namespace torch_memory_saver
