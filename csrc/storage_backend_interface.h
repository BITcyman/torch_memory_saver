#pragma once

#include <string>
#include <memory>
#include "utils.h"
#include "macro.h"

namespace torch_memory_saver {

// Storage backend type enum
enum class StorageBackendType {
    CPU_MEMORY,      // Current CPU memory implementation
    MOONCAKE_STORE,  // Mooncake distributed storage
    NVME_DISK        // Local NVMe SSD (future extension)
};

// Abstract storage backend interface
class StorageBackendInterface {
public:
    virtual ~StorageBackendInterface() = default;

    /**
     * Backup data from GPU to storage
     * @param gpu_ptr GPU memory pointer
     * @param size Data size in bytes
     * @param key Object identifier (used for Mooncake and other backends)
     * @param backup_handle Output: backup handle (implementation-specific)
     * @return cudaSuccess on success, error code otherwise
     */
    virtual cudaError_t backup(
        const void* gpu_ptr,
        size_t size,
        const std::string& key,
        void** backup_handle
    ) = 0;

    /**
     * Restore data from storage to GPU
     * @param backup_handle Backup handle from backup()
     * @param gpu_ptr Target GPU memory pointer
     * @param size Data size in bytes
     * @param key Object identifier
     * @return cudaSuccess on success, error code otherwise
     */
    virtual cudaError_t restore(
        void* backup_handle,
        void* gpu_ptr,
        size_t size,
        const std::string& key
    ) = 0;

    /**
     * Get CPU-accessible backup pointer (only supported by CPU backend)
     * @param backup_handle Backup handle from backup()
     * @param offset Offset within the backup
     * @return CPU pointer to backup data, or nullptr if not supported
     */
    virtual uint8_t* get_cpu_backup_pointer(
        void* backup_handle,
        size_t offset
    ) = 0;

    /**
     * Deallocate backup resources
     * @param backup_handle Backup handle to deallocate
     */
    virtual void deallocate(void* backup_handle) = 0;

    /**
     * Get the backend type
     * @return Backend type enum
     */
    virtual StorageBackendType get_type() const = 0;
};

} // namespace torch_memory_saver
