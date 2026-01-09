#pragma once

#include "storage_backend_interface.h"
#include <unordered_map>
#include <mutex>
#include <memory>

// Forward declarations to avoid including full Mooncake headers
namespace mooncake {
class RealClient;
struct ReplicateConfig;
}

namespace torch_memory_saver {

/**
 * Configuration for Mooncake storage backend
 */
struct MooncakeConfig {
    std::string local_hostname = "127.0.0.1:0";
    std::string metadata_server = "P2PHANDSHAKE";
    std::string protocol = "tcp";
    std::string master_server_addr = "127.0.0.1:50051";
    std::string rdma_devices = "";
    size_t global_segment_size = 16 * 1024 * 1024;  // 16MB
    size_t local_buffer_size = 16 * 1024 * 1024;    // 16MB
    size_t replica_num = 2;
    bool with_soft_pin = true;
    bool prefer_alloc_in_same_node = false;
};

/**
 * Backup handle for Mooncake backend
 * Stores the key string used to identify the object in Mooncake store
 */
struct MooncakeBackupHandle {
    std::string key;
    void* intermediate_cpu_buffer;  // Temporary buffer for data transfer
    size_t buffer_size;
    bool buffer_registered;  // Whether buffer is registered with Mooncake

    MooncakeBackupHandle()
        : intermediate_cpu_buffer(nullptr)
        , buffer_size(0)
        , buffer_registered(false) {}

    ~MooncakeBackupHandle();
};

/**
 * Mooncake storage backend - uses Mooncake distributed store for backup
 */
class MooncakeStorageBackend : public StorageBackendInterface {
public:
    /**
     * Constructor
     * @param config Mooncake configuration
     */
    explicit MooncakeStorageBackend(const MooncakeConfig& config);

    ~MooncakeStorageBackend() override;

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
        return StorageBackendType::MOONCAKE_STORE;
    }

    /**
     * Check if the backend is properly initialized
     */
    bool is_initialized() const { return mooncake_client_ != nullptr; }

private:
    MooncakeConfig config_;
    std::shared_ptr<mooncake::RealClient> mooncake_client_;
    std::unique_ptr<mooncake::ReplicateConfig> replicate_config_;

    // Track intermediate buffers for cleanup
    std::unordered_map<std::string, MooncakeBackupHandle*> active_handles_;
    std::mutex handles_mutex_;

    /**
     * Initialize Mooncake client
     * @return true on success, false on failure
     */
    bool initialize_client();
};

} // namespace torch_memory_saver
