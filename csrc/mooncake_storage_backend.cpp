#include "mooncake_storage_backend.h"

// Include Mooncake headers
#include "real_client.h"
#include "replica.h"

#include <span>
#include <iostream>

namespace torch_memory_saver {

// MooncakeBackupHandle destructor
MooncakeBackupHandle::~MooncakeBackupHandle() {
    // Cleanup will be done by MooncakeStorageBackend::deallocate
}

MooncakeStorageBackend::MooncakeStorageBackend(const MooncakeConfig& config)
    : config_(config)
    , mooncake_client_(nullptr)
    , replicate_config_(nullptr)
{
    if (!initialize_client()) {
        std::cerr << "[MooncakeStorageBackend] Failed to initialize Mooncake client" << std::endl;
    }
}

MooncakeStorageBackend::~MooncakeStorageBackend() {
    // Clean up all active handles
    {
        std::lock_guard<std::mutex> lock(handles_mutex_);
        for (auto& [key, handle] : active_handles_) {
            if (handle) {
                // Remove from Mooncake store
                if (mooncake_client_) {
                    mooncake_client_->remove(key);
                }

                // Unregister and free intermediate buffer
                if (handle->intermediate_cpu_buffer) {
                    if (mooncake_client_ && handle->buffer_registered) {
                        mooncake_client_->unregister_buffer(handle->intermediate_cpu_buffer);
                    }
                    cudaFreeHost(handle->intermediate_cpu_buffer);
                }

                delete handle;
            }
        }
        active_handles_.clear();
    }

    // Teardown Mooncake client
    if (mooncake_client_) {
        mooncake_client_->tearDownAll();
    }
}

bool MooncakeStorageBackend::initialize_client() {
    try {
        // 1. Create Mooncake client
        mooncake_client_ = mooncake::RealClient::create();
        if (!mooncake_client_) {
            std::cerr << "[MooncakeStorageBackend] Failed to create RealClient" << std::endl;
            return false;
        }

        // 2. Setup real client
        int result = mooncake_client_->setup_real(
            config_.local_hostname,
            config_.metadata_server,
            config_.global_segment_size,
            config_.local_buffer_size,
            config_.protocol,
            config_.rdma_devices,
            config_.master_server_addr,
            nullptr,  // transfer_engine (auto-created)
            ""        // ipc_socket_path
        );

        if (result != 0) {
            std::cerr << "[MooncakeStorageBackend] setup_real failed with code: "
                      << result << std::endl;
            mooncake_client_.reset();
            return false;
        }

        // 3. Initialize all components
        result = mooncake_client_->initAll(
            config_.protocol,
            "",  // device_name
            config_.global_segment_size
        );

        if (result != 0) {
            std::cerr << "[MooncakeStorageBackend] initAll failed with code: "
                      << result << std::endl;
            mooncake_client_->tearDownAll();
            mooncake_client_.reset();
            return false;
        }

        // 4. Setup replicate configuration
        replicate_config_ = std::make_unique<mooncake::ReplicateConfig>();
        replicate_config_->replica_num = config_.replica_num;
        replicate_config_->with_soft_pin = config_.with_soft_pin;
        replicate_config_->prefer_alloc_in_same_node = config_.prefer_alloc_in_same_node;

        std::cout << "[MooncakeStorageBackend] Successfully initialized"
                  << " hostname=" << config_.local_hostname
                  << " metadata=" << config_.metadata_server
                  << " protocol=" << config_.protocol
                  << " master=" << config_.master_server_addr
                  << " replica_num=" << config_.replica_num
                  << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[MooncakeStorageBackend] Exception during initialization: "
                  << e.what() << std::endl;
        if (mooncake_client_) {
            mooncake_client_->tearDownAll();
            mooncake_client_.reset();
        }
        return false;
    }
}

cudaError_t MooncakeStorageBackend::backup(
    const void* gpu_ptr,
    size_t size,
    const std::string& key,
    void** backup_handle
) {
    if (!mooncake_client_) {
        std::cerr << "[MooncakeStorageBackend] Client not initialized" << std::endl;
        return cudaErrorInitializationError;
    }

    try {
        // 1. Allocate temporary CPU buffer (pinned memory for fast transfer)
        void* cpu_buffer = nullptr;
        cudaError_t err = cudaMallocHost(&cpu_buffer, size);
        if (err != cudaSuccess) {
            std::cerr << "[MooncakeStorageBackend] Failed to allocate CPU buffer: "
                      << cudaGetErrorString(err) << std::endl;
            return err;
        }

        // 2. Copy data from GPU to CPU
        err = cudaMemcpy(cpu_buffer, gpu_ptr, size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "[MooncakeStorageBackend] Failed to copy GPU to CPU: "
                      << cudaGetErrorString(err) << std::endl;
            cudaFreeHost(cpu_buffer);
            return err;
        }

        // 3. Register buffer with Mooncake for zero-copy (optional optimization)
        bool buffer_registered = false;
        int register_result = mooncake_client_->register_buffer(cpu_buffer, size);
        if (register_result == 0) {
            buffer_registered = true;
#ifdef TMS_DEBUG_LOG
            std::cout << "[MooncakeStorageBackend] Buffer registered with Mooncake" << std::endl;
#endif
        } else {
            std::cerr << "[MooncakeStorageBackend] Warning: Failed to register buffer, "
                      << "will use copy mode (code: " << register_result << ")" << std::endl;
        }

        // 4. Write data to Mooncake store
        std::span<const char> data_span(static_cast<const char*>(cpu_buffer), size);
        int put_result = mooncake_client_->put(key, data_span, *replicate_config_);

        if (put_result != 0) {
            std::cerr << "[MooncakeStorageBackend] Failed to put data to Mooncake: "
                      << put_result << std::endl;
            if (buffer_registered) {
                mooncake_client_->unregister_buffer(cpu_buffer);
            }
            cudaFreeHost(cpu_buffer);
            return cudaErrorUnknown;
        }

        // 5. Create backup handle
        MooncakeBackupHandle* handle = new MooncakeBackupHandle();
        handle->key = key;
        handle->intermediate_cpu_buffer = cpu_buffer;
        handle->buffer_size = size;
        handle->buffer_registered = buffer_registered;

        // 6. Track the handle
        {
            std::lock_guard<std::mutex> lock(handles_mutex_);
            active_handles_[key] = handle;
        }

        *backup_handle = handle;

#ifdef TMS_DEBUG_LOG
        std::cout << "[MooncakeStorageBackend] backup: key=" << key
                  << " size=" << size << " gpu_ptr=" << gpu_ptr
                  << " handle=" << handle << std::endl;
#endif

        return cudaSuccess;

    } catch (const std::exception& e) {
        std::cerr << "[MooncakeStorageBackend] Exception in backup: " << e.what() << std::endl;
        return cudaErrorUnknown;
    }
}

cudaError_t MooncakeStorageBackend::restore(
    void* backup_handle,
    void* gpu_ptr,
    size_t size,
    const std::string& key
) {
    if (!mooncake_client_) {
        std::cerr << "[MooncakeStorageBackend] Client not initialized" << std::endl;
        return cudaErrorInitializationError;
    }

    if (backup_handle == nullptr) {
        std::cerr << "[MooncakeStorageBackend] Invalid backup handle" << std::endl;
        return cudaErrorInvalidValue;
    }

    try {
        MooncakeBackupHandle* handle = static_cast<MooncakeBackupHandle*>(backup_handle);

        // 1. Allocate temporary CPU buffer for reading
        void* cpu_buffer = nullptr;
        cudaError_t err = cudaMallocHost(&cpu_buffer, size);
        if (err != cudaSuccess) {
            std::cerr << "[MooncakeStorageBackend] Failed to allocate CPU buffer: "
                      << cudaGetErrorString(err) << std::endl;
            return err;
        }

        // 2. Read data from Mooncake store
        int64_t bytes_read = mooncake_client_->get_into(handle->key, cpu_buffer, size);

        if (bytes_read < 0) {
            std::cerr << "[MooncakeStorageBackend] Failed to get data from Mooncake: "
                      << bytes_read << std::endl;
            cudaFreeHost(cpu_buffer);
            return cudaErrorUnknown;
        }

        if (static_cast<size_t>(bytes_read) != size) {
            std::cerr << "[MooncakeStorageBackend] Size mismatch: expected=" << size
                      << " got=" << bytes_read << std::endl;
            cudaFreeHost(cpu_buffer);
            return cudaErrorUnknown;
        }

        // 3. Copy data from CPU to GPU
        err = cudaMemcpy(gpu_ptr, cpu_buffer, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "[MooncakeStorageBackend] Failed to copy CPU to GPU: "
                      << cudaGetErrorString(err) << std::endl;
            cudaFreeHost(cpu_buffer);
            return err;
        }

        // 4. Free temporary CPU buffer
        cudaFreeHost(cpu_buffer);

        // 5. Clean up intermediate buffer from backup
        if (handle->intermediate_cpu_buffer) {
            if (handle->buffer_registered) {
                mooncake_client_->unregister_buffer(handle->intermediate_cpu_buffer);
                handle->buffer_registered = false;
            }
            cudaFreeHost(handle->intermediate_cpu_buffer);
            handle->intermediate_cpu_buffer = nullptr;
        }

        // 6. Optionally remove object from Mooncake store to save space
        // (Uncomment if you want to delete after restore)
        // mooncake_client_->remove(handle->key);

#ifdef TMS_DEBUG_LOG
        std::cout << "[MooncakeStorageBackend] restore: key=" << handle->key
                  << " size=" << size << " gpu_ptr=" << gpu_ptr << std::endl;
#endif

        return cudaSuccess;

    } catch (const std::exception& e) {
        std::cerr << "[MooncakeStorageBackend] Exception in restore: " << e.what() << std::endl;
        return cudaErrorUnknown;
    }
}

uint8_t* MooncakeStorageBackend::get_cpu_backup_pointer(
    void* backup_handle,
    size_t offset
) {
    // Mooncake storage backend does not support direct CPU access
    // Data must be explicitly retrieved using restore()
    // or loaded on-demand (not implemented in this version)

    // For compatibility, we could load the data to intermediate_cpu_buffer
    // if it's still available
    if (backup_handle == nullptr) {
        return nullptr;
    }

    MooncakeBackupHandle* handle = static_cast<MooncakeBackupHandle*>(backup_handle);

    // If intermediate buffer is still available (before restore), return pointer
    if (handle->intermediate_cpu_buffer) {
        return static_cast<uint8_t*>(handle->intermediate_cpu_buffer) + offset;
    }

    // Otherwise, would need to fetch from Mooncake store
    // Not implemented for now - return nullptr
    std::cerr << "[MooncakeStorageBackend] Warning: get_cpu_backup_pointer called but "
              << "intermediate buffer not available. Use restore() instead." << std::endl;
    return nullptr;
}

void MooncakeStorageBackend::deallocate(void* backup_handle) {
    if (backup_handle == nullptr) {
        return;
    }

    try {
        MooncakeBackupHandle* handle = static_cast<MooncakeBackupHandle*>(backup_handle);

        // 1. Remove from active handles tracking
        {
            std::lock_guard<std::mutex> lock(handles_mutex_);
            active_handles_.erase(handle->key);
        }

        // 2. Remove object from Mooncake store
        if (mooncake_client_) {
            int result = mooncake_client_->remove(handle->key);
            if (result != 0) {
                std::cerr << "[MooncakeStorageBackend] Warning: Failed to remove key "
                          << handle->key << " from Mooncake (code: " << result << ")" << std::endl;
            }
        }

        // 3. Clean up intermediate buffer if still exists
        if (handle->intermediate_cpu_buffer) {
            if (mooncake_client_ && handle->buffer_registered) {
                mooncake_client_->unregister_buffer(handle->intermediate_cpu_buffer);
            }
            cudaFreeHost(handle->intermediate_cpu_buffer);
            handle->intermediate_cpu_buffer = nullptr;
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[MooncakeStorageBackend] deallocate: key=" << handle->key << std::endl;
#endif

        // 4. Delete the handle
        delete handle;

    } catch (const std::exception& e) {
        std::cerr << "[MooncakeStorageBackend] Exception in deallocate: "
                  << e.what() << std::endl;
    }
}

} // namespace torch_memory_saver
