#include "cpu_storage_backend.h"

namespace torch_memory_saver {

cudaError_t CPUStorageBackend::backup(
    const void* gpu_ptr,
    size_t size,
    const std::string& key,
    void** backup_handle
) {
    // 1. Allocate pinned CPU memory
    void* cpu_backup = nullptr;
    cudaError_t err = cudaMallocHost(&cpu_backup, size);
    if (err != cudaSuccess) {
        std::cerr << "[CPUStorageBackend] Failed to allocate CPU memory: "
                  << cudaGetErrorString(err) << std::endl;
        return err;
    }

    // 2. Copy data from GPU to CPU (synchronous)
    err = cudaMemcpy(cpu_backup, gpu_ptr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "[CPUStorageBackend] Failed to copy from GPU to CPU: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFreeHost(cpu_backup);
        return err;
    }

    // 3. Return the CPU pointer as backup handle
    *backup_handle = cpu_backup;

#ifdef TMS_DEBUG_LOG
    std::cout << "[CPUStorageBackend] backup: key=" << key
              << " size=" << size << " gpu_ptr=" << gpu_ptr
              << " cpu_backup=" << cpu_backup << std::endl;
#endif

    return cudaSuccess;
}

cudaError_t CPUStorageBackend::restore(
    void* backup_handle,
    void* gpu_ptr,
    size_t size,
    const std::string& key
) {
    if (backup_handle == nullptr) {
        std::cerr << "[CPUStorageBackend] restore: backup_handle is nullptr" << std::endl;
        return cudaErrorInvalidValue;
    }

    // Copy data from CPU to GPU (synchronous)
    cudaError_t err = cudaMemcpy(gpu_ptr, backup_handle, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "[CPUStorageBackend] Failed to copy from CPU to GPU: "
                  << cudaGetErrorString(err) << std::endl;
        return err;
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[CPUStorageBackend] restore: key=" << key
              << " size=" << size << " gpu_ptr=" << gpu_ptr
              << " cpu_backup=" << backup_handle << std::endl;
#endif

    return cudaSuccess;
}

uint8_t* CPUStorageBackend::get_cpu_backup_pointer(
    void* backup_handle,
    size_t offset
) {
    if (backup_handle == nullptr) {
        return nullptr;
    }
    return static_cast<uint8_t*>(backup_handle) + offset;
}

void CPUStorageBackend::deallocate(void* backup_handle) {
    if (backup_handle != nullptr) {
        cudaError_t err = cudaFreeHost(backup_handle);
        if (err != cudaSuccess) {
            std::cerr << "[CPUStorageBackend] Failed to free CPU memory: "
                      << cudaGetErrorString(err) << std::endl;
        }
#ifdef TMS_DEBUG_LOG
        else {
            std::cout << "[CPUStorageBackend] deallocate: cpu_backup="
                      << backup_handle << std::endl;
        }
#endif
    }
}

} // namespace torch_memory_saver
