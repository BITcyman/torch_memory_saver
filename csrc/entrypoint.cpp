#include "utils.h"
#include "core.h"
#include "api_forwarder.h"
#include <optional>
#include "macro.h"

// ----------------------------------------------- threadlocal configs --------------------------------------------------

class ThreadLocalConfig {
public:
    std::string current_tag_ = "default";

    bool is_interesting_region() {
        if (!is_interesting_region_.has_value()) {
            is_interesting_region_ = get_bool_env_var("TMS_INIT_ENABLE");
        }
        return is_interesting_region_.value();
    }

    void set_interesting_region(bool value) {
        is_interesting_region_ = value;
    }

    bool enable_cpu_backup() {
        if (!enable_cpu_backup_.has_value()) {
            enable_cpu_backup_ = get_bool_env_var("TMS_INIT_ENABLE_CPU_BACKUP");
        }
        return enable_cpu_backup_.value();
    }

    void set_enable_cpu_backup(bool value) {
        enable_cpu_backup_ = value;
    }

private:
    std::optional<bool> is_interesting_region_;
    std::optional<bool> enable_cpu_backup_;
};
static thread_local ThreadLocalConfig thread_local_config;

// ------------------------------------------------- entrypoints :: hook ------------------------------------------------

#ifdef TMS_HOOK_MODE_PRELOAD
cudaError_t cudaMalloc(void **ptr, size_t size) {
    if (thread_local_config.is_interesting_region()) {
        return TorchMemorySaver::instance().malloc(
            ptr, CUDAUtils::cu_ctx_get_device(), size, thread_local_config.current_tag_, thread_local_config.enable_cpu_backup());
    } else {
        return APIForwarder::call_real_cuda_malloc(ptr, size);
    }
}

cudaError_t cudaFree(void *ptr) {
    return TorchMemorySaver::instance().free(ptr);
}
#endif

#ifdef TMS_HOOK_MODE_TORCH
extern "C" {
void *tms_torch_malloc(ssize_t size, int device, cudaStream_t stream) {
#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] entrypoint::tms_torch_malloc "
              << " size=" << size << " device=" << device << " stream=" << stream
              << std::endl;
#endif
    SIMPLE_CHECK(thread_local_config.is_interesting_region(), "only support interesting region");
    void *ptr;
    CUDA_ERROR_CHECK(TorchMemorySaver::instance().malloc(
        &ptr, CUDAUtils::cu_device_get(device), size, thread_local_config.current_tag_, thread_local_config.enable_cpu_backup()));
    return ptr;
}

void tms_torch_free(void *ptr, ssize_t ssize, int device, cudaStream_t stream) {
#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] entrypoint::tms_torch_free "
              << " ptr=" << ptr << " ssize=" << ssize << " device=" << device << " stream=" << stream
              << std::endl;
#endif
    SIMPLE_CHECK(thread_local_config.is_interesting_region(), "only support interesting region");
    CUDA_ERROR_CHECK(TorchMemorySaver::instance().free(ptr));
}
}
#endif

// ------------------------------------------------- entrypoints :: others ------------------------------------------------

extern "C" {
void tms_set_interesting_region(bool is_interesting_region) {
    thread_local_config.set_interesting_region(is_interesting_region);
}

bool tms_get_interesting_region() {
    return thread_local_config.is_interesting_region();
}

void tms_set_current_tag(const char* tag) {
    SIMPLE_CHECK(tag != nullptr, "tag should not be null");
    thread_local_config.current_tag_ = tag;
}

bool tms_get_enable_cpu_backup() {
    return thread_local_config.enable_cpu_backup();
}

void tms_set_enable_cpu_backup(bool enable_cpu_backup) {
    thread_local_config.set_enable_cpu_backup(enable_cpu_backup);
}

void set_memory_margin_bytes(uint64_t value) {
    TorchMemorySaver::instance().set_memory_margin_bytes(value);
}

void tms_pause(const char* tag) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    TorchMemorySaver::instance().pause(tag_str);
}

void tms_resume(const char* tag) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    TorchMemorySaver::instance().resume(tag_str);
}

uint8_t* tms_get_cpu_backup_pointer(const uint8_t* gpu_ptr, uint64_t size) {
    return TorchMemorySaver::instance().get_cpu_backup_pointer(gpu_ptr, size);
}

// --------- Storage backend configuration APIs ---------

void tms_set_storage_backend_type(const char* backend_type_str) {
    SIMPLE_CHECK(backend_type_str != nullptr, "backend_type_str should not be null");

    torch_memory_saver::StorageBackendType backend_type;
    std::string type_str(backend_type_str);

    if (type_str == "cpu" || type_str == "CPU_MEMORY") {
        backend_type = torch_memory_saver::StorageBackendType::CPU_MEMORY;
    } else if (type_str == "mooncake" || type_str == "MOONCAKE_STORE") {
        backend_type = torch_memory_saver::StorageBackendType::MOONCAKE_STORE;
    } else if (type_str == "nvme" || type_str == "NVME_DISK") {
        backend_type = torch_memory_saver::StorageBackendType::NVME_DISK;
    } else {
        std::cerr << "[entrypoint] Unknown storage backend type: " << type_str << std::endl;
        return;
    }

    TorchMemorySaver::instance().set_storage_backend_type(backend_type);
}

int tms_get_storage_backend_type() {
    return static_cast<int>(TorchMemorySaver::instance().get_current_backend_type());
}

void tms_set_mooncake_config(
    const char* local_hostname,
    const char* metadata_server,
    const char* protocol,
    const char* master_server_addr,
    const char* rdma_devices,
    uint64_t global_segment_size,
    uint64_t local_buffer_size,
    uint64_t replica_num,
    bool with_soft_pin,
    bool prefer_alloc_in_same_node
) {
    torch_memory_saver::MooncakeConfig config;

    if (local_hostname != nullptr) config.local_hostname = local_hostname;
    if (metadata_server != nullptr) config.metadata_server = metadata_server;
    if (protocol != nullptr) config.protocol = protocol;
    if (master_server_addr != nullptr) config.master_server_addr = master_server_addr;
    if (rdma_devices != nullptr) config.rdma_devices = rdma_devices;

    if (global_segment_size > 0) config.global_segment_size = global_segment_size;
    if (local_buffer_size > 0) config.local_buffer_size = local_buffer_size;
    if (replica_num > 0) config.replica_num = replica_num;

    config.with_soft_pin = with_soft_pin;
    config.prefer_alloc_in_same_node = prefer_alloc_in_same_node;

    TorchMemorySaver::instance().set_mooncake_config(config);
}
}
