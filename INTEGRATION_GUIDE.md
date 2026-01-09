# Torch Memory Saver + Mooncake Store Integration Guide

## 概述

本指南介绍如何将 `torch_memory_saver` 与 `Mooncake` 分布式存储系统集成，实现 GPU 显存备份到分布式存储的功能。

## 架构设计

### 存储后端架构

```
torch_memory_saver (Python层)
  ↓
存储后端抽象层 (C++)
  ├─ CPUStorageBackend (CPU内存)
  ├─ MooncakeStorageBackend (Mooncake分布式存储)
  └─ NVMeStorageBackend (本地NVMe SSD - 未来扩展)
```

### 关键组件

1. **StorageBackendInterface** (`csrc/storage_backend_interface.h`)
   - 抽象存储后端接口
   - 定义 `backup()`, `restore()`, `deallocate()` 等方法

2. **CPUStorageBackend** (`csrc/cpu_storage_backend.{h,cpp}`)
   - 原有的 CPU 内存备份实现
   - 使用 `cudaMallocHost()` 和 `cudaMemcpy()`

3. **MooncakeStorageBackend** (`csrc/mooncake_storage_backend.{h,cpp}`)
   - 新增的 Mooncake 存储后端
   - 使用 Mooncake store 的 `put()` 和 `get_into()` API

4. **Python 配置层** (`torch_memory_saver/storage_config.py`)
   - `StorageBackend` 枚举：CPU, MOONCAKE, NVME
   - `MooncakeConfig` 数据类：Mooncake 配置参数

## 编译配置

### 修改 CMakeLists.txt

需要添加以下内容来链接 Mooncake 库：

```cmake
# Find Mooncake
find_path(MOONCAKE_INCLUDE_DIR
    NAMES real_client.h
    PATHS ${CMAKE_SOURCE_DIR}/../Mooncake/mooncake-store/include
)

find_library(MOONCAKE_LIBRARY
    NAMES mooncake_store
    PATHS ${CMAKE_SOURCE_DIR}/../Mooncake/mooncake-store/build
)

if(MOONCAKE_INCLUDE_DIR AND MOONCAKE_LIBRARY)
    message(STATUS "Found Mooncake: ${MOONCAKE_LIBRARY}")
    set(USE_MOONCAKE ON)
else()
    message(WARNING "Mooncake not found, building without Mooncake support")
    set(USE_MOONCAKE OFF)
endif()

# Add sources
set(TMS_SOURCES
    csrc/core.cpp
    csrc/cpu_storage_backend.cpp
    csrc/entrypoint.cpp
    csrc/api_forwarder.cpp
    csrc/utils.cpp
)

if(USE_MOONCAKE)
    list(APPEND TMS_SOURCES csrc/mooncake_storage_backend.cpp)
    add_definitions(-DUSE_MOONCAKE)
endif()

# Link libraries
if(USE_MOONCAKE)
    target_include_directories(torch_memory_saver PRIVATE ${MOONCAKE_INCLUDE_DIR})
    target_link_libraries(torch_memory_saver PRIVATE ${MOONCAKE_LIBRARY})
endif()
```

### 编译步骤

1. **编译 Mooncake**:
   ```bash
   cd Mooncake/mooncake-store
   mkdir build && cd build
   cmake ..
   make -j
   ```

2. **编译 torch_memory_saver**:
   ```bash
   cd torch_memory_saver
   pip install -e .
   ```

## 使用示例

### 基本用法

```python
import torch
from torch_memory_saver import torch_memory_saver, MooncakeConfig, StorageBackend

# 1. 配置 Mooncake
mooncake_config = MooncakeConfig(
    local_hostname="127.0.0.1:0",
    metadata_server="P2PHANDSHAKE",
    protocol="tcp",
    master_server_addr="127.0.0.1:50051",
    global_segment_size=1024 * 1024 * 1024,  # 1GB
    local_buffer_size=512 * 1024 * 1024,     # 512MB
    replica_num=2,
    with_soft_pin=True
)

torch_memory_saver.mooncake_config = mooncake_config

# 2. 使用 Mooncake 存储后端
with torch_memory_saver.region(
    tag="model_weights",
    storage_backend=StorageBackend.MOONCAKE
):
    model = LargeModel()
    model.load_state_dict(checkpoint)

# 3. 暂停（备份到 Mooncake）
torch_memory_saver.pause("model_weights")
# GPU 显存已释放，数据存储在 Mooncake

# 4. 恢复（从 Mooncake 加载）
torch_memory_saver.resume("model_weights")
# 数据从 Mooncake 恢复到 GPU
```

### 多模型快速切换

```python
# 加载多个模型
with torch_memory_saver.region(tag="model1", storage_backend=StorageBackend.MOONCAKE):
    model1 = Model1()

with torch_memory_saver.region(tag="model2", storage_backend=StorageBackend.MOONCAKE):
    model2 = Model2()

# 快速切换
for task in tasks:
    if task.use_model1:
        torch_memory_saver.pause("model2")
        torch_memory_saver.resume("model1")
        output = model1(task.input)
    else:
        torch_memory_saver.pause("model1")
        torch_memory_saver.resume("model2")
        output = model2(task.input)
```

### 混合使用不同存储后端

```python
# 重要数据使用 Mooncake（持久化，多副本）
with torch_memory_saver.region(
    tag="model_weights",
    storage_backend=StorageBackend.MOONCAKE
):
    model_weights = load_weights()

# 临时数据使用 CPU（速度快）
with torch_memory_saver.region(
    tag="kv_cache",
    storage_backend=StorageBackend.CPU
):
    kv_cache = torch.zeros(batch, seq_len, hidden_dim, device='cuda')
```

## API 参考

### StorageBackend (枚举)

```python
class StorageBackend(Enum):
    CPU = "cpu"          # CPU 内存备份
    MOONCAKE = "mooncake"  # Mooncake 分布式存储
    NVME = "nvme"        # 本地 NVMe SSD（未实现）
```

### MooncakeConfig (数据类)

```python
@dataclass
class MooncakeConfig:
    # 网络配置
    local_hostname: str = "127.0.0.1:0"
    metadata_server: str = "P2PHANDSHAKE"
    protocol: str = "tcp"  # "tcp" 或 "rdma"
    master_server_addr: str = "127.0.0.1:50051"
    rdma_devices: str = ""

    # 内存配置
    global_segment_size: int = 16 * 1024 * 1024  # 16MB
    local_buffer_size: int = 16 * 1024 * 1024    # 16MB

    # 副本配置
    replica_num: int = 2  # 副本数量
    with_soft_pin: bool = True  # 防止逐出
    prefer_alloc_in_same_node: bool = False  # 是否优选同节点
```

### TorchMemorySaver 方法

#### `mooncake_config` 属性

```python
torch_memory_saver.mooncake_config = MooncakeConfig(...)
config = torch_memory_saver.mooncake_config
```

#### `storage_backend` 属性

```python
# 设置默认存储后端
torch_memory_saver.storage_backend = StorageBackend.MOONCAKE

# 获取当前存储后端
current = torch_memory_saver.storage_backend
```

#### `region()` 方法

```python
with torch_memory_saver.region(
    tag="my_tag",
    enable_cpu_backup=False,  # 已弃用，使用 storage_backend 参数
    storage_backend=StorageBackend.MOONCAKE  # 指定存储后端
):
    # 在此区域内分配的张量将使用指定的存储后端
    tensor = torch.randn(1000, 1000, device='cuda')
```

## 性能对比

| 存储后端 | 备份速度 | 恢复速度 | 容量 | 持久化 | 多副本 |
|---------|---------|---------|------|--------|--------|
| CPU | 极快 (100GB/s+) | 极快 | 有限 (GB-TB) | 否 | 否 |
| Mooncake | 中等 (1-10GB/s) | 中等 | 无限 (PB+) | 是 | 是 |
| NVMe | 快 (3-7GB/s) | 快 | 大 (TB) | 是 | 否 |

## 故障排查

### 问题：Mooncake 连接失败

**症状**：
```
[MooncakeStorageBackend] Failed to initialize Mooncake client
setup_real failed with code: -1
```

**解决方案**：
1. 确认 Mooncake master server 正在运行：
   ```bash
   curl http://127.0.0.1:50051/health
   ```
2. 检查网络配置和防火墙设置
3. 验证 `master_server_addr` 配置正确

### 问题：编译时找不到 Mooncake 头文件

**症状**：
```
fatal error: real_client.h: No such file or directory
```

**解决方案**：
1. 确认 Mooncake 已编译：
   ```bash
   ls Mooncake/mooncake-store/build/libmooncake_store.so
   ```
2. 检查 CMakeLists.txt 中的路径配置
3. 使用 `-DMOONCAKE_INCLUDE_DIR` 和 `-DMOONCAKE_LIBRARY` 指定路径：
   ```bash
   pip install -e . --config-settings=cmake.args="-DMOONCAKE_INCLUDE_DIR=/path/to/include"
   ```

### 问题：运行时找不到共享库

**症状**：
```
ImportError: libmooncake_store.so: cannot open shared object file
```

**解决方案**：
1. 添加到 LD_LIBRARY_PATH：
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/Mooncake/mooncake-store/build
   ```
2. 或者在 `/etc/ld.so.conf.d/` 中添加路径并运行 `ldconfig`

## 高级特性

### 批量操作优化

Mooncake 支持批量 put/get 操作以提高吞吐量。可以在 `MooncakeStorageBackend` 中实现 `batch_backup()` 和 `batch_restore()` 方法。

### 异步传输

可以利用 CUDA streams 和 Mooncake 的异步 API 实现重叠传输：
- 在 backup 时使用 `cudaMemcpyAsync()`
- 在 restore 时使用异步的 Mooncake get 操作

### 数据压缩

在备份前对数据进行压缩可以显著减少网络传输和存储空间：
```cpp
// 在 MooncakeStorageBackend::backup() 中
compressed_data = compress(gpu_data);  // 使用 LZ4、Zstd 等
mooncake_client_->put(key, compressed_data, config);
```

## 贡献指南

如果您想为此集成贡献代码或报告问题，请：

1. 查看 GitHub Issues
2. 提交 Pull Request 时包含：
   - 清晰的问题描述
   - 测试用例
   - 性能基准测试结果（如适用）

## 许可证

本集成遵循 torch_memory_saver 和 Mooncake 各自的许可证。

## 参考资料

- [torch_memory_saver 官方文档](https://github.com/fzyzcjy/torch_memory_saver)
- [Mooncake 官方文档](https://github.com/mooncake-store/mooncake)
- [CUDA Virtual Memory Management API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__VIRTUAL__MEMORY__MANAGEMENT.html)
