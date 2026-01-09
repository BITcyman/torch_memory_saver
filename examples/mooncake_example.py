"""
Example: Using torch_memory_saver with Mooncake distributed storage backend

This example demonstrates how to configure and use the Mooncake storage backend
for backing up GPU memory to a distributed storage system.
"""

import torch
from torch_memory_saver import torch_memory_saver, MooncakeConfig, StorageBackend


def main():
    # Step 1: Configure Mooncake storage backend
    print("Configuring Mooncake storage backend...")
    mooncake_config = MooncakeConfig(
        local_hostname="127.0.0.1:0",              # Local hostname:port (0 for auto-assign)
        metadata_server="P2PHANDSHAKE",            # Metadata service type
        protocol="rdma",                            # Transport protocol (tcp or rdma)
        rdma_devices="ibp12s0",                     # RDMA device
        master_server_addr="127.0.0.1:50051",     # Mooncake master server address
        global_segment_size=1024 * 1024 * 1024,   # 1GB segment size
        local_buffer_size=512 * 1024 * 1024,      # 512MB local buffer
        replica_num=2,                             # Number of replicas
        with_soft_pin=True,                        # Prevent eviction of important data
        prefer_alloc_in_same_node=False           # Distribute across nodes
    )

    # Apply the configuration
    torch_memory_saver.mooncake_config = mooncake_config

    # Step 2: Set Mooncake as the default storage backend
    torch_memory_saver.storage_backend = StorageBackend.MOONCAKE
    print(f"Current storage backend: {torch_memory_saver.storage_backend}")

    # Step 3: Create model weights using Mooncake storage backend
    print("\nCreating model weights with Mooncake backup...")
    with torch_memory_saver.region(tag="model_weights", storage_backend=StorageBackend.MOONCAKE):
        # Simulate large model weights
        weight1 = torch.randn(1000, 1000, device='cuda')
        weight2 = torch.randn(2000, 2000, device='cuda')
        print(f"Created weight1: {weight1.shape}, weight2: {weight2.shape}")

    # Step 4: Pause (backup) model weights to Mooncake store
    print("\nPausing model weights - backing up to Mooncake store...")
    torch_memory_saver.pause("model_weights")
    print("Model weights backed up to Mooncake and GPU memory released")

    # Step 5: Use GPU for other tasks
    print("\nGPU memory is now available for other tasks...")
    with torch_memory_saver.region(tag="temp_data", storage_backend=StorageBackend.CPU):
        temp_tensor = torch.randn(3000, 3000, device='cuda')
        print(f"Created temporary tensor: {temp_tensor.shape}")

    # Step 6: Resume (restore) model weights from Mooncake store
    print("\nResuming model weights - restoring from Mooncake store...")
    torch_memory_saver.pause("temp_data")  # Pause temp data first
    torch_memory_saver.resume("model_weights")
    print("Model weights restored from Mooncake to GPU")

    # Step 7: Use the restored weights
    print("\nUsing restored weights...")
    print(f"weight1 mean: {weight1.mean().item():.4f}")
    print(f"weight2 std: {weight2.std().item():.4f}")

    print("\n=== Example completed successfully! ===")


def compare_backends():
    """Compare different storage backends"""
    print("\n=== Comparing Storage Backends ===\n")

    # CPU Backend (default)
    print("1. CPU Backend:")
    with torch_memory_saver.region(tag="cpu_test", storage_backend=StorageBackend.CPU):
        cpu_tensor = torch.randn(1000, 1000, device='cuda')
    torch_memory_saver.pause("cpu_test")
    print("   - Fast backup to CPU memory")
    print("   - Limited by CPU RAM size")
    torch_memory_saver.resume("cpu_test")

    # Mooncake Backend
    print("\n2. Mooncake Backend:")
    if torch_memory_saver.mooncake_config:
        with torch_memory_saver.region(tag="mooncake_test", storage_backend=StorageBackend.MOONCAKE):
            mooncake_tensor = torch.randn(1000, 1000, device='cuda')
        torch_memory_saver.pause("mooncake_test")
        print("   - Distributed storage across multiple nodes")
        print("   - Unlimited storage capacity")
        print("   - Higher latency than CPU")
        torch_memory_saver.resume("mooncake_test")
    else:
        print("   - Not configured")

    print("\nBackend comparison complete!")


def multi_model_switching():
    """Example: Quick switching between multiple large models using Mooncake"""
    print("\n=== Multi-Model Quick Switching ===\n")

    # Ensure Mooncake is configured
    if not torch_memory_saver.mooncake_config:
        print("Mooncake not configured, skipping this example")
        return

    # Create Model 1
    print("Loading Model 1...")
    with torch_memory_saver.region(tag="model1", storage_backend=StorageBackend.MOONCAKE):
        model1_weights = [torch.randn(1000, 1000, device='cuda') for _ in range(5)]
    print("Model 1 loaded")

    # Create Model 2
    print("\nLoading Model 2...")
    with torch_memory_saver.region(tag="model2", storage_backend=StorageBackend.MOONCAKE):
        model2_weights = [torch.randn(1500, 1500, device='cuda') for _ in range(5)]
    print("Model 2 loaded")

    # Quick switching
    for i in range(3):
        print(f"\n--- Iteration {i+1} ---")

        # Switch to Model 1
        print("Switching to Model 1...")
        torch_memory_saver.pause("model2")
        torch_memory_saver.resume("model1")
        print(f"Model 1 active - weight sum: {sum(w.sum().item() for w in model1_weights):.2f}")

        # Switch to Model 2
        print("Switching to Model 2...")
        torch_memory_saver.pause("model1")
        torch_memory_saver.resume("model2")
        print(f"Model 2 active - weight sum: {sum(w.sum().item() for w in model2_weights):.2f}")

    print("\nMulti-model switching complete!")


if __name__ == "__main__":
    # Note: You need to have a Mooncake master server running at 127.0.0.1:50051
    # before running this example. See Mooncake documentation for setup instructions.

    try:
        main()
        compare_backends()
        multi_model_switching()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. Mooncake master server is running")
        print("2. CUDA is available")
        print("3. torch_memory_saver is properly installed with Mooncake support")
