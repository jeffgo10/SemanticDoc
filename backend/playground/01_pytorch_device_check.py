import platform
import time
import torch

def get_compute_device() -> torch.device:
    """
    Auto-detects the best available hardware accelerator for PyTorch:
    1. CUDA (NVIDIA GPUs on Linux/Windows servers)
    2. MPS (Apple Silicon Metal GPU when running natively on macOS)
    3. CPU (Fallback, e.g., inside a Linux Docker container on macOS)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def main():
    print(f"OS / Kernel : {platform.system()} ({platform.machine()})")
    print(f"PyTorch Ver : {torch.__version__}")
    
    device = get_compute_device()
    print(f"Active Device: {device} (MPS available: {torch.backends.mps.is_available()}, CUDA available: {torch.cuda.is_available()})")

    # Create two 3000x3000 tensors on the active device and multiply them
    size = 3000
    tensor_a = torch.randn(size, size, device=device)
    tensor_b = torch.randn(size, size, device=device)

    start = time.perf_counter()
    result = torch.matmul(tensor_a, tensor_b)
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"Matrix Multiply ({size}x{size}) -> Shape: {result.shape} | Time: {elapsed_ms:.2f} ms")

if __name__ == "__main__":
    main()
