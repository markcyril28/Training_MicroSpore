#!/usr/bin/env python3
"""Diagnose ROCm/PyTorch compatibility issues on MI210."""

import sys

def main():
    print("=" * 60)
    print("ROCm/PyTorch Diagnostic for MI210")
    print("=" * 60)
    
    # Check PyTorch
    try:
        import torch
        print(f"\n[PyTorch]")
        print(f"  Version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  HIP available: {torch.backends.cuda.is_built()}")
        
        if torch.cuda.is_available():
            print(f"  Device count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Total memory: {props.total_memory / 1e9:.1f} GB")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Check ROCm
    print(f"\n[ROCm Environment]")
    import os
    rocm_vars = ['ROCM_PATH', 'HIP_PATH', 'HSA_PATH', 'ROCBLAS_TENSILE_LIBPATH']
    for var in rocm_vars:
        print(f"  {var}: {os.environ.get(var, '(not set)')}")
    
    # Test basic operations
    print(f"\n[Testing Basic Operations]")
    
    if not torch.cuda.is_available():
        print("  SKIP: CUDA/HIP not available")
        return 1
    
    device = torch.device('cuda')
    
    # Test 1: Small tensor operations
    print("  Test 1: Small tensor add... ", end="", flush=True)
    try:
        a = torch.randn(10, 10, device=device)
        b = torch.randn(10, 10, device=device)
        c = a + b
        torch.cuda.synchronize()
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
    
    # Test 2: Small matmul
    print("  Test 2: Small matmul (10x10)... ", end="", flush=True)
    try:
        a = torch.randn(10, 10, device=device)
        b = torch.randn(10, 10, device=device)
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
    
    # Test 3: Medium matmul
    print("  Test 3: Medium matmul (512x512)... ", end="", flush=True)
    try:
        a = torch.randn(512, 512, device=device)
        b = torch.randn(512, 512, device=device)
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
    
    # Test 4: Large matmul (like the fc layer)
    print("  Test 4: Large matmul (512x4096 @ 4096x128)... ", end="", flush=True)
    try:
        a = torch.randn(512, 4096, device=device)
        b = torch.randn(4096, 128, device=device)
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
    
    # Test 5: nn.Linear
    print("  Test 5: nn.Linear(4096, 128)... ", end="", flush=True)
    try:
        import torch.nn as nn
        layer = nn.Linear(4096, 128).to(device)
        x = torch.randn(512, 4096, device=device)
        y = layer(x)
        torch.cuda.synchronize()
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
    
    # Test 6: nn.Linear with backward
    print("  Test 6: nn.Linear backward... ", end="", flush=True)
    try:
        import torch.nn as nn
        layer = nn.Linear(4096, 128).to(device)
        x = torch.randn(512, 4096, device=device, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        torch.cuda.synchronize()
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
    
    # Test 7: Conv2d
    print("  Test 7: Conv2d... ", end="", flush=True)
    try:
        import torch.nn as nn
        layer = nn.Conv2d(16, 64, 3, padding=1).to(device)
        x = torch.randn(32, 16, 8, 8, device=device)
        y = layer(x)
        torch.cuda.synchronize()
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
    
    # Test 8: Conv2d backward
    print("  Test 8: Conv2d backward... ", end="", flush=True)
    try:
        import torch.nn as nn
        layer = nn.Conv2d(16, 64, 3, padding=1).to(device)
        x = torch.randn(32, 16, 8, 8, device=device, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        torch.cuda.synchronize()
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
    
    print("\n[Recommendations]")
    print("  If tests fail, try:")
    print("  1. Reinstall PyTorch for your ROCm version:")
    print("     pip install torch --index-url https://download.pytorch.org/whl/rocm6.2")
    print("  2. Check ROCm installation: rocm-smi")
    print("  3. Verify driver: dmesg | grep -i amdgpu")
    print("  4. Try: export HSA_OVERRIDE_GFX_VERSION=9.0.0")
    print("")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
