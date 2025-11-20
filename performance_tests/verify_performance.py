#!/usr/bin/env python3
"""
Performance Verification Script
Verifies that the pipeline is running with GPU acceleration
"""

import torch
import onnxruntime as ort
import sys
import os

def verify_environment():
    """Comprehensive environment verification"""
    print("🔍 Verifying Environment Setup")

    
    # 1. Check PyTorch GPU
    print("1. PyTorch GPU:")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
        print(f"   ✅ GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("   ❌ CUDA not available")

    
    # 2. Check ONNX Runtime
    print("2. ONNX Runtime:")
    providers = ort.get_available_providers()
    print(f"   Available providers: {providers}")
    
    if 'CUDAExecutionProvider' in providers:
        print("   ✅ CUDAExecutionProvider available")
        gpu_available = True
    else:
        print("   ❌ CUDAExecutionProvider NOT available - performance will be poor")
        gpu_available = False

    
    # 3. Check critical imports
    print("3. Critical Imports:")
    try:
        from pipeline.unified_pipeline import UnifiedPosePipeline
        print("   ✅ UnifiedPosePipeline import successful")
    except ImportError as e:
        print(f"   ❌ Pipeline import failed: {e}")

    
    # 4. Performance expectations
    print("4. Performance Expectations:")
    if gpu_available:
        print("   ✅ Expected: 19+ FPS pose estimation, 13+ FPS overall")
    else:
        print("   ⚠️  Expected: 8-9 FPS pose estimation, 6-7 FPS overall (CPU only)")

    
    return gpu_available

if __name__ == "__main__":
    verify_environment()