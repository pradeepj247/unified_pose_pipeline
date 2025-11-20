#!/usr/bin/env python3
"""
Working verification script with correct import names
"""

import sys
sys.path.insert(0, '.')

def check_import(import_name, display_name, pip_name=None):
    """Check if a package can be imported"""
    if pip_name is None:
        pip_name = display_name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {display_name} (import: {import_name}): {version}")
        return True
    except ImportError as e:
        print(f"❌ {display_name}: Cannot import as '{import_name}' - {e}")
        return False

print("🔍 ACCURATE VERIFICATION")
print("=" * 50)

# Critical packages - these must work
print("\n🎯 CRITICAL PACKAGES (Must Work):")
critical_ok = True
critical_ok &= check_import("torch", "PyTorch")
critical_ok &= check_import("onnxruntime", "ONNX Runtime GPU")
critical_ok &= check_import("pipeline.unified_pipeline", "UnifiedPosePipeline")

# Utility packages - these should work but aren't always critical
print("\n🔧 UTILITY PACKAGES (Should Work):")
utility_ok = True
utility_ok &= check_import("cv2", "OpenCV", "opencv-python")
utility_ok &= check_import("PIL", "Pillow", "Pillow")
utility_ok &= check_import("sklearn", "Scikit-learn", "scikit-learn")
utility_ok &= check_import("lap", "LAP", "lapx")
utility_ok &= check_import("yaml", "PyYAML", "PyYAML")

# Check CUDA and providers
print("\n⚡ HARDWARE ACCELERATION:")
try:
    import torch
    print(f"   PyTorch CUDA: {'✅ Available' if torch.cuda.is_available() else '❌ Not available'}")
    if torch.cuda.is_available():
        print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
except:
    print("   PyTorch CUDA: ❌ Could not check")

try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"   ONNX GPU: {'✅ Available' if 'CUDAExecutionProvider' in providers else '❌ Not available'}")
except:
    print("   ONNX GPU: ❌ Could not check")

# Final pipeline test
print("\n🚀 PIPELINE FUNCTIONAL TEST:")
try:
    from pipeline.unified_pipeline import UnifiedPosePipeline
    pipeline = UnifiedPosePipeline()
    print("   ✅ Pipeline instance created successfully!")
    pipeline_working = True
except Exception as e:
    print(f"   ❌ Pipeline creation failed: {e}")
    pipeline_working = False

print("\n" + "=" * 50)
print("📊 FINAL STATUS:")
if critical_ok and pipeline_working:
    print("🎉 SUCCESS: Core pipeline is READY TO USE!")
    print("   Even if some utilities are missing, the main functionality works.")
    print("   You can run: python demo.py")
else:
    print("❌ CRITICAL ISSUES: Core pipeline components are missing.")
    print("   Please check the installation.")
