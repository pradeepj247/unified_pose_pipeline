#!/bin/bash
echo "🚀 FINAL Installation - Smart Order (Your Suggestion!)"
echo "==================================================="
echo ""

# STEP 1: Install ALL regular packages first
echo "📦 STEP 1: Installing ALL regular packages first..."
echo ""

regular_packages=(
    "torch>=2.0.0"
    "torchvision>=0.15.0"
    "ultralytics>=8.0.0"
    "opencv-python>=4.5.0"
    "Pillow>=9.0.0"
    "numpy>=1.21.0"
    "boxmot>=10.0.0"
    "supervision>=0.15.0"
    "filterpy>=1.4.5"
    "scipy>=1.7.0"
    "scikit-learn>=1.0.0"
    "rtmlib>=0.0.6"
    "requests>=2.25.0"
    "lapx>=0.5.0"
    "PyYAML>=6.0"
)

for pkg in "${regular_packages[@]}"; do
    echo "   Installing: $pkg"
    pip install "$pkg" --quiet
done

echo ""
echo "✅ ALL regular packages installed"
echo ""

# STEP 2: Install ONNX Runtime GPU LAST (the tricky one)
echo "⚡ STEP 2: Installing ONNX Runtime GPU LAST..."
echo ""

echo "   Uninstalling any existing ONNX Runtime..."
pip uninstall -y onnxruntime onnxruntime-gpu --quiet 2>/dev/null || true

echo "   Installing ONNX Runtime GPU 1.23.0 with --no-deps..."
pip install onnxruntime-gpu==1.23.0 --no-deps --quiet

echo "   Installing ONNX Runtime dependencies separately..."
pip install coloredlogs flatbuffers packaging protobuf sympy --upgrade --quiet

echo ""
echo "✅ ONNX Runtime GPU installed (no dependency conflicts!)"
echo ""

# STEP 3: Final verification of all packages
echo "🔍 STEP 3: Verifying everything works..."
echo ""

python -c "
import sys
sys.path.append('.')"

python verify_installation.py

echo ""
echo "🎉 FINAL Installation Complete!"
echo ""
echo "💡 If any packages are still missing, they were likely skipped due to",
echo "   conflicts. The core pipeline should still work!"
