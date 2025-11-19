#!/bin/bash
# install_dependencies.sh - UPDATED WITH GPU OPTIMIZATIONS

echo "🚀 Installing Unified Pose Pipeline Dependencies with GPU Optimization"
echo "================================================="

# Update pip
pip install --upgrade pip

# CRITICAL: Install GPU-accelerated ONNX Runtime
echo "Installing GPU-accelerated ONNX Runtime..."
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.23.0

# Install core dependencies
echo ""
echo "📦 Installing core dependencies..."
pip install torch torchvision

# Install computer vision packages
echo ""
echo "👁️ Installing computer vision packages..."
pip install ultralytics opencv-python numpy Pillow

# Install tracking packages
echo ""
echo "🎯 Installing tracking packages..."
pip install boxmot supervision filterpy scipy scikit-learn

# Install pose estimation
echo ""
echo "💃 Installing pose estimation..."
pip install rtmlib

# Install utilities
echo ""
echo "🔧 Installing utilities..."
pip install requests lapx PyYAML

echo ""
echo "✅ All dependencies installed successfully!"
echo ""
echo "🚀 Performance Note:"
echo "   - With GPU ONNX: ~19 FPS pose estimation"
echo "   - Without GPU ONNX: ~8 FPS pose estimation"
echo ""
echo "📋 Usage:"
echo "   python demo.py"
echo "   or"
echo "   from pipeline.unified_pipeline import UnifiedPosePipeline"