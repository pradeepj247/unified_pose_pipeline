# Installation Guide for Unified Pose Pipeline

## Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- Google Colab T4 instance (tested)

## Step-by-Step Installation

### 1. Clone Repository
```bash
git clone https://github.com/pradeepj247/unified_pose_pipeline
cd unified_pose_pipeline
```

### 2. Install Dependencies

**CRITICAL: Use ONNX Runtime GPU version for optimal performance**
```bash
# Install core dependencies
pip install torch torchvision ultralytics opencv-python numpy Pillow

# Install tracking packages
pip install boxmot supervision filterpy scipy scikit-learn

# Install pose estimation - MUST USE GPU VERSION
pip uninstall -y onnxruntime
pip install onnxruntime-gpu==1.23.0
pip install rtmlib

# Install utilities
pip install requests lapx PyYAML
```

### 3. Verify Installation
```python
# Test critical imports
import torch, torchvision, ultralytics, boxmot, rtmlib, onnxruntime

# Verify GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"ONNX Runtime providers: {onnxruntime.get_available_providers()}")
```

### 4. Performance Optimization Notes
- **Without GPU ONNX**: ~8.86 FPS pose estimation
- **With GPU ONNX**: ~19.41 FPS pose estimation (+119% improvement)
- Always verify `CUDAExecutionProvider` is available in ONNX Runtime

## Troubleshooting

### Common Issues:
1. **Low FPS**: Check if ONNX Runtime is using GPU
2. **Import errors**: Ensure you're in the correct directory
3. **Model download failures**: Check internet connection

## Expected Performance on T4 GPU
- Tracking: 40+ FPS
- Pose Estimation: 19+ FPS
- Overall: 13+ FPS