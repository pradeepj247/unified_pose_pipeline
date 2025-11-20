# Unified Pose Pipeline

🚀 **Combine Object Tracking & 2D Pose Estimation in One Efficient Pipeline - Optimized for Google Colab**

## Features
- **Multi-Person Tracking**: BoxMOT-based tracking (OcSort, ByteTrack, etc.)
- **2D Pose Estimation**: RTMPose for accurate pose detection
- **Efficient Two-Stage Processing**: No redundant computations
- **Smart Person Selection**: Automatically finds longest-tracked person
- **Comprehensive Outputs**: Videos, JSON data, and analysis reports
- **Colab Optimized**: One-command installation with GPU acceleration

## 🚀 Quick Start (Google Colab)

### One-Command Installation
```bash
# Clone and install in one command
!git clone https://github.com/pradeepj247/unified_pose_pipeline
%cd unified_pose_pipeline
!bash install_colab_final.sh
```

### Manual Installation (if needed)
```bash
# Clone repository
!git clone https://github.com/pradeepj247/unified_pose_pipeline
%cd unified_pose_pipeline

# Install all dependencies
!pip install torch>=2.0.0 torchvision>=0.15.0
!pip install ultralytics>=8.0.0 opencv-python>=4.5.0 Pillow>=9.0.0
!pip install boxmot>=10.0.0 supervision>=0.15.0 filterpy>=1.4.5
!pip install scipy>=1.7.0 scikit-learn>=1.0.0 rtmlib>=0.0.6
!pip install requests>=2.25.0 lapx>=0.5.0 PyYAML>=6.0

# Install ONNX Runtime GPU separately (critical for performance)
!pip uninstall -y onnxruntime onnxruntime-gpu
!pip install onnxruntime-gpu==1.23.0 --no-deps
!pip install coloredlogs flatbuffers packaging protobuf sympy --upgrade
```

### Verify Installation
```bash
python verify_accurate.py
```

## Performance Optimization
- **GPU Acceleration**: ONNX Runtime with CUDA support for 2x faster pose estimation
- **Tesla T4 Optimized**: Tested and verified on Colab's T4 GPU
- **Performance**: ~19 FPS pose estimation with GPU vs ~8 FPS without

## Usage
```python
import sys
sys.path.append('.')  # Add current directory to path

from pipeline.unified_pipeline import UnifiedPosePipeline

# Initialize pipeline
pipeline = UnifiedPosePipeline(
    tracker_type='ocsort',  # Options: ocsort, bytetrack, botsort, strongsort
    confidence_threshold=0.5,
    device='cuda'  # Use 'cuda' for GPU acceleration
)

# Run complete pipeline
results = pipeline.run_complete_pipeline(
    input_video='path/to/your/video.mp4',
    max_frames=1000  # Optional: limit frames for testing
)
```

## Simple Usage
```python
from pipeline.unified_pipeline import UnifiedPosePipeline

pipeline = UnifiedPosePipeline()
pipeline.run_complete_pipeline('your_video.mp4')
```

## Outputs
- `stage1_tracking.mp4` - All tracked persons with IDs
- `unifiedpipelineoutput.mp4` - Tracking + Pose skeleton
- `longest_running_person_ID.txt` - Analysis report
- `person_X_bboxes.json` - Bounding box data

## Performance
- **Tracking Stage**: ~40 FPS
- **Pose Stage**: ~19 FPS (with GPU acceleration)
- **Overall**: Efficient real-time processing

## 🎬 Demo
The repository includes sample videos for immediate testing:

```bash
# Run the demo
python demo.py
```

The demo will:
1. Track all persons in the video using BoxMOT
2. Identify the longest-running person
3. Estimate 2D pose for that person using RTMPose
4. Generate output videos and analysis files

## 📁 Project Structure
```
unified_pose_pipeline/
├── install_colab_final.sh    # 🆕 One-command Colab installation
├── verify_accurate.py        # 🆕 Comprehensive verification
├── docs/                     # 📚 Documentation
├── performance_tests/        # 🧪 Performance testing scripts
├── pipeline/                 # 🔧 Core pipeline code
├── configs/                  # ⚙️ Configuration files
├── examples/                 # 💡 Usage examples
├── test_videos/              # 🎬 Sample videos for testing
└── outputs/                  # 📊 Output directory
```

## Troubleshooting
- **Missing CUDAExecutionProvider**: Run the installation script again
- **Import errors**: Ensure you're in the correct directory and ran verification
- **Performance issues**: Verify ONNX Runtime is using GPU with `verify_accurate.py`

## License
MIT License
