# Unified Pose Pipeline

🚀 **Combine Object Tracking & 2D Pose Estimation in One Efficient Pipeline**

## Features
- **Multi-Person Tracking**: BoxMOT-based tracking (OcSort, ByteTrack, etc.)
- **2D Pose Estimation**: RTMPose for accurate pose detection
- **Efficient Two-Stage Processing**: No redundant computations
- **Smart Person Selection**: Automatically finds longest-tracked person
- **Comprehensive Outputs**: Videos, JSON data, and analysis reports

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/pradeepj247/unified_pose_pipeline
cd unified_pose_pipeline

# Install dependencies (choose one method)

# Method 1: Using requirements.txt
pip install -r requirements.txt

# Method 2: Using installation script
chmod +x install_dependencies.sh
./install_dependencies.sh

# Method 3: Manual installation
pip install torch torchvision ultralytics boxmot rtmlib opencv-python
```

### Usage
```python
from pipeline.unified_pipeline import UnifiedPosePipeline

# Initialize pipeline
pipeline = UnifiedPosePipeline(
    tracker_type='ocsort',  # Options: ocsort, bytetrack, botsort, strongsort
    confidence_threshold=0.5,
    device='cuda'  # or 'cpu'
)

# Run complete pipeline
results = pipeline.run_complete_pipeline(
    input_video='path/to/your/video.mp4',
    max_frames=1000  # Optional: limit frames for testing
)
```## Usage
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
- **Tracking Stage**: ~29 FPS
- **Pose Stage**: ~16 FPS
- **Overall**: Efficient real-time processing

## License
MIT License
### 🎬 Demo
The repository includes a real sample video (`test_videos/campusWalk.mp4`) for immediate testing:

```bash
# Run the demo (processes first 100 frames for quick testing)
python demo.py
```

The demo will:
1. Track all persons in the video using BoxMOT
2. Identify the longest-running person
3. Estimate 2D pose for that person using RTMPose
4. Generate output videos and analysis files in `unifiedpipelineoutputs/`
