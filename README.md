# Unified Pose Pipeline

🚀 **Combine Object Tracking & 2D Pose Estimation in One Efficient Pipeline**

## Features
- **Multi-Person Tracking**: BoxMOT-based tracking (OcSort, ByteTrack, etc.)
- **2D Pose Estimation**: RTMPose for accurate pose detection
- **Efficient Two-Stage Processing**: No redundant computations
- **Smart Person Selection**: Automatically finds longest-tracked person
- **Comprehensive Outputs**: Videos, JSON data, and analysis reports

## Quick Start
```bash
git clone https://github.com/pradeepj247/unified_pose_pipeline
cd unified_pose_pipeline
pip install -r requirements.txt
python demo.py
```

## Usage
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