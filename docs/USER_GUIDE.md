# 🚀 User Guide - Unified Pose Pipeline

## Complete Installation & Usage

### 1. Clone Repository
```bash
git clone https://github.com/pradeepj247/unified_pose_pipeline
cd unified_pose_pipeline
```

### 2. Install Dependencies
```bash
# Method A: Using requirements.txt
pip install -r requirements.txt

# Method B: Using installation script
chmod +x install_dependencies.sh
./install_dependencies.sh

# Method C: Install individually
pip install torch torchvision ultralytics boxmot rtmlib
pip install opencv-python numpy Pillow supervision
```

### 3. Run Demo
```bash
python demo.py
```

### 4. Use in Your Code
```python
import cv2
from pipeline.unified_pipeline import UnifiedPosePipeline

# Initialize pipeline
pipeline = UnifiedPosePipeline(
    tracker_type='ocsort',
    confidence_threshold=0.5,
    device='cuda'
)

# Process video
results = pipeline.run_complete_pipeline(
    input_video='your_video.mp4',
    output_dir='results'
)
```

## Outputs Generated
- `stage1_tracking.mp4` - Multi-person tracking with IDs
- `unifiedpipelineoutput.mp4` - Tracking + Pose skeleton
- `longest_running_person_ID.txt` - Analysis report
- `person_X_bboxes.json` - Bounding box coordinates

## Performance
- Tracking: ~31 FPS
- Pose Estimation: ~16 FPS
- Overall: ~11 FPS

## Troubleshooting
If you get import errors:
```bash
pip install --upgrade boxmot rtmlib ultralytics
```