# Installation Guide for Unified Pose Pipeline - COLAB VERSION

## One-Command Installation for Google Colab
```bash
# Run this single command in Colab:
!git clone https://github.com/pradeepj247/unified_pose_pipeline
%cd unified_pose_pipeline
!bash install_colab_proper.sh
```

## Important Finding:
- **Do NOT run `pip install -e .`** - it causes dependency conflicts
- **The pipeline works without package installation** - imports work directly
- Just ensure you're in the project directory and imports will work

## Proper Installation Method:
1. Install ALL packages from requirements.txt first
2. Install onnxruntime-gpu==1.23.0 SEPARATELY with --no-deps
3. No need to install the package itself - imports work directly

## Usage Example:
```python
import sys
sys.path.append('.')  # Add current directory to path
from pipeline.unified_pipeline import UnifiedPosePipeline
# Your pipeline code here...
```

## Why This Works Better:
- Avoids all dependency conflicts from setup.py
- Cleaner and more reliable installation
- Easier to modify and test code
- Follows the original requirements.txt structure exactly

## Verification Script:
Run `python verify_installation.py` to check everything is working

## 🧪 Performance Testing

After installation, you can run performance tests:
```bash
cd performance_tests
python verify_performance.py
python test_corrected_pipeline.py
```

These scripts verify the pipeline performance and correctness.
