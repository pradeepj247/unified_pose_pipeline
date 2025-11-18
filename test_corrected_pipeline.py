#!/usr/bin/env python3
"""
Test script for Unified Pose Pipeline
"""

import sys
import os

# Add unified_pose_pipeline to path
sys.path.insert(0, '/content/unified_pose_pipeline')

try:
    from pipeline.unified_pipeline import UnifiedPosePipeline
    print("✅ UnifiedPosePipeline imports successfully!")
    
    # Test initialization
    pipeline = UnifiedPosePipeline()
    print("✅ Pipeline initialization successful!")
    
    # Test running the pipeline
    video_path = '/content/pjboxmot/test_videos/Virat.mp4'
    if os.path.exists(video_path):
        print(f"✅ Video found: {video_path}")
        # Run with limited frames for quick test
        results = pipeline.run_complete_pipeline(video_path, max_frames=100)
        print(f"✅ Pipeline completed: {results}")
    else:
        print("❌ Video not found")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()