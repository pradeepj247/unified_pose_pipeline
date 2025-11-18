#!/usr/bin/env python3
"""
Demo script for Unified Pose Pipeline
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.unified_pipeline import UnifiedPosePipeline

def main():
    print("🚀 Unified Pose Pipeline Demo")
    
    # Initialize pipeline
    pipeline = UnifiedPosePipeline(
        tracker_type='ocsort',
        confidence_threshold=0.5,
        device='cuda'
    )
    
    # Run on sample video
    video_path = '/content/pjboxmot/test_videos/Virat.mp4'
    
    if os.path.exists(video_path):
        results = pipeline.run_complete_pipeline(
            input_video=video_path,
            max_frames=500  # Limit for testing
        )
        print(f"✅ Demo completed: {results}")
    else:
        print("❌ Video file not found. Please update video_path.")

if __name__ == "__main__":
    main()