#!/usr/bin/env python3
"""
Demo script for Unified Pose Pipeline
Uses included test video for immediate testing
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.unified_pipeline import UnifiedPosePipeline

def main():
    print("🚀 Unified Pose Pipeline Demo")
    print("==============================")
    
    # Initialize pipeline
    pipeline = UnifiedPosePipeline(
        tracker_type='ocsort',
        confidence_threshold=0.5,
        device='cuda'
    )
    
    # Use the included test video
    video_path = 'test_videos/campusWalk.mp4'
    
    if os.path.exists(video_path):
        print(f"🎬 Processing included demo video: {video_path}")
        print("Note: This will process the first 100 frames for quick testing")
        print("")

        
        results = pipeline.run_complete_pipeline(
            input_video=video_path,
            max_frames=100  # Limit for quick demo
        )
        print(f"")

        print(f"✅ Demo completed successfully!")
        print(f"📊 Results: {results}")
    else:
        print(f"❌ Test video not found: {video_path}")
        print("💡 The video should be included in the repository")
        print("💡 If missing, please check the repository structure")

if __name__ == "__main__":
    main()