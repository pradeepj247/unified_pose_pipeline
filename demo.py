#!/usr/bin/env python3
"""
Demo script for Unified Pose Pipeline
Optimized for GPU performance with ONNX Runtime GPU
"""

import os
import sys
import time
from pipeline.unified_pipeline import UnifiedPosePipeline

def check_environment():
    """Verify that GPU acceleration is available"""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            print("✅ GPU acceleration available (ONNX Runtime GPU)")
            return True
        else:
            print("⚠️  GPU acceleration NOT available - using CPU")
            print(f"Available providers: {providers}")
            return False
    except ImportError:
        print("❌ ONNX Runtime not installed properly")
        return False

def main():
    print("🚀 Unified Pose Pipeline Demo")
    print("==============================")

    
    # Check environment first
    if not check_environment():
        print("\n💡 Performance Tip: Install onnxruntime-gpu for 2x faster pose estimation")

    
    # Initialize pipeline with optimized settings
    print("🚀 Initializing Unified Pose Pipeline...")
    pipeline = UnifiedPosePipeline(
        tracker_type='ocsort',  # Options: ocsort, bytetrack, botsort, strongsort
        confidence_threshold=0.5,
        device='cuda'  # Use GPU for both tracking and pose estimation
    )
    
    # Demo video path
    demo_video = 'test_videos/campusWalk.mp4'
    if not os.path.exists(demo_video):
        print(f"❌ Demo video not found: {demo_video}")
        return
    
    print(f"🎬 Processing included demo video: {demo_video}")
    print("Note: This will process the first 100 frames for quick testing\n")
    
    # Run the pipeline
    results = pipeline.run_complete_pipeline(
        input_video=demo_video,
        max_frames=100  # Process first 100 frames for quick demo
    )
    
    print("\n✅ Demo completed successfully!")

    
    # Performance summary
    print("📊 Performance Summary:")
    print(f"   - Total Frames: {results['tracking_results']['frame_count']}")
    print(f"   - Overall FPS: {results['overall_fps']:.2f}")
    print(f"   - Tracking FPS: {results['tracking_results']['stage1_fps']:.2f}")
    print(f"   - Pose Estimation FPS: {results['pose_results']['stage2_fps']:.2f}")
    print(f"   - Target Person: ID {results['pose_results']['target_person']}")

    
    print("📁 Outputs saved to: unifiedpipelineoutputs/")

    
    # Performance expectations
    if results['pose_results']['stage2_fps'] < 15:
        print("💡 Performance Tip: Install 'onnxruntime-gpu' for 2x faster pose estimation")

if __name__ == "__main__":
    main()