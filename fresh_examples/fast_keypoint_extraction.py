# fast_keypoint_extraction.py
"""
Fast Keypoint Extraction Pipeline
- YOLOv8s for person detection
- RTMPose-M for pose estimation  
- No visualization or video I/O for maximum speed
- Output: JSON file with keypoints and timings
"""

import cv2
import time
import numpy as np
import json
from ultralytics import YOLO
from rtmlib.tools import Body

def main():
    print("🚀 Fast Keypoint Extraction Pipeline")
    print("=" * 50)
    
    # Initialize models
    print("Initializing models...")
    detector = YOLO('yolov8s.pt')
    pose_estimator = Body(
        to_openpose=True, 
        mode='balanced', 
        backend='onnxruntime',
        device='cuda'
    )
    print("✓ Models loaded")
    
    # Video setup
    video_path = 'campus_walk.mp4'
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
    
    # Timing storage
    timings = {
        'detection': [],
        'pose_estimation': [],
        'total_per_frame': []
    }
    
    all_keypoints = []
    frame_count = 0
    
    print("\n⏱️ Starting keypoint extraction...")
    start_total_time = time.time()
    
    while True:
        frame_start = time.time()
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Person detection
        det_start = time.time()
        results = detector(frame, classes=[0], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        det_time = (time.time() - det_start) * 1000
        timings['detection'].append(det_time)
        
        pose_time = 0
        if len(boxes) > 0:
            # Get largest person
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            largest_idx = np.argmax(areas)
            x1, y1, x2, y2 = map(int, boxes[largest_idx])
            
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    # Pose estimation
                    pose_start = time.time()
                    keypoints, scores = pose_estimator(crop)
                    pose_time = (time.time() - pose_start) * 1000
                    
                    if len(keypoints) > 0:
                        # Adjust coordinates to original frame
                        keypoints[0, :, 0] += x1
                        keypoints[0, :, 1] += y1
                        
                        # Store results
                        all_keypoints.append({
                            'frame': frame_count,
                            'keypoints': keypoints[0].tolist(),
                            'scores': scores[0].tolist(),
                            'bbox': [x1, y1, x2, y2],
                            'detection_time_ms': det_time,
                            'pose_time_ms': pose_time
                        })
        
        timings['pose_estimation'].append(pose_time)
        frame_time = (time.time() - frame_start) * 1000
        timings['total_per_frame'].append(frame_time)
        
        frame_count += 1
        
        # Progress update every 50 frames
        if frame_count % 50 == 0:
            current_fps = 1000 / np.mean(timings['total_per_frame'][-50:])
            print(f"📊 Frame {frame_count}/{total_frames} - Current FPS: {current_fps:.1f}")
    
    cap.release()
    end_total_time = time.time()
    
    # Save keypoints to JSON
    output_file = 'extracted_keypoints.json'
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'video_path': video_path,
                'total_frames': total_frames,
                'frames_processed': frame_count,
                'total_processing_time_seconds': end_total_time - start_total_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'keypoints': all_keypoints,
            'performance': {
                'avg_detection_time_ms': np.mean(timings['detection']),
                'avg_pose_time_ms': np.mean(timings['pose_estimation']),
                'avg_total_time_ms': np.mean(timings['total_per_frame']),
                'avg_fps': 1000 / np.mean(timings['total_per_frame'])
            }
        }, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 50)
    print("✅ EXTRACTION COMPLETED!")
    print("=" * 50)
    print(f"📊 Performance Summary:")
    print(f"   Frames processed: {frame_count}/{total_frames}")
    print(f"   Total time: {end_total_time - start_total_time:.2f} seconds")
    print(f"   Average FPS: {1000/np.mean(timings['total_per_frame']):.1f}")
    print(f"   Detection: {np.mean(timings['detection']):.2f} ms")
    print(f"   Pose: {np.mean(timings['pose_estimation']):.2f} ms")
    print(f"   Total per frame: {np.mean(timings['total_per_frame']):.2f} ms")
    print(f"\n💾 Keypoints saved to: {output_file}")
    print(f"   Total keypoint frames: {len(all_keypoints)}")

if __name__ == "__main__":
    main()
