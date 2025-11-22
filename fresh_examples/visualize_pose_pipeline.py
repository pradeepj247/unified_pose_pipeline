# visualize_pose_pipeline.py
"""
Visualization Pipeline with Video Output
- YOLOv8s for person detection  
- RTMPose-M for pose estimation
- Real-time visualization with skeletons and bounding boxes
- Output video with overlaid results
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
from rtmlib.tools import Body
from rtmlib.visualization.draw import draw_skeleton

def main():
    print("🎬 Pose Estimation Visualization Pipeline")
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
    
    print(f"📹 Input Video: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
    
    # Output video setup
    output_path = 'pose_estimation_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Timing storage
    timings = {
        'detection': [],
        'pose_estimation': [],
        'visualization': [],
        'total_per_frame': []
    }
    
    frame_count = 0
    
    print("\n🎨 Starting visualization pipeline...")
    start_total_time = time.time()
    
    while True:
        frame_start = time.time()
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        original_frame = frame.copy()
        
        # Person detection
        det_start = time.time()
        results = detector(frame, classes=[0], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        det_time = (time.time() - det_start) * 1000
        timings['detection'].append(det_time)
        
        vis_start = time.time()
        if len(boxes) > 0:
            # Get largest person
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            largest_idx = np.argmax(areas)
            x1, y1, x2, y2 = map(int, boxes[largest_idx])
            
            if x2 > x1 and y2 > y1:
                crop = original_frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    # Pose estimation
                    pose_start = time.time()
                    keypoints, scores = pose_estimator(crop)
                    pose_time = (time.time() - pose_start) * 1000
                    timings['pose_estimation'].append(pose_time)
                    
                    if len(keypoints) > 0:
                        # Adjust coordinates to original frame
                        keypoints[0, :, 0] += x1
                        keypoints[0, :, 1] += y1
                        
                        # Draw skeleton
                        frame = draw_skeleton(
                            frame, 
                            keypoints[0:1], 
                            scores[0:1], 
                            openpose_skeleton=True, 
                            kpt_thr=0.3
                        )
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add info text
                    cv2.putText(frame, f'Persons: {len(boxes)}', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f'Frame: {frame_count}', (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        vis_time = (time.time() - vis_start) * 1000
        timings['visualization'].append(vis_time)
        
        # Write frame to output video
        out.write(frame)
        
        frame_time = (time.time() - frame_start) * 1000
        timings['total_per_frame'].append(frame_time)
        
        frame_count += 1
        
        # Progress update every 50 frames
        if frame_count % 50 == 0:
            current_fps = 1000 / np.mean(timings['total_per_frame'][-50:])
            print(f"📊 Frame {frame_count}/{total_frames} - Current FPS: {current_fps:.1f}")
    
    cap.release()
    out.release()
    end_total_time = time.time()
    
    # Print final summary
    print("\n" + "=" * 50)
    print("✅ VISUALIZATION COMPLETED!")
    print("=" * 50)
    print(f"📊 Performance Summary:")
    print(f"   Frames processed: {frame_count}/{total_frames}")
    print(f"   Total time: {end_total_time - start_total_time:.2f} seconds")
    print(f"   Average FPS: {1000/np.mean(timings['total_per_frame']):.1f}")
    print(f"   Detection: {np.mean(timings['detection']):.2f} ms")
    print(f"   Pose: {np.mean(timings['pose_estimation']):.2f} ms")
    print(f"   Visualization: {np.mean(timings['visualization']):.2f} ms")
    print(f"   Total per frame: {np.mean(timings['total_per_frame']):.2f} ms")
    print(f"\n💾 Output video saved to: {output_path}")

if __name__ == "__main__":
    main()
