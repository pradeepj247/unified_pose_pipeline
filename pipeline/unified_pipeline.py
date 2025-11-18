"""
Main Unified Pipeline Class - Corrected Path Version
Combines tracking/detection with 2D pose estimation
"""

import cv2
import time
import json
import os
import sys
import numpy as np
from collections import defaultdict


class UnifiedPosePipeline:
    """Main pipeline combining tracking/detection with 2D pose estimation"""
    
    def __init__(self, tracker_type='ocsort', confidence_threshold=0.5, 
                 device='cuda', output_dir='unifiedpipelineoutputs'):
        self.tracker_type = tracker_type
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("🚀 Initializing Unified Pose Pipeline...")
        
        # Ensure repos are in path
        self.setup_system_paths()
        
        # Initialize components
        self.setup_trackdet_components()
        self.setup_pose2d_components()
        
        # Tracking data storage
        self.track_history = defaultdict(list)
        self.frame_data = {}
    
    def setup_system_paths(self):
        """Ensure both repositories are in Python path"""
        repo_paths = [
            '/content/pjboxmot',
            '/content/pjpose2d'
        ]
        
        for path in repo_paths:
            if path not in sys.path:
                sys.path.insert(0, path)
                print(f"✅ Added to path: {path}")
            else:
                print(f"✅ Already in path: {path}")
    
    def setup_trackdet_components(self):
        """Setup tracking & detection components from BoxMOT"""
        print("🔧 Setting up TrackDet components...")
        
        try:
            from ultralytics import YOLO
            from boxmot import OcSort, ByteTrack, BotSort, StrongSort
            
            self.detector = YOLO('yolov8s.pt')
            tracker_config = {
                'model_weights': 'osnet_x0_25_msmt17.pt',
                'device': self.device, 'half': False, 'per_class': False,
                'det_thresh': 0.2, 'max_age': 30, 'min_hits': 3, 'iou_threshold': 0.3
            }
            
            trackers = {'ocsort': OcSort, 'bytetrack': ByteTrack, 
                       'botsort': BotSort, 'strongsort': StrongSort}
            self.tracker = trackers[self.tracker_type](**tracker_config)
            print("✅ TrackDet components initialized")
            
        except ImportError as e:
            print(f"❌ Failed to import TrackDet components: {e}")
            raise
    
    def setup_pose2d_components(self):
        """Setup 2D pose estimation components from RTMPose"""
        print("🔧 Setting up Pose2D components...")
        
        try:
            from rtmlib import Body, draw_skeleton
            
            self.pose_estimator = Body(
                to_openpose=True,
                mode='balanced',
                backend='onnxruntime', 
                device=self.device
            )
            self.draw_skeleton = draw_skeleton
            print("✅ Pose2D components initialized")
            
        except ImportError as e:
            print(f"❌ Failed to import Pose2D components: {e}")
            raise

    def stage1_trackdet(self, input_video, max_frames=None):
        """Stage 1: Track all persons and find the longest-running person"""
        print(f"\n{'='*60}")
        print("🎬 STAGE 1: Tracking & Detection")
        print(f"{'='*60}")
        
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"Input video not found: {input_video}")
        
        # Video setup
        cap = cv2.VideoCapture(input_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames and max_frames > 0:
            total_frames = min(total_frames, max_frames)
        
        # Stage 1 output video
        stage1_output = os.path.join(self.output_dir, 'stage1_tracking.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_stage1 = cv2.VideoWriter(stage1_output, fourcc, fps, (width, height))
        
        print(f"📹 Video: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
        print("⏳ Tracking all persons...")
        
        # Timing for Stage 1
        stage1_start = time.time()
        frame_count = 0
        frame_times = []
        
        while frame_count < total_frames:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret: 
                break
            
            # Run detection and tracking
            results = self.detector(frame, conf=self.confidence_threshold, verbose=False)
            detections = results[0].boxes.data.cpu().numpy()
            
            # Filter person detections (class 0)
            person_detections = []
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if cls == 0 and conf >= self.confidence_threshold:
                    person_detections.append([x1, y1, x2, y2, conf, cls])
            
            detections_array = np.array(person_detections) if person_detections else np.empty((0, 6))
            tracks = self.tracker.update(detections_array, frame)
            
            # Store tracking data and draw on frame
            current_frame_tracks = {}
            for track in tracks:
                if len(track) >= 6:
                    x1, y1, x2, y2 = map(int, track[:4])
                    track_id = int(track[4])
                    conf = track[5]
                    
                    # Store track data
                    self.track_history[track_id].append(frame_count)
                    current_frame_tracks[track_id] = [x1, y1, x2, y2]
                    
                    # Draw bounding box and ID
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"ID:{track_id} ({conf:.2f})"
                    cv2.putText(frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            self.frame_data[frame_count] = current_frame_tracks
            out_stage1.write(frame)
            frame_count += 1
            
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            
            # Progress reporting
            if frame_count % 50 == 0:
                elapsed = time.time() - stage1_start
                recent_fps = 50 / (elapsed - sum(frame_times[:-50])) if len(frame_times) > 50 else 0
                print(f"   Frame {frame_count:04d}/{total_frames} | "
                      f"Progress: {(frame_count/total_frames)*100:5.1f}% | "
                      f"FPS: {recent_fps:5.1f} | "
                      f"Active tracks: {len(current_frame_tracks):2d}")
        
        cap.release()
        out_stage1.release()
        
        # Stage 1 timing results
        stage1_time = time.time() - stage1_start
        stage1_fps = frame_count / stage1_time if stage1_time > 0 else 0
        
        print(f"\n✅ STAGE 1 COMPLETE:")
        print(f"   Frames processed: {frame_count}")
        print(f"   Total time: {stage1_time:.2f}s")
        print(f"   Average FPS: {stage1_fps:.2f}")
        print(f"   Output: {stage1_output}")
        
        return {
            'frame_count': frame_count,
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'stage1_time': stage1_time,
            'stage1_fps': stage1_fps
        }

    def analyze_tracking_results(self, tracking_results):
        """Analyze track history to find person with longest duration"""
        print(f"\n🔍 Analyzing track history...")
        
        if not self.track_history:
            raise ValueError("No tracking data found. Run Stage 1 first.")
        
        # Find person with most frames
        longest_person = None
        max_frames = 0
        start_frame = end_frame = 0
        
        for track_id, frames in self.track_history.items():
            if len(frames) > max_frames:
                max_frames = len(frames)
                longest_person = track_id
                start_frame = min(frames)
                end_frame = max(frames)
        
        print(f"🎯 Longest-running person: ID {longest_person}")
        print(f"   Frames: {max_frames}")
        print(f"   Duration: {start_frame} to {end_frame}")
        
        # Save to text file
        txt_output = os.path.join(self.output_dir, 'longest_running_person_ID.txt')
        with open(txt_output, 'w') as f:
            f.write(f"Longest Running Person Analysis\n")
            f.write(f"===============================\n")
            f.write(f"Person ID: {longest_person}\n")
            f.write(f"Frames Appeared: {max_frames}\n")
            f.write(f"Start Frame: {start_frame}\n")
            f.write(f"End Frame: {end_frame}\n")
            f.write(f"Track Duration: {end_frame - start_frame + 1} frames\n")
        
        print(f"💾 Saved analysis: {txt_output}")
        return longest_person

    def save_bbox_data(self, target_person_id):
        """Save bbox coordinates for the target person as JSON"""
        print(f"\n💾 Saving bbox data for person {target_person_id}...")
        
        bbox_data = {}
        for frame_id, tracks in self.frame_data.items():
            if target_person_id in tracks:
                bbox_data[frame_id] = tracks[target_person_id]
        
        json_output = os.path.join(self.output_dir, f'person_{target_person_id}_bboxes.json')
        with open(json_output, 'w') as f:
            json.dump(bbox_data, f, indent=2)
        
        print(f"   Saved bbox data: {json_output}")
        print(f"   Frames with bboxes: {len(bbox_data)}")
        return bbox_data

    def stage2_pose2d(self, input_video, target_person_id, tracking_results, bbox_data=None):
        """Stage 2: 2D pose estimation for the target person"""
        if bbox_data is None:
            bbox_data = self.save_bbox_data(target_person_id)
            
        print(f"\n{'='*60}")
        print("🎬 STAGE 2: 2D Pose Estimation")
        print(f"{'='*60}")
        
        # Stage 2 output video
        stage2_output = os.path.join(self.output_dir, 'unifiedpipelineoutput.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_stage2 = cv2.VideoWriter(stage2_output, fourcc, tracking_results['fps'], 
                                   (tracking_results['width'], tracking_results['height']))
        
        cap = cv2.VideoCapture(input_video)
        
        print(f"🎯 Processing 2D pose for Person ID: {target_person_id}")
        print(f"⏳ Frames to process: {len(bbox_data)}")
        
        # Timing for Stage 2
        stage2_start = time.time()
        frame_count = 0
        processed_frames = 0
        frame_times = []
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= tracking_results['total_frames']:
                break
            
            frame_start = time.time()
            
            if frame_count in bbox_data:
                # Get pre-computed bbox from Stage 1
                x1, y1, x2, y2 = bbox_data[frame_count]
                
                # Ensure valid crop
                if y2 > y1 and x2 > x1 and x1 >= 0 and y1 >= 0:
                    try:
                        crop = frame[y1:y2, x1:x2]
                        
                        if crop.size > 0:
                            # Run pose estimation on crop
                            keypoints, scores = self.pose_estimator.pose_model(crop)
                            
                            # Adjust coordinates back to original frame
                            keypoints[:, :, 0] += x1
                            keypoints[:, :, 1] += y1
                            
                            # Draw skeleton on original frame
                            frame = self.draw_skeleton(
                                frame, 
                                keypoints, 
                                scores, 
                                openpose_skeleton=True, 
                                kpt_thr=0.3
                            )
                            
                            processed_frames += 1
                    
                    except Exception as e:
                        # Silent error handling
                        pass
                
                # Draw tracking bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{target_person_id} (Pose2D)", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            out_stage2.write(frame)
            frame_count += 1
            
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            
            # Progress reporting
            if processed_frames % 30 == 0 and processed_frames > 0:
                elapsed = time.time() - stage2_start
                recent_fps = 30 / (elapsed - sum(frame_times[:-30])) if len(frame_times) > 30 else 0
                print(f"   Pose2D frames: {processed_frames:04d} | FPS: {recent_fps:5.1f}")
        
        cap.release()
        out_stage2.release()
        
        # Stage 2 timing results
        stage2_time = time.time() - stage2_start
        stage2_fps = processed_frames / stage2_time if stage2_time > 0 else 0
        
        print(f"\n✅ STAGE 2 COMPLETE:")
        print(f"   Pose2D frames processed: {processed_frames}")
        print(f"   Total time: {stage2_time:.2f}s")
        print(f"   Average FPS: {stage2_fps:.2f}")
        print(f"   Output: {stage2_output}")
        
        return {
            'processed_frames': processed_frames,
            'stage2_time': stage2_time,
            'stage2_fps': stage2_fps,
            'target_person': target_person_id
        }

    def run_complete_pipeline(self, input_video, max_frames=None):
        """Run the complete unified pipeline"""
        print("🚀 STARTING UNIFIED POSE PIPELINE")
        print("=" * 60)
        
        # Stage 1: Tracking and detection
        tracking_results = self.stage1_trackdet(input_video, max_frames)
        
        # Find longest-running person
        target_person = self.analyze_tracking_results(tracking_results)
        
        # Stage 2: 2D pose estimation
        pose_results = self.stage2_pose2d(input_video, target_person, tracking_results)
        
        # Final summary
        total_time = tracking_results['stage1_time'] + pose_results['stage2_time']
        overall_fps = tracking_results['frame_count'] / total_time if total_time > 0 else 0
        
        print(f"\n🎉 UNIFIED POSE PIPELINE COMPLETE!")
        print(f"📊 PERFORMANCE SUMMARY:")
        print(f"   Total Frames: {tracking_results['frame_count']}")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Overall FPS: {overall_fps:.2f}")
        print(f"   TrackDet FPS: {tracking_results['stage1_fps']:.2f}")
        print(f"   Pose2D FPS: {pose_results['stage2_fps']:.2f}")
        print(f"   Target Person: ID {pose_results['target_person']}")
        print(f"📁 All outputs saved to: {self.output_dir}/")
        
        return {
            'tracking_results': tracking_results,
            'pose_results': pose_results,
            'total_time': total_time,
            'overall_fps': overall_fps
        }