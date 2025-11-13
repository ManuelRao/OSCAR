"""
Diagnostic script to understand why prediction mapping isn't triggering.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import cv2 as cv
import numpy as np
from src.tracking import MultiObjectTracker, TrackingConfig, TrackingVisualizer


def main():
    # Create config with prediction enabled
    config = TrackingConfig()
    config.set_max_objects(2) \
          .set_prediction_enabled(True) \
          .set_prediction_steps(3) \
          .set_importance_map_enabled(True)
    
    print("Configuration:")
    print(f"  use_prediction: {config.use_prediction}")
    print(f"  prediction_steps: {config.prediction_steps}")
    print(f"  prediction_min_speed_object: {config.prediction_min_speed_object}")
    print(f"  prediction_min_speed_track: {config.prediction_min_speed_track}")
    print(f"  use_importance_map: {config.use_importance_map}")
    print()
    
    tracker = MultiObjectTracker(config)
    visualizer = TrackingVisualizer(tracker)
    
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    print("Starting tracker with prediction diagnostics...")
    print("Move objects quickly (>10 px/frame) to trigger prediction")
    print()
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        stats = tracker.update(frame)
        frame_count += 1
        
        # Diagnostic output every 30 frames
        if frame_count % 30 == 0:
            print(f"\n=== Frame {frame_count} Diagnostics ===")
            print(f"Objects: {len(tracker.get_objects())}")
            
            for obj in tracker.get_objects():
                speed = obj.get_speed()
                print(f"\nObject #{obj.id}:")
                print(f"  Position: {obj.position}")
                print(f"  Velocity: {obj.velocity}")
                print(f"  Speed: {speed:.2f} px/frame")
                print(f"  Tracks in pool: {len(obj.track_pool)}")
                
                # Check if prediction should trigger
                if speed > config.prediction_min_speed_object:
                    print(f"  ✓ Speed ABOVE object threshold ({config.prediction_min_speed_object}) - object predictions WILL trigger")
                else:
                    print(f"  ✗ Speed BELOW object threshold ({config.prediction_min_speed_object}) - object predictions WON'T trigger")
                
                # Check importance map
                if obj.importance_map:
                    pred_map_max = np.max(obj.importance_map.prediction_map)
                    pred_map_sum = np.sum(obj.importance_map.prediction_map)
                    print(f"  Prediction map - Max: {pred_map_max:.2f}, Sum: {pred_map_sum:.2f}")
                    
                    if pred_map_max > 0.1:
                        print(f"  ✓ Prediction map HAS data")
                    else:
                        print(f"  ✗ Prediction map is EMPTY")
                else:
                    print(f"  ✗ No importance map initialized")
                
                # Check tracks
                for track in obj.track_pool:
                    track_speed = track.get_speed()
                    print(f"    Track #{track.id}: speed={track_speed:.2f}")
                    if track_speed > config.prediction_min_speed_track:
                        print(f"      ✓ Track speed above {config.prediction_min_speed_track} - should add predictions")
                    else:
                        print(f"      ✗ Track speed below {config.prediction_min_speed_track} - won't add predictions")
        
        # Visualization
        visualizer.show_all(frame, stats)
        
        key = visualizer.wait_key(1)
        should_quit = visualizer.handle_key(key)
        if should_quit:
            break
    
    cap.release()
    visualizer.close_all()
    
    print("\n=== Final Summary ===")
    print(f"Total frames: {frame_count}")
    print(f"Objects tracked: {len(tracker.get_objects())}")
    
    for obj in tracker.get_objects():
        if obj.importance_map:
            pred_max = np.max(obj.importance_map.prediction_map)
            print(f"Object #{obj.id} prediction map max value: {pred_max:.2f}")


if __name__ == "__main__":
    main()
