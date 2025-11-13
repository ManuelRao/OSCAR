"""
Basic example of using the multi-object tracking library.

This example shows:
1. Creating a tracking configuration
2. Initializing the tracker
3. Processing frames from webcam
4. Visualizing results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2 as cv
from src.tracking import MultiObjectTracker, TrackingConfig, TrackingVisualizer


def main():
    # 1. Create and configure the tracker
    config = TrackingConfig()
    config.set_max_objects(2) \
          .set_blob_min_area(25) \
          .set_blob_max_area(2000) \
          .set_diff_threshold(20) \
          .set_temporal_smoothing(0.3) \
          .set_prediction_enabled(True) \
          .set_prediction_steps(5) \
          .set_importance_map_enabled(True)
    
    # 2. Initialize tracker with config
    tracker = MultiObjectTracker(config)
    
    # 3. Create visualizer
    visualizer = TrackingVisualizer(tracker)
    
    # 4. Open webcam
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    print("Multi-Object Tracker - Basic Example")
    print("=====================================")
    print("Controls:")
    print("  q - Quit")
    print("  s - Save diff map")
    print("  b - Toggle blob detection")
    print("  t - Toggle track visualization")
    print("  o - Toggle object visualization")
    print("  i - Toggle importance map")
    print("  p - Toggle prediction")
    print("  +/- - Change prediction steps")
    print("  1-9 - Set max objects")
    print("  c - Clear/reset tracking")
    print()
    
    # 5. Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break
        
        # Update tracker
        stats = tracker.update(frame)
        
        # Show visualizations
        visualizer.show_all(frame, stats)
        
        # Handle keyboard
        key = visualizer.wait_key(1)
        should_quit = visualizer.handle_key(key)
        if should_quit:
            break
    
    # 6. Cleanup
    cap.release()
    visualizer.close_all()
    
    print("\nFinal statistics:")
    print(f"  Total frames: {tracker.get_frame_count()}")
    print(f"  Active objects: {len(tracker.get_objects())}")
    print(f"  Active tracks: {len(tracker.get_tracks())}")


if __name__ == "__main__":
    main()
