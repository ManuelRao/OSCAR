"""
Visualization and debugging tools for multi-object tracking.

Provides visualization windows for:
- Blob detection
- Track trajectories
- Object fusion
- Importance maps
- Prediction corridors
"""

import cv2 as cv
import numpy as np
from typing import List, Optional
from .multi_object_tracker import MultiObjectTracker, Track, Object, BlobInfo


class TrackingVisualizer:
    """
    Provides visualization and debugging tools for tracking.
    
    Usage:
        visualizer = TrackingVisualizer(tracker)
        visualizer.show_frame(frame)
        visualizer.show_all()
        
        # Handle keyboard input
        key = visualizer.wait_key(1)
        if key == ord('q'):
            break
    """
    
    def __init__(self, tracker: MultiObjectTracker):
        self.tracker = tracker
        
        # Visualization toggles
        self.show_debug = True
        self.show_blobs = True
        self.show_tracks = True
        self.show_objects = True
        self.show_importance = False
        
        # Saved frame count
        self.saved_count = 0
    
    def draw_blobs(self, frame: np.ndarray, keypoints) -> np.ndarray:
        """Draw detected blobs on frame."""
        vis_frame = frame.copy()
        
        if self.show_blobs and keypoints:
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                size = int(kp.size)
                cv.circle(vis_frame, (x, y), size, (0, 255, 0), 2)
                cv.circle(vis_frame, (x, y), 3, (0, 0, 255), -1)
        
        return vis_frame
    
    def draw_tracks(self, frame: np.ndarray, tracks: List[Track], 
                    valuable_track_ids: set) -> np.ndarray:
        """Draw tracks with certainty visualization."""
        vis_frame = frame.copy()
        
        if not self.show_tracks or not tracks:
            return vis_frame
        
        # Pre-calculate scores
        track_scores = {}
        for obj in self.tracker.get_objects():
            for track in obj.track_pool:
                score = self.tracker._calculate_track_object_score(track, obj)
                track_scores[track.id] = score
        
        for track in tracks:
            # Skip if position has NaN values
            if np.any(np.isnan(track.position)):
                continue
            
            is_valuable = track.id in valuable_track_ids
            certainty = track.get_certainty()
            track_score = track_scores.get(track.id, 0) if is_valuable else 0
            
            # Color based on value
            if is_valuable:
                ring_color = (0, 255, 0)  # Green
            else:
                ring_color = (100, 100, 100)  # Gray
            
            track_color = track.color if is_valuable else (80, 80, 80)
            
            # Draw history trail
            if len(track.position_history) > 1:
                points = [tuple(map(int, pos_frame[0])) for pos_frame in track.position_history 
                         if not np.any(np.isnan(pos_frame[0]))]
                line_thickness = max(1, int(certainty * 3)) if is_valuable else 1
                for i in range(len(points) - 1):
                    cv.line(vis_frame, points[i], points[i + 1], track_color, line_thickness)
            
            # Draw current position
            pos = tuple(map(int, track.position))
            circle_size = 4 if is_valuable else 3
            cv.circle(vis_frame, pos, circle_size, track_color, -1)
            
            # Outer ring (size = score, thickness = certainty)
            if is_valuable:
                normalized_score = min(track_score / 100.0, 1.0)
                ring_radius = circle_size + int(2 + normalized_score * 10)
                ring_thickness = max(1, int(certainty * 3))
            else:
                ring_radius = circle_size + 2
                ring_thickness = 1
            
            cv.circle(vis_frame, pos, ring_radius, ring_color, ring_thickness)
            
            # Velocity arrow
            if is_valuable:
                speed = track.get_speed()
                if speed > 5:
                    end_pos = tuple(map(int, track.position + track.velocity * 0.3))
                    cv.arrowedLine(vis_frame, pos, end_pos, track_color, 2, tipLength=0.3)
            
            # Track ID
            font_scale = 0.5 if is_valuable else 0.3
            cv.putText(vis_frame, f"#{track.id}", 
                      (pos[0] + 15, pos[1] - 10),
                      cv.FONT_HERSHEY_SIMPLEX, font_scale, track_color, 2 if is_valuable else 1)
        
        return vis_frame
    
    def draw_objects(self, frame: np.ndarray, objects: List[Object]) -> np.ndarray:
        """Draw objects with their track pools."""
        vis_frame = frame.copy()
        
        if not self.show_objects or not objects:
            return vis_frame
        
        for obj in objects:
            # Skip if position has NaN values
            if np.any(np.isnan(obj.position)):
                continue
            
            obj_pos = tuple(map(int, obj.position))
            
            # Draw connections to tracks
            for track in obj.track_pool:
                if np.any(np.isnan(track.position)):
                    continue
                track_pos = tuple(map(int, track.position))
                certainty = track.get_certainty()
                correlation = obj.calculate_track_correlation(track)
                line_thickness = max(1, int(correlation * certainty * 3))
                cv.line(vis_frame, obj_pos, track_pos, obj.color, line_thickness)
            
            # Draw object circle
            cv.circle(vis_frame, obj_pos, 35, obj.color, 4)
            
            # Velocity arrow
            speed = obj.get_speed()
            if speed > 3 and not np.any(np.isnan(obj.velocity)):
                end_pos = tuple(map(int, obj.position + obj.velocity * 0.5))
                cv.arrowedLine(vis_frame, obj_pos, end_pos, obj.color, 4, tipLength=0.3)
            
            # Object ID and track count
            active_tracks = len(obj.track_pool)
            cv.putText(vis_frame, f"OBJ #{obj.id} [{active_tracks} tracks]", 
                      (obj_pos[0] - 50, obj_pos[1] - 45),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, obj.color, 2)
            
            # Track IDs
            if active_tracks > 0:
                track_ids = ",".join([f"#{t.id}" for t in obj.track_pool[:5]])
                cv.putText(vis_frame, track_ids, 
                          (obj_pos[0] - 50, obj_pos[1] + 50),
                          cv.FONT_HERSHEY_SIMPLEX, 0.4, obj.color, 1)
        
        return vis_frame
    
    def draw_stats(self, frame: np.ndarray, stats: dict) -> np.ndarray:
        """Draw statistics overlay."""
        vis_frame = frame.copy()
        
        # Get valuable track count
        valuable_track_ids = set()
        for obj in self.tracker.get_objects():
            valuable_track_ids.update([t.id for t in obj.track_pool])
        
        num_valuable = len(valuable_track_ids)
        num_noise = stats.get('num_tracks', 0) - num_valuable
        
        cv.putText(vis_frame, f"Frame: {stats.get('frame_count', 0)}", (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.putText(vis_frame, f"Blobs: {stats.get('num_blobs', 0)}", (10, 60), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.putText(vis_frame, f"Tracks: {stats.get('num_tracks', 0)} ({num_valuable} valuable, {num_noise} noise)", 
                  (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.putText(vis_frame, f"Objects: {stats.get('num_objects', 0)}/{self.tracker.config.max_objects}", 
                  (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.putText(vis_frame, f"Mean: {stats.get('mean_diff', 0):.1f} Max: {stats.get('max_diff', 0):.1f}", 
                  (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Toggles
        cv.putText(vis_frame, f"Blobs: {'ON' if self.show_blobs else 'OFF'} (b)", 
                  (10, 180), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv.putText(vis_frame, f"Tracks: {'ON' if self.show_tracks else 'OFF'} (t)", 
                  (10, 210), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv.putText(vis_frame, f"Objects: {'ON' if self.show_objects else 'OFF'} (o)", 
                  (10, 240), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv.putText(vis_frame, f"Importance: {'ON' if self.show_importance else 'OFF'} (i)", 
                  (10, 270), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        config = self.tracker.config
        cv.putText(vis_frame, 
                  f"Prediction: {'ON' if config.use_prediction else 'OFF'} (p) Steps: {config.prediction_steps} (+/-)", 
                  (10, 300), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return vis_frame
    
    def show_frame(self, frame: np.ndarray, stats: Optional[dict] = None):
        """
        Show main tracking visualization frame.
        
        Args:
            frame: Original input frame
            stats: Optional stats dict from tracker.update()
        """
        if stats is None:
            stats = {}
        
        # Get valuable track IDs
        valuable_track_ids = set()
        for obj in self.tracker.get_objects():
            valuable_track_ids.update([t.id for t in obj.track_pool])
        
        # Build visualization
        vis_frame = frame.copy()
        vis_frame = self.draw_blobs(vis_frame, self.tracker.current_keypoints)
        vis_frame = self.draw_tracks(vis_frame, self.tracker.get_tracks(), valuable_track_ids)
        vis_frame = self.draw_objects(vis_frame, self.tracker.get_objects())
        vis_frame = self.draw_stats(vis_frame, stats)
        
        cv.imshow("Tracking - Main View", vis_frame)
    
    def show_diff_map(self):
        """Show difference map."""
        diff_map = self.tracker.get_diff_map()
        if diff_map is not None:
            cv.imshow("Diff Map", diff_map)
            
            if self.show_debug:
                colored_diff = cv.applyColorMap(diff_map, cv.COLORMAP_JET)
                cv.imshow("Diff Map (Colored)", colored_diff)
    
    def show_importance_maps(self, frame: np.ndarray):
        """Show combined importance map and prediction corridors."""
        if not self.show_importance:
            try:
                cv.destroyWindow("Combined Importance Map")
                cv.destroyWindow("Prediction Corridors")
            except:
                pass
            return
        
        objects = self.tracker.get_objects()
        if not objects:
            return
        
        height, width = frame.shape[:2] if len(frame.shape) == 3 else frame.shape
        combined_positive = np.zeros((height, width), dtype=np.float32)
        combined_negative = np.zeros((height, width), dtype=np.float32)
        combined_prediction = np.zeros((height, width), dtype=np.float32)
        
        for obj in objects:
            if obj.importance_map:
                combined_positive += obj.importance_map.positive_map
                combined_negative += obj.importance_map.negative_map
                combined_prediction += obj.importance_map.prediction_map
        
        # Combined view
        full_combined = combined_positive + combined_prediction - combined_negative
        combined_vis = np.clip(full_combined, -20, 20)
        map_min = combined_vis.min()
        map_max = combined_vis.max()
        if map_max > map_min:
            combined_vis = ((combined_vis - map_min) / (map_max - map_min) * 255).astype(np.uint8)
        else:
            combined_vis = np.zeros_like(combined_vis, dtype=np.uint8)
        
        importance_colored = cv.applyColorMap(combined_vis, cv.COLORMAP_JET)
        importance_overlay = cv.addWeighted(frame, 0.4, importance_colored, 0.6, 0)
        
        label_text = f"Combined Importance ({len(objects)} objects)"
        cv.putText(importance_overlay, label_text, 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv.imshow("Combined Importance Map", importance_overlay)
        
        # Prediction corridors
        if self.tracker.config.use_prediction and np.max(combined_prediction) > 0.1:
            pred_vis = np.clip(combined_prediction, 0, 20)
            if pred_vis.max() > 0:
                pred_vis = (pred_vis / pred_vis.max() * 255).astype(np.uint8)
            else:
                pred_vis = np.zeros_like(pred_vis, dtype=np.uint8)
            
            pred_colored = cv.applyColorMap(pred_vis, cv.COLORMAP_HOT)
            pred_overlay = cv.addWeighted(frame, 0.5, pred_colored, 0.5, 0)
            
            cv.putText(pred_overlay, f"Prediction Corridors ({self.tracker.config.prediction_steps} steps)", 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(pred_overlay, "Red/Yellow = Predicted positions", 
                      (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv.imshow("Prediction Corridors", pred_overlay)
        else:
            try:
                cv.destroyWindow("Prediction Corridors")
            except:
                pass
    
    def show_all(self, frame: np.ndarray, stats: Optional[dict] = None):
        """Show all visualization windows."""
        self.show_frame(frame, stats)
        self.show_diff_map()
        self.show_importance_maps(frame)
    
    def handle_key(self, key: int) -> bool:
        """
        Handle keyboard input for visualization controls.
        
        Args:
            key: Key code from cv.waitKey()
        
        Returns:
            True if should quit, False otherwise
        """
        if key == ord('q'):
            return True
        elif key == ord('s'):
            diff_map = self.tracker.get_diff_map()
            if diff_map is not None:
                filename = f"diff_map_{self.saved_count:03d}.png"
                cv.imwrite(filename, diff_map)
                print(f"Saved: {filename}")
                self.saved_count += 1
        elif key == ord('d'):
            self.show_debug = not self.show_debug
            print(f"Debug views: {'ON' if self.show_debug else 'OFF'}")
            if not self.show_debug:
                try:
                    cv.destroyWindow("Diff Map (Colored)")
                except:
                    pass
        elif key == ord('b'):
            self.show_blobs = not self.show_blobs
            print(f"Blob detection: {'ON' if self.show_blobs else 'OFF'}")
        elif key == ord('t'):
            self.show_tracks = not self.show_tracks
            print(f"Track visualization: {'ON' if self.show_tracks else 'OFF'}")
        elif key == ord('o'):
            self.show_objects = not self.show_objects
            print(f"Object visualization: {'ON' if self.show_objects else 'OFF'}")
        elif key == ord('i'):
            self.show_importance = not self.show_importance
            print(f"Importance map: {'ON' if self.show_importance else 'OFF'}")
        elif key == ord('p'):
            self.tracker.config.use_prediction = not self.tracker.config.use_prediction
            print(f"Predictive importance: {'ON' if self.tracker.config.use_prediction else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            self.tracker.config.prediction_steps = min(10, self.tracker.config.prediction_steps + 1)
            print(f"Prediction steps: {self.tracker.config.prediction_steps}")
        elif key == ord('-') or key == ord('_'):
            self.tracker.config.prediction_steps = max(1, self.tracker.config.prediction_steps - 1)
            print(f"Prediction steps: {self.tracker.config.prediction_steps}")
        elif key >= ord('1') and key <= ord('9'):
            new_max = key - ord('0')
            if new_max != self.tracker.config.max_objects:
                self.tracker.config.max_objects = new_max
                objects = self.tracker.get_objects()
                if len(objects) > new_max:
                    self.tracker.objects = objects[:new_max]
                print(f"Max objects set to: {new_max}")
        elif key == ord('c'):
            self.tracker.reset()
            print("Reset - cleared all tracking state")
        
        return False
    
    def wait_key(self, delay: int = 1) -> int:
        """Wait for keyboard input."""
        return cv.waitKey(delay) & 0xFF
    
    def close_all(self):
        """Close all OpenCV windows."""
        cv.destroyAllWindows()
