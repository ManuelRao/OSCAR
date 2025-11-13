"""
Multi-Object Tracking System

A comprehensive tracking library that provides:
- Blob detection from difference maps
- Track persistence and filtering
- Multi-track fusion into objects
- Predictive importance mapping
- Per-object spatial learning

Usage:
    config = TrackingConfig()
    config.set_max_objects(2)
    config.set_blob_min_area(25)
    
    tracker = MultiObjectTracker(config)
    tracker.update(frame)
    
    objects = tracker.get_objects()
    tracks = tracker.get_tracks()
    blobs = tracker.get_blobs()
"""

import cv2 as cv
import numpy as np
from collections import deque
import time
from typing import List, Tuple, Optional, Dict


class BlobInfo:
    """Represents a detected blob with position and properties."""
    
    def __init__(self, keypoint, frame_id):
        self.pos = np.array(keypoint.pt)
        self.size = keypoint.size
        self.frame_id = frame_id
        self.velocity = np.array([0.0, 0.0])
    
    def __repr__(self):
        return f"Blob(pos={self.pos}, size={self.size:.1f})"


class Track:
    """Represents a persistent track of moving objects."""
    
    next_id = 0
    
    def __init__(self, initial_blob, frame_id):
        self.id = Track.next_id
        Track.next_id += 1
        
        self.position = initial_blob.pos.copy()
        self.velocity = np.array([0.0, 0.0])
        self.position_history = deque(maxlen=10)
        self.position_history.append((initial_blob.pos.copy(), frame_id))
        
        self.blobs = [initial_blob]
        self.blob_history = deque(maxlen=5)
        
        self.last_seen_frame = frame_id
        self.creation_frame = frame_id
        self.frames_since_update = 0
        self.total_updates = 1
        self.certainty = 0.0
        
        self.alpha = 0.4  # Reverted to original for smooth tracking
        
        # Activity tracking
        self.frames_with_no_activity = 0
        self.activity_check_radius = 20
        
        self.color = tuple(np.random.randint(50, 255, 3).tolist())
    
    def get_certainty(self):
        """Calculate track certainty (0-1) based on age, consistency, and recent activity."""
        age_factor = min(self.total_updates / 20.0, 1.0)
        recency_factor = 1.0 / (1.0 + self.frames_since_update * 0.1)
        
        consistency_factor = 1.0
        if len(self.blob_history) > 3:
            avg_blobs = np.mean(list(self.blob_history))
            variance = np.var(list(self.blob_history))
            if avg_blobs > 0:
                consistency_factor = max(0.5, 1.0 - (variance / (avg_blobs + 1)))
        
        self.certainty = (age_factor * 0.5 + recency_factor * 0.3 + consistency_factor * 0.2)
        return self.certainty
    
    def predict_position(self, dt=1.0):
        """Predict next position based on velocity."""
        return self.position + self.velocity * dt
    
    def update(self, matched_blobs, frame_id):
        """Update track with matched blobs using smoothing."""
        if not matched_blobs:
            self.frames_since_update += 1
            return
        
        self.frames_with_no_activity = 0
        
        blob_positions = np.array([blob.pos for blob in matched_blobs])
        raw_new_position = np.mean(blob_positions, axis=0)
        
        smoothed_position = self.alpha * raw_new_position + (1 - self.alpha) * self.position
        
        if len(self.position_history) >= 2:
            prev_pos, prev_frame = self.position_history[-1]
            dt = max(frame_id - prev_frame, 1)
            raw_velocity = (smoothed_position - prev_pos) / dt
            self.velocity = self.alpha * raw_velocity + (1 - self.alpha) * self.velocity
        
        self.position = smoothed_position
        self.position_history.append((smoothed_position.copy(), frame_id))
        
        self.blobs = matched_blobs
        self.blob_history.append(len(matched_blobs))
        
        self.last_seen_frame = frame_id
        self.frames_since_update = 0
        self.total_updates += 1
    
    def check_area_activity(self, diff_map):
        """Check if there's any activity in the track's area."""
        x, y = int(self.position[0]), int(self.position[1])
        radius = self.activity_check_radius
        
        height, width = diff_map.shape
        x_min = max(0, x - radius)
        x_max = min(width, x + radius)
        y_min = max(0, y - radius)
        y_max = min(height, y + radius)
        
        region = diff_map[y_min:y_max, x_min:x_max]
        return np.any(region > 0)
    
    def update_activity_status(self, diff_map):
        """Update the no-activity frame counter."""
        if self.check_area_activity(diff_map):
            self.frames_with_no_activity = 0
        else:
            self.frames_with_no_activity += 1
    
    def is_alive(self, current_frame, max_frames_lost=30):
        """Check if track should be kept alive."""
        frames_lost = current_frame - self.last_seen_frame
        
        if self.frames_with_no_activity >= 5:
            return False
        
        if self.total_updates < 3:
            return frames_lost < 10
        
        return frames_lost < max_frames_lost
    
    def get_speed(self):
        """Get current speed magnitude."""
        return np.linalg.norm(self.velocity)
    
    def __repr__(self):
        return f"Track(id={self.id}, pos={self.position}, vel={self.velocity}, age={self.total_updates})"


class Object:
    """Represents a high-level object with multi-track fusion."""
    
    next_id = 0
    
    def __init__(self, initial_track, frame_id):
        self.id = Object.next_id
        Object.next_id += 1
        
        self.position = initial_track.position.copy()
        self.velocity = initial_track.velocity.copy()
        
        self.alpha = 0.3  # Reverted to original for smooth tracking
        self.raw_position = initial_track.position.copy()
        self.raw_velocity = initial_track.velocity.copy()
        
        self.track_pool = [initial_track]
        self.track_history = deque(maxlen=10)
        self.track_history.append(initial_track.id)
        
        self.last_seen_frame = frame_id
        self.creation_frame = frame_id
        self.frames_since_update = 0
        self.total_updates = 1
        
        self.color = tuple(np.random.randint(100, 255, 3).tolist())
        self.importance_map = None
    
    def predict_position(self, dt=1.0):
        """Predict next position based on velocity."""
        return self.position + self.velocity * dt
    
    def _remove_outlier_tracks(self):
        """Remove tracks that are too far from the pool average (likely stale tracks)."""
        if len(self.track_pool) < 3:
            return
        
        # Calculate pool statistics
        positions = np.array([t.position for t in self.track_pool])
        velocities = np.array([t.velocity for t in self.track_pool])
        
        mean_position = np.mean(positions, axis=0)
        mean_velocity = np.mean(velocities, axis=0)
        
        # Calculate standard deviations
        position_distances = np.linalg.norm(positions - mean_position, axis=1)
        velocity_distances = np.linalg.norm(velocities - mean_velocity, axis=1)
        
        pos_std = np.std(position_distances)
        vel_std = np.std(velocity_distances)
        
        # Remove tracks that are more than 2.5 standard deviations away
        # (in both position AND velocity - must be outlier in both)
        filtered_pool = []
        for i, track in enumerate(self.track_pool):
            pos_outlier = position_distances[i] > (pos_std * 2.5 + 30)  # +30 for tolerance
            vel_outlier = velocity_distances[i] > (vel_std * 2.5 + 5)   # +5 for tolerance
            
            # Only remove if it's an outlier in position (velocity alone not enough)
            if not pos_outlier:
                filtered_pool.append(track)
            elif len(self.track_pool) > 5 and pos_outlier and vel_outlier:
                # If we have many tracks and this is outlier in both, remove it
                continue
            else:
                filtered_pool.append(track)
        
        # Only apply filter if we're not removing too many tracks
        if len(filtered_pool) >= max(2, len(self.track_pool) // 2):
            self.track_pool = filtered_pool
    
    def calculate_track_correlation(self, track):
        """Calculate correlation between track and object (0-1)."""
        distance = np.linalg.norm(track.position - self.position)
        max_distance = 80
        distance_corr = max(0, 1 - (distance / max_distance))
        
        vel_corr = 0.5
        if np.linalg.norm(track.velocity) > 0.5 and np.linalg.norm(self.velocity) > 0.5:
            track_vel_norm = track.velocity / np.linalg.norm(track.velocity)
            obj_vel_norm = self.velocity / np.linalg.norm(self.velocity)
            vel_dot = np.dot(track_vel_norm, obj_vel_norm)
            vel_corr = (vel_dot + 1) / 2
        
        certainty = track.get_certainty()
        correlation = (distance_corr * 0.4 + vel_corr * 0.3 + certainty * 0.3)
        return correlation
    
    def update(self, candidate_tracks, frame_id, correlation_threshold=0.3):
        """Update object using multi-track fusion."""
        if not candidate_tracks:
            # Don't clear track pool immediately - filter out dead/old tracks instead
            # Keep tracks that are still alive and recent (reduced from 15 to 10 frames)
            self.track_pool = [t for t in self.track_pool 
                             if t.is_alive(frame_id) and 
                             (frame_id - t.last_seen_frame) < 10]
            self.frames_since_update += 1
            return
        
        track_correlations = []
        for track in candidate_tracks:
            correlation = self.calculate_track_correlation(track)
            if correlation >= correlation_threshold:
                track_correlations.append((track, correlation))
        
        # Also include existing pool tracks that are still alive
        existing_track_ids = {t.id for t, _ in track_correlations}
        for track in self.track_pool:
            if track.id not in existing_track_ids and track.is_alive(frame_id):
                correlation = self.calculate_track_correlation(track)
                if correlation >= correlation_threshold * 0.85:  # Reduced bonus (was 0.8)
                    track_correlations.append((track, correlation))
        
        track_correlations.sort(key=lambda x: x[1], reverse=True)
        
        self.track_pool = [track for track, _ in track_correlations[:10]]
        
        # Remove outlier tracks (those too far from pool average)
        if len(self.track_pool) > 3:
            self._remove_outlier_tracks()
        
        for track, _ in track_correlations:
            if track.id not in self.track_history:
                self.track_history.append(track.id)
        
        weights = []
        for track, correlation in track_correlations:
            certainty = track.get_certainty()
            weight = correlation * certainty
            weights.append(weight)
        
        total_weight = sum(weights)
        
        if total_weight > 0:
            weighted_positions = [track.position * weight 
                                 for (track, _), weight in zip(track_correlations, weights)]
            self.raw_position = sum(weighted_positions) / total_weight
            
            weighted_velocities = [track.velocity * weight 
                                  for (track, _), weight in zip(track_correlations, weights)]
            self.raw_velocity = sum(weighted_velocities) / total_weight
        else:
            self.raw_position = np.mean([t.position for t, _ in track_correlations], axis=0)
            self.raw_velocity = np.mean([t.velocity for t, _ in track_correlations], axis=0)
        
        # Check for NaN and skip update if invalid
        if np.any(np.isnan(self.raw_position)) or np.any(np.isnan(self.raw_velocity)):
            return
        
        self.position = self.alpha * self.raw_position + (1 - self.alpha) * self.position
        self.velocity = self.alpha * self.raw_velocity + (1 - self.alpha) * self.velocity
        
        self.last_seen_frame = frame_id
        self.frames_since_update = 0
        self.total_updates += 1
    
    def is_alive(self, current_frame, max_frames_lost=20):
        """Check if object should still exist."""
        frames_lost = current_frame - self.last_seen_frame
        return frames_lost < max_frames_lost
    
    def get_speed(self):
        """Get current speed magnitude."""
        return np.linalg.norm(self.velocity)
    
    def update_importance_map(self, all_tracks, valuable_score=5.0, unimportant_score=-3.0, 
                              use_prediction=True, prediction_steps=3):
        """Update importance map with prediction."""
        if self.importance_map is None:
            return
        
        pool_track_ids = {t.id for t in self.track_pool}
        
        important_tracks = [t for t in all_tracks if t.id in pool_track_ids]
        unimportant_tracks = [t for t in all_tracks if t.id not in pool_track_ids]
        
        self.importance_map.update(important_tracks, unimportant_tracks, 
                                   valuable_score, unimportant_score)
        
        if use_prediction:
            # Object-level predictions (for fast-moving objects)
            object_speed = self.get_speed()
            
            if object_speed > 5:
                for step in range(1, prediction_steps + 1):
                    # Project much further forward: multiply velocity by (step * 3)
                    predicted_obj_pos = self.position + self.velocity * (step * 3)
                    # Stronger prediction: 4.0x base strength, slower decay
                    prediction_strength = valuable_score * 4.0 * (1.0 - step * 0.08)
                    # Larger radius: starts at 60, decays slower
                    prediction_radius = max(30, 60 - step * 3)
                    
                    self.importance_map.add_prediction(
                        predicted_obj_pos,
                        prediction_strength,
                        radius=prediction_radius
                    )
            
            # Track-level predictions (independent of object speed)
            for track in important_tracks:
                track_speed = track.get_speed()
                
                if track_speed > 3:
                    for step in range(1, prediction_steps + 1):
                        # Project further forward: multiply velocity by (step * 2)
                        predicted_track_pos = track.position + track.velocity * (step * 2)
                        # Stronger prediction: 3.0x base strength, slower decay
                        track_prediction_strength = valuable_score * 3.0 * (1.0 - step * 0.1)
                        # Larger radius: starts at 40, minimum 20
                        track_prediction_radius = max(20, 40 - step * 2)
                        
                        self.importance_map.add_prediction(
                            predicted_track_pos,
                            track_prediction_strength,
                            radius=track_prediction_radius
                        )
    
    def __repr__(self):
        return f"Object(id={self.id}, pos={self.position}, tracks={len(self.track_pool)})"


class ImportanceMap:
    """Spatial importance map with three layers for learning track quality."""
    
    def __init__(self, width, height, positive_decay=0.85, negative_decay=0.98, prediction_decay=0.75):
        self.positive_map = np.zeros((height, width), dtype=np.float32)
        self.negative_map = np.zeros((height, width), dtype=np.float32)
        self.prediction_map = np.zeros((height, width), dtype=np.float32)
        self.positive_decay = positive_decay
        self.negative_decay = negative_decay
        self.prediction_decay = prediction_decay
        self.width = width
        self.height = height
    
    def add_gaussian_influence(self, center, value, radius=30):
        """Add Gaussian-weighted influence to positive or negative map."""
        # Validate input
        if np.any(np.isnan(center)) or np.isnan(value):
            return
        
        x, y = int(center[0]), int(center[1])
        
        # Skip if completely out of bounds
        if x < -radius or x >= self.width + radius or y < -radius or y >= self.height + radius:
            return
        
        y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
        sigma = radius / 3.0
        gaussian = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        
        x_min = max(0, x - radius)
        x_max = min(self.width, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(self.height, y + radius + 1)
        
        # Ensure valid region
        if x_min >= x_max or y_min >= y_max:
            return
        
        g_x_min = max(0, radius - x)
        g_x_max = g_x_min + (x_max - x_min)
        g_y_min = max(0, radius - y)
        g_y_max = g_y_min + (y_max - y_min)
        
        # Ensure valid gaussian region
        if g_x_min >= g_x_max or g_y_min >= g_y_max:
            return
        
        try:
            influence = gaussian[g_y_min:g_y_max, g_x_min:g_x_max] * value
            
            if value > 0:
                self.positive_map[y_min:y_max, x_min:x_max] += influence
            else:
                region = self.negative_map[y_min:y_max, x_min:x_max]
                self.negative_map[y_min:y_max, x_min:x_max] = np.maximum(region, -influence)
        except (ValueError, IndexError):
            # Silently skip if shapes don't match (edge case)
            pass
    
    def add_prediction(self, center, value, radius=20):
        """Add prediction influence to separate prediction layer."""
        # Validate input
        if np.any(np.isnan(center)) or np.isnan(value):
            return
        
        x, y = int(center[0]), int(center[1])
        
        # Skip if completely out of bounds
        if x < -radius or x >= self.width + radius or y < -radius or y >= self.height + radius:
            return
        
        y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
        sigma = radius / 3.0
        gaussian = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        
        x_min = max(0, x - radius)
        x_max = min(self.width, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(self.height, y + radius + 1)
        
        # Ensure valid region
        if x_min >= x_max or y_min >= y_max:
            return
        
        g_x_min = max(0, radius - x)
        g_x_max = g_x_min + (x_max - x_min)
        g_y_min = max(0, radius - y)
        g_y_max = g_y_min + (y_max - y_min)
        
        # Ensure valid gaussian region
        if g_x_min >= g_x_max or g_y_min >= g_y_max:
            return
        
        try:
            influence = gaussian[g_y_min:g_y_max, g_x_min:g_x_max] * value
            self.prediction_map[y_min:y_max, x_min:x_max] += influence
        except (ValueError, IndexError):
            # Silently skip if shapes don't match (edge case)
            pass
    
    def update(self, valuable_tracks, noise_tracks, valuable_score=5.0, noise_score=-3.0):
        """Update importance map with current tracks."""
        self.positive_map *= self.positive_decay
        self.negative_map *= self.negative_decay
        self.prediction_map *= self.prediction_decay
        
        for track in valuable_tracks:
            self.add_gaussian_influence(track.position, valuable_score, radius=30)
        
        for track in noise_tracks:
            self.add_gaussian_influence(track.position, noise_score, radius=25)
    
    def get_score(self, position):
        """Get combined importance score at a position."""
        x, y = int(position[0]), int(position[1])
        if 0 <= x < self.width and 0 <= y < self.height:
            positive_val = self.positive_map[y, x]
            negative_val = self.negative_map[y, x]
            prediction_val = self.prediction_map[y, x]
            return positive_val + prediction_val - negative_val
        return 0.0
    
    def get_visualization(self):
        """Get visualization of combined importance map."""
        combined = self.positive_map + self.prediction_map - self.negative_map
        combined = np.clip(combined, -20, 20)
        
        map_min = combined.min()
        map_max = combined.max()
        if map_max > map_min:
            map_vis = ((combined - map_min) / (map_max - map_min) * 255).astype(np.uint8)
        else:
            map_vis = np.zeros_like(combined, dtype=np.uint8)
        
        return cv.applyColorMap(map_vis, cv.COLORMAP_JET)
    
    def reset(self):
        """Reset all maps to zeros."""
        self.positive_map.fill(0)
        self.negative_map.fill(0)
        self.prediction_map.fill(0)


class TrackingConfig:
    """Configuration for the multi-object tracker."""
    
    def __init__(self):
        # Blob detection parameters
        self.blob_min_area = 25
        self.blob_max_area = 2000
        self.blob_min_circularity = 0.1
        
        # Diff map processing
        self.diff_threshold = 20
        self.gaussian_blur_size = (5, 5)
        self.gaussian_blur_sigma = 2
        self.temporal_smoothing_alpha = 0.3  # Back to original
        
        # Track parameters
        self.track_max_frames_lost_new = 10
        self.track_max_frames_lost_established = 30
        self.track_activity_timeout = 5
        self.track_min_persistence = 3
        self.track_max_match_distance = 50
        
        # Object parameters
        self.max_objects = 2
        self.object_max_frames_lost = 20
        self.object_min_track_score = 18  # Balanced: not too strict, not too loose
        self.object_correlation_threshold = 0.30  # Balanced threshold
        self.object_init_min_distance = 100
        
        # Importance map parameters
        self.use_importance_map = True
        self.importance_positive_decay = 0.85
        self.importance_negative_decay = 0.98
        self.importance_prediction_decay = 0.85  # Slower decay (was 0.75) for longer-lasting predictions
        self.use_prediction = True
        self.prediction_steps = 3  # Reduced from 5 for better performance
        self.prediction_min_speed_object = 5  # Min speed for object-level predictions
        self.prediction_min_speed_track = 3   # Min speed for track-level predictions
        
    # Setter methods for easy configuration
    def set_max_objects(self, count: int):
        """Set maximum number of objects to track."""
        self.max_objects = max(1, min(9, count))
        return self
    
    def set_blob_min_area(self, area: int):
        """Set minimum blob area in pixels."""
        self.blob_min_area = max(1, area)
        return self
    
    def set_blob_max_area(self, area: int):
        """Set maximum blob area in pixels."""
        self.blob_max_area = area
        return self
    
    def set_diff_threshold(self, threshold: int):
        """Set difference map threshold (0-255)."""
        self.diff_threshold = max(0, min(255, threshold))
        return self
    
    def set_temporal_smoothing(self, alpha: float):
        """Set temporal smoothing factor (0-1)."""
        self.temporal_smoothing_alpha = max(0.0, min(1.0, alpha))
        return self
    
    def set_prediction_enabled(self, enabled: bool):
        """Enable/disable predictive importance mapping."""
        self.use_prediction = enabled
        return self
    
    def set_prediction_steps(self, steps: int):
        """Set number of prediction steps (1-10)."""
        self.prediction_steps = max(1, min(10, steps))
        return self
    
    def set_importance_map_enabled(self, enabled: bool):
        """Enable/disable importance mapping."""
        self.use_importance_map = enabled
        return self


class MultiObjectTracker:
    """
    Main tracking class that manages blob detection, tracks, and objects.
    
    Usage:
        config = TrackingConfig()
        tracker = MultiObjectTracker(config)
        
        # Update with new frame
        tracker.update(frame)
        
        # Get tracking data
        objects = tracker.get_objects()
        tracks = tracker.get_tracks()
        blobs = tracker.get_blobs()
        
        # Modify objects
        tracker.create_object(position)
        tracker.delete_object(object_id)
        tracker.update_object_position(object_id, new_position)
    """
    
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        # State variables
        self.prev_gray = None
        self.prev_blobs = []
        self.accumulated_diff = None
        
        # Tracking data
        self.tracks: List[Track] = []
        self.objects: List[Object] = []
        self.current_blobs: List[BlobInfo] = []
        self.current_keypoints = []
        self.diff_map_final = None
        
        # History buffers
        self.unmatched_history = deque(maxlen=5)
        self.unmatched_tracks_history = deque(maxlen=3)
        
        # Blob detector
        self.blob_detector = self._setup_blob_detector()
        
        # Frame dimensions (set on first frame)
        self.frame_width = None
        self.frame_height = None
        
    def _setup_blob_detector(self):
        """Configure blob detector with current config."""
        params = cv.SimpleBlobDetector_Params()
        
        params.filterByArea = True
        params.minArea = self.config.blob_min_area
        params.maxArea = self.config.blob_max_area
        
        params.filterByCircularity = True
        params.minCircularity = self.config.blob_min_circularity
        params.maxCircularity = 1.0
        
        params.filterByConvexity = False
        params.filterByInertia = False
        params.filterByColor = False
        
        return cv.SimpleBlobDetector_create(params)
    
    def update(self, frame: np.ndarray, diff_func=None) -> Dict:
        """
        Update tracker with new frame.
        
        Args:
            frame: BGR or grayscale image
            diff_func: Optional custom difference function (default uses picture_difference from math_func)
        
        Returns:
            dict with tracking statistics
        """
        self.frame_count += 1
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Initialize frame dimensions
        if self.frame_height is None:
            self.frame_height, self.frame_width = gray.shape
            self._initialize_importance_maps()
        
        stats = {
            'frame_count': self.frame_count,
            'num_blobs': 0,
            'num_tracks': 0,
            'num_objects': 0,
            'mean_diff': 0.0,
            'max_diff': 0.0
        }
        
        if self.prev_gray is not None:
            # Import here to avoid circular dependency
            from src import math_func as mf
            diff_func = diff_func or mf.picture_diference
            
            # Calculate and process difference map
            diff_map = diff_func(gray, self.prev_gray)
            diff_map = cv.normalize(diff_map, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            
            # Faster temporal smoothing: use cv.addWeighted instead of manual array ops
            if self.accumulated_diff is None:
                self.accumulated_diff = diff_map.astype(np.float32)
            else:
                alpha = self.config.temporal_smoothing_alpha
                # Convert to same type for addWeighted
                diff_float = diff_map.astype(np.float32)
                cv.addWeighted(diff_float, alpha, self.accumulated_diff, 1 - alpha, 0, 
                              dst=self.accumulated_diff)
            
            # Combined blur and threshold (faster than separate operations)
            diff_map_smoothed = cv.GaussianBlur(self.accumulated_diff.astype(np.uint8), 
                                                self.config.gaussian_blur_size, 
                                                self.config.gaussian_blur_sigma)
            
            self.diff_map_final = cv.threshold(diff_map_smoothed, 
                                               self.config.diff_threshold, 
                                               255, cv.THRESH_BINARY)[1]
            
            # Detect blobs
            self.current_keypoints = self.blob_detector.detect(self.diff_map_final)
            self._process_blobs()
            
            # Update tracks
            self._update_tracks()
            
            # Update objects
            self._update_objects()
            
            # Update statistics
            stats['num_blobs'] = len(self.current_blobs)
            stats['num_tracks'] = len(self.tracks)
            stats['num_objects'] = len(self.objects)
            stats['mean_diff'] = float(np.mean(self.diff_map_final))
            stats['max_diff'] = float(np.max(self.diff_map_final))
        
        self.prev_gray = gray.copy()
        self.last_frame_time = time.time()
        
        return stats
    
    def _initialize_importance_maps(self):
        """Initialize importance maps for objects."""
        if not self.config.use_importance_map:
            return
        
        for obj in self.objects:
            if obj.importance_map is None:
                obj.importance_map = ImportanceMap(
                    self.frame_width, 
                    self.frame_height,
                    positive_decay=self.config.importance_positive_decay,
                    negative_decay=self.config.importance_negative_decay,
                    prediction_decay=self.config.importance_prediction_decay
                )
    
    def _process_blobs(self):
        """Convert keypoints to BlobInfo objects with velocity."""
        current_time = time.time()
        dt = current_time - self.last_frame_time
        
        self.current_blobs = []
        for kp in self.current_keypoints:
            blob = BlobInfo(kp, self.frame_count)
            
            if self.prev_blobs:
                closest_dist = float('inf')
                closest_blob = None
                for prev_blob in self.prev_blobs:
                    dist = np.linalg.norm(blob.pos - prev_blob.pos)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_blob = prev_blob
                
                if closest_blob and closest_dist < 50 and dt > 0:
                    blob.velocity = (blob.pos - closest_blob.pos) / dt
            
            self.current_blobs.append(blob)
        
        self.prev_blobs = self.current_blobs
    
    def _update_tracks(self):
        """Update existing tracks and create new ones."""
        # Match blobs to tracks
        track_assignments, unmatched_blobs = self._match_blobs_to_tracks()
        
        # Update tracks
        for track in self.tracks:
            if track.id in track_assignments:
                track.update(track_assignments[track.id], self.frame_count)
        
        # Update activity status
        for track in self.tracks:
            track.update_activity_status(self.diff_map_final)
        
        # Add unmatched blobs to history
        self.unmatched_history.append(unmatched_blobs)
        
        # Find persistent blobs and create tracks
        persistent_blobs = self._find_persistent_blobs()
        
        for persistent_blob in persistent_blobs:
            too_close = any(
                np.linalg.norm(persistent_blob.pos - track.position) < 30
                for track in self.tracks
            )
            
            if not too_close:
                new_track = Track(persistent_blob, self.frame_count)
                self.tracks.append(new_track)
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.is_alive(self.frame_count)]
    
    def _update_objects(self):
        """Update objects with track fusion and importance maps."""
        alive_tracks = self.tracks
        
        # Initialize objects if needed
        if len(self.objects) < self.config.max_objects and len(alive_tracks) > 0:
            for track in alive_tracks:
                if len(self.objects) >= self.config.max_objects:
                    break
                
                too_close = any(
                    np.linalg.norm(track.position - obj.position) < self.config.object_init_min_distance
                    for obj in self.objects
                )
                
                if not too_close:
                    new_object = Object(track, self.frame_count)
                    if self.config.use_importance_map and self.frame_width:
                        new_object.importance_map = ImportanceMap(
                            self.frame_width, 
                            self.frame_height,
                            positive_decay=self.config.importance_positive_decay,
                            negative_decay=self.config.importance_negative_decay,
                            prediction_decay=self.config.importance_prediction_decay
                        )
                    self.objects.append(new_object)
        
        # Match tracks to objects
        if alive_tracks and self.objects:
            object_assignments, _ = self._match_tracks_to_objects()
            
            valuable_track_ids = set()
            
            for obj in self.objects:
                if obj.id in object_assignments and object_assignments[obj.id]:
                    obj.update(object_assignments[obj.id], 
                              self.frame_count, 
                              correlation_threshold=self.config.object_correlation_threshold)
                else:
                    # If object has no tracks and is desperate, try slightly relaxed matching
                    if len(obj.track_pool) == 0 and alive_tracks:
                        desperate_threshold = self.config.object_correlation_threshold * 0.7  # Less desperate (was 0.5)
                        obj.update(alive_tracks, self.frame_count, 
                                 correlation_threshold=desperate_threshold)
                    else:
                        obj.update([], self.frame_count)
                
                valuable_track_ids.update([t.id for t in obj.track_pool])
                
                # Update importance map (skip if object has no tracks - waste of CPU)
                if self.config.use_importance_map and len(obj.track_pool) > 0:
                    obj.update_importance_map(
                        alive_tracks,
                        use_prediction=self.config.use_prediction,
                        prediction_steps=self.config.prediction_steps
                    )
    
    def _match_blobs_to_tracks(self):
        """Match current blobs to existing tracks."""
        if not self.current_blobs or not self.tracks:
            return {}, self.current_blobs
        
        # Score matrix
        scores = np.zeros((len(self.current_blobs), len(self.tracks)))
        
        for i, blob in enumerate(self.current_blobs):
            for j, track in enumerate(self.tracks):
                scores[i, j] = self._calculate_blob_track_score(blob, track)
        
        # Greedy assignment
        assignments = {}
        unmatched_blobs = list(self.current_blobs)
        
        while np.max(scores) > 15:
            blob_idx, track_idx = np.unravel_index(np.argmax(scores), scores.shape)
            
            track_id = self.tracks[track_idx].id
            blob = self.current_blobs[blob_idx]
            
            if track_id not in assignments:
                assignments[track_id] = []
            assignments[track_id].append(blob)
            
            if blob in unmatched_blobs:
                unmatched_blobs.remove(blob)
            
            scores[blob_idx, :] = -np.inf
            scores[:, track_idx] = -np.inf
        
        return assignments, unmatched_blobs
    
    def _match_tracks_to_objects(self):
        """Match tracks to objects."""
        if not self.tracks or not self.objects:
            return {}, self.tracks
        
        # Pre-calculate certainties once (expensive operation)
        track_certainties = {track.id: track.get_certainty() for track in self.tracks}
        
        scores = np.zeros((len(self.tracks), len(self.objects)))
        
        for i, track in enumerate(self.tracks):
            for j, obj in enumerate(self.objects):
                scores[i, j] = self._calculate_track_object_score(track, obj)
        
        assignments = {}
        max_iterations = len(self.tracks) * len(self.objects)  # Safety limit
        iteration = 0
        
        while np.max(scores) > self.config.object_min_track_score and iteration < max_iterations:
            track_idx, obj_idx = np.unravel_index(np.argmax(scores), scores.shape)
            
            obj_id = self.objects[obj_idx].id
            track = self.tracks[track_idx]
            
            if obj_id not in assignments:
                assignments[obj_id] = []
            assignments[obj_id].append(track)
            
            scores[track_idx, :] = -np.inf
            iteration += 1
        
        return assignments, []
    
    def _calculate_blob_track_score(self, blob, track):
        """Calculate matching score between blob and track."""
        distance = np.linalg.norm(blob.pos - track.position)
        max_distance = 50
        
        if distance > max_distance:
            return 0.0
        
        distance_score = (max_distance - distance) / max_distance * 50
        
        if np.linalg.norm(blob.velocity) > 0.1 and np.linalg.norm(track.velocity) > 0.1:
            blob_vel_norm = blob.velocity / (np.linalg.norm(blob.velocity) + 1e-6)
            track_vel_norm = track.velocity / (np.linalg.norm(track.velocity) + 1e-6)
            vel_correlation = np.dot(blob_vel_norm, track_vel_norm)
            vel_score = (vel_correlation + 1) / 2 * 30
            distance_score += vel_score
        
        frames_since = self.frame_count - track.last_seen_frame
        if frames_since == 0:
            distance_score += 20
        elif frames_since == 1:
            distance_score += 15
        elif frames_since == 2:
            distance_score += 10
        
        return distance_score
    
    def _calculate_track_object_score(self, track, obj):
        """Calculate matching score between track and object."""
        certainty = track.get_certainty()
        if certainty < 0.15:  # Lowered from 0.2 to accept lower certainty tracks
            return 0.0
        
        predicted_pos = obj.predict_position()
        distance = np.linalg.norm(track.position - predicted_pos)
        max_distance = 150  # Increased from 100 to allow matching further tracks
        
        if distance > max_distance:
            return 0.0
        
        distance_score = (max_distance - distance) / max_distance * 40
        
        if np.linalg.norm(track.velocity) > 0.1 and np.linalg.norm(obj.velocity) > 0.1:
            track_vel_norm = track.velocity / (np.linalg.norm(track.velocity) + 1e-6)
            obj_vel_norm = obj.velocity / (np.linalg.norm(obj.velocity) + 1e-6)
            vel_correlation = np.dot(track_vel_norm, obj_vel_norm)
            vel_score = (vel_correlation + 1) / 2 * 20
            distance_score += vel_score
        
        frames_since = self.frame_count - obj.last_seen_frame
        if frames_since == 0:
            recency_score = 10
        elif frames_since == 1:
            recency_score = 8
        elif frames_since == 2:
            recency_score = 5
        else:
            recency_score = 2
        
        distance_score += recency_score
        distance_score *= (1.0 + certainty)
        
        return distance_score
    
    def _find_persistent_blobs(self):
        """Find blobs that persist across frames."""
        if len(self.unmatched_history) < self.config.track_min_persistence:
            return []
        
        current_unmatched = self.unmatched_history[-1]
        persistent = []
        
        for blob in current_unmatched:
            match_count = 1
            
            for i in range(len(self.unmatched_history) - 2, -1, -1):
                found_match = False
                for hist_blob in self.unmatched_history[i]:
                    distance = np.linalg.norm(blob.pos - hist_blob.pos)
                    if distance < self.config.track_max_match_distance:
                        found_match = True
                        break
                
                if found_match:
                    match_count += 1
                else:
                    break
            
            if match_count >= self.config.track_min_persistence:
                persistent.append(blob)
        
        return persistent
    
    # Public API methods
    
    def get_objects(self) -> List[Object]:
        """Get all tracked objects."""
        return self.objects.copy()
    
    def get_tracks(self) -> List[Track]:
        """Get all active tracks."""
        return self.tracks.copy()
    
    def get_blobs(self) -> List[BlobInfo]:
        """Get blobs from current frame."""
        return self.current_blobs.copy()
    
    def get_object_by_id(self, object_id: int) -> Optional[Object]:
        """Get object by ID."""
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        return None
    
    def create_object(self, position: np.ndarray, velocity: Optional[np.ndarray] = None) -> Optional[Object]:
        """
        Manually create a new object at specified position.
        
        Args:
            position: (x, y) position
            velocity: Optional (vx, vy) velocity
        
        Returns:
            Created object or None if max objects reached
        """
        if len(self.objects) >= self.config.max_objects:
            return None
        
        # Create a dummy track to initialize object
        class DummyBlob:
            def __init__(self, pos):
                self.pos = pos
                self.size = 10
                self.frame_id = self.frame_count
                self.velocity = np.array([0.0, 0.0])
        
        dummy_blob = DummyBlob(np.array(position))
        dummy_track = Track(dummy_blob, self.frame_count)
        
        if velocity is not None:
            dummy_track.velocity = np.array(velocity)
        
        new_object = Object(dummy_track, self.frame_count)
        
        if self.config.use_importance_map and self.frame_width:
            new_object.importance_map = ImportanceMap(
                self.frame_width,
                self.frame_height,
                positive_decay=self.config.importance_positive_decay,
                negative_decay=self.config.importance_negative_decay,
                prediction_decay=self.config.importance_prediction_decay
            )
        
        self.objects.append(new_object)
        return new_object
    
    def delete_object(self, object_id: int) -> bool:
        """
        Delete object by ID.
        
        Returns:
            True if deleted, False if not found
        """
        for i, obj in enumerate(self.objects):
            if obj.id == object_id:
                del self.objects[i]
                return True
        return False
    
    def update_object_position(self, object_id: int, position: np.ndarray, 
                               velocity: Optional[np.ndarray] = None) -> bool:
        """
        Manually update object position and optionally velocity.
        
        Returns:
            True if updated, False if not found
        """
        obj = self.get_object_by_id(object_id)
        if obj is None:
            return False
        
        obj.position = np.array(position)
        if velocity is not None:
            obj.velocity = np.array(velocity)
        
        return True
    
    def reset(self):
        """Reset all tracking state."""
        self.frame_count = 0
        self.prev_gray = None
        self.prev_blobs = []
        self.accumulated_diff = None
        self.tracks.clear()
        self.objects.clear()
        self.current_blobs.clear()
        self.unmatched_history.clear()
        self.unmatched_tracks_history.clear()
        Track.next_id = 0
        Object.next_id = 0
    
    def get_diff_map(self) -> Optional[np.ndarray]:
        """Get the final processed difference map."""
        return self.diff_map_final
    
    def get_frame_count(self) -> int:
        """Get current frame count."""
        return self.frame_count
    
    def get_config(self) -> TrackingConfig:
        """Get current configuration."""
        return self.config
