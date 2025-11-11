"""
Demo: Difference Map with Blob Detection and Multi-Object Tracking

This script creates a difference map between consecutive frames, detects blobs,
and tracks them over time by grouping related blobs into persistent tracks.

Tracking algorithm:
- Blobs are matched to tracks based on position, velocity correlation, and persistence
- Tracks are formed when correlated blobs persist over multiple frames
- Tracks are terminated when no matching blobs are found for several frames

Controls:
  'q' - Quit
  's' - Save current diff_map
  'd' - Toggle debug views
  'c' - Clear/reset
  'b' - Toggle blob detection overlay
  't' - Toggle track visualization
"""

import cv2 as cv
import numpy as np
from src import math_func as mf
from collections import deque
import time


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
        
        self.blobs = [initial_blob]  # Sub-tracks (current frame blobs)
        self.blob_history = deque(maxlen=5)  # Recent blob associations
        
        self.last_seen_frame = frame_id
        self.creation_frame = frame_id
        self.frames_since_update = 0
        self.total_updates = 1
        self.certainty = 0.0  # 0-1 scale, increases with stability
        
        # Smoothing factor for gradual position updates
        self.alpha = 0.4  # 0=no change, 1=instant change (lower = smoother)
        
        # Activity tracking - monitor diff_map activity in track's area
        self.frames_with_no_activity = 0  # Count frames with zero diff_map activity
        self.activity_check_radius = 20  # Radius to check for activity
        
        self.color = tuple(np.random.randint(50, 255, 3).tolist())
    
    def get_certainty(self):
        """
        Calculate track certainty based on age, consistency, and recent activity.
        Returns 0-1 where 1 is most certain.
        """
        # Factor 1: Age (more updates = more certain, caps at 20 updates)
        age_factor = min(self.total_updates / 20.0, 1.0)
        
        # Factor 2: Recent activity (penalize if not seen recently)
        recency_factor = 1.0 / (1.0 + self.frames_since_update * 0.1)
        
        # Factor 3: Consistency (stable blob count in history)
        consistency_factor = 1.0
        if len(self.blob_history) > 3:
            avg_blobs = np.mean(list(self.blob_history))
            variance = np.var(list(self.blob_history))
            if avg_blobs > 0:
                consistency_factor = max(0.5, 1.0 - (variance / (avg_blobs + 1)))
        
        # Combined certainty
        self.certainty = (age_factor * 0.5 + recency_factor * 0.3 + consistency_factor * 0.2)
        return self.certainty
    
    def predict_position(self, dt=1.0):
        """Predict next position based on velocity."""
        return self.position + self.velocity * dt
    
    def update(self, matched_blobs, frame_id):
        """Update track with matched blobs using smoothing for gradual transitions."""
        if not matched_blobs:
            self.frames_since_update += 1
            return
        
        # Reset activity counter when we have blobs (there's activity)
        self.frames_with_no_activity = 0
        
        # Calculate raw new position as weighted average of blob positions
        blob_positions = np.array([blob.pos for blob in matched_blobs])
        raw_new_position = np.mean(blob_positions, axis=0)
        
        # Apply low-pass filter for smooth position updates
        # position = alpha * raw + (1-alpha) * previous
        smoothed_position = self.alpha * raw_new_position + (1 - self.alpha) * self.position
        
        # Update velocity if we have history (use smoothed position)
        if len(self.position_history) >= 2:
            prev_pos, prev_frame = self.position_history[-1]
            dt = max(frame_id - prev_frame, 1)
            raw_velocity = (smoothed_position - prev_pos) / dt
            # Also smooth velocity
            self.velocity = self.alpha * raw_velocity + (1 - self.alpha) * self.velocity
        
        # Update position and history
        self.position = smoothed_position
        self.position_history.append((smoothed_position.copy(), frame_id))
        
        # Update blob tracking
        self.blobs = matched_blobs
        self.blob_history.append(len(matched_blobs))
        
        self.last_seen_frame = frame_id
        self.frames_since_update = 0
        self.total_updates += 1
    
    def check_area_activity(self, diff_map):
        """
        Check if there's any activity (non-zero pixels) in the track's area.
        Returns True if activity detected, False if area is completely inactive.
        """
        x, y = int(self.position[0]), int(self.position[1])
        radius = self.activity_check_radius
        
        # Get bounds
        height, width = diff_map.shape
        x_min = max(0, x - radius)
        x_max = min(width, x + radius)
        y_min = max(0, y - radius)
        y_max = min(height, y + radius)
        
        # Check if there's any non-zero activity in the region
        region = diff_map[y_min:y_max, x_min:x_max]
        has_activity = np.any(region > 0)
        
        return has_activity
    
    def update_activity_status(self, diff_map):
        """Update the no-activity frame counter based on diff_map."""
        if self.check_area_activity(diff_map):
            self.frames_with_no_activity = 0
        else:
            self.frames_with_no_activity += 1
    
    def is_alive(self, current_frame, max_frames_lost=30):
        """
        Check if track should be kept alive.
        Reduced lifetime and added activity-based killing.
        """
        frames_lost = current_frame - self.last_seen_frame
        
        # Kill immediately if no activity in area for several frames
        if self.frames_with_no_activity >= 5:  # 5 frames of zero activity = death
            return False
        
        # Reduced max frames tolerance
        if self.total_updates < 3:
            return frames_lost < 10  # New tracks: 10 frames (reduced from 15)
        
        return frames_lost < max_frames_lost  # Established tracks: 30 frames (reduced from 50)
    
    def get_speed(self):
        """Get current speed magnitude."""
        return np.linalg.norm(self.velocity)
    
    def __repr__(self):
        return f"Track(id={self.id}, pos={self.position}, vel={self.velocity}, age={self.total_updates})"


class Object:
    """Represents a high-level object composed of multiple tracks (Kalman-like fusion)."""
    
    next_id = 0
    
    def __init__(self, initial_track, frame_id):
        self.id = Object.next_id
        Object.next_id += 1
        
        self.position = initial_track.position.copy()
        self.velocity = initial_track.velocity.copy()
        
        # Low-pass filter for smoothing
        self.alpha = 0.3  # Smoothing factor (0=no change, 1=instant change)
        self.raw_position = initial_track.position.copy()  # Unfiltered position
        self.raw_velocity = initial_track.velocity.copy()  # Unfiltered velocity
        
        # Track pool: stores track objects (not just IDs) for active fusion
        self.track_pool = [initial_track]  # Active tracks contributing to this object
        self.track_history = deque(maxlen=10)  # Historical track IDs
        self.track_history.append(initial_track.id)
        
        self.last_seen_frame = frame_id
        self.creation_frame = frame_id
        self.frames_since_update = 0
        self.total_updates = 1
        
        self.color = tuple(np.random.randint(100, 255, 3).tolist())
        
        # Each object has its own importance map (initialized later with frame dimensions)
        self.importance_map = None
    
    def predict_position(self, dt=1.0):
        """Predict next position based on velocity."""
        return self.position + self.velocity * dt
    
    def calculate_track_correlation(self, track):
        """
        Calculate how well a track correlates with the object's current state.
        Returns correlation score 0-1 (1 = perfect correlation).
        """
        # Distance correlation (closer = better)
        distance = np.linalg.norm(track.position - self.position)
        max_distance = 80
        distance_corr = max(0, 1 - (distance / max_distance))
        
        # Velocity correlation (similar direction = better)
        vel_corr = 0.5
        if np.linalg.norm(track.velocity) > 0.5 and np.linalg.norm(self.velocity) > 0.5:
            track_vel_norm = track.velocity / np.linalg.norm(track.velocity)
            obj_vel_norm = self.velocity / np.linalg.norm(self.velocity)
            vel_dot = np.dot(track_vel_norm, obj_vel_norm)
            vel_corr = (vel_dot + 1) / 2  # Map from [-1,1] to [0,1]
        
        # Certainty factor
        certainty = track.get_certainty()
        
        # Combined correlation (weighted average)
        correlation = (distance_corr * 0.4 + vel_corr * 0.3 + certainty * 0.3)
        return correlation
    
    def update(self, candidate_tracks, frame_id, correlation_threshold=0.3):
        """
        Update object using multi-track fusion (Kalman-like).
        Uses all tracks above correlation threshold, weighted by correlation and certainty.
        """
        if not candidate_tracks:
            self.frames_since_update += 1
            # Keep predicting position based on velocity when no tracks
            if self.frames_since_update < 5:
                self.position = self.predict_position(1.0)
            self.track_pool = []  # Clear track pool
            return
        
        # Calculate correlation for each candidate track
        track_correlations = []
        for track in candidate_tracks:
            correlation = self.calculate_track_correlation(track)
            if correlation >= correlation_threshold:
                track_correlations.append((track, correlation))
        
        # If no tracks pass threshold, don't update (maintain prediction)
        if not track_correlations:
            self.frames_since_update += 1
            if self.frames_since_update < 5:
                self.position = self.predict_position(1.0)
            self.track_pool = []
            return
        
        # Update track pool with correlated tracks
        self.track_pool = [tc[0] for tc in track_correlations]
        for track, _ in track_correlations:
            if track.id not in self.track_history:
                self.track_history.append(track.id)
        
        # Multi-track fusion: weight by both correlation AND certainty
        weights = []
        for track, correlation in track_correlations:
            certainty = track.get_certainty()
            # Combined weight: correlation * certainty
            weight = correlation * certainty
            weights.append(weight)
        
        total_weight = sum(weights)
        
        if total_weight > 0:
            # Weighted average of positions (raw/unfiltered)
            weighted_positions = [track.position * weight 
                                 for (track, _), weight in zip(track_correlations, weights)]
            self.raw_position = sum(weighted_positions) / total_weight
            
            # Weighted average of velocities (raw/unfiltered)
            weighted_velocities = [track.velocity * weight 
                                  for (track, _), weight in zip(track_correlations, weights)]
            self.raw_velocity = sum(weighted_velocities) / total_weight
        else:
            # Fallback to simple average
            self.raw_position = np.mean([t.position for t, _ in track_correlations], axis=0)
            self.raw_velocity = np.mean([t.velocity for t, _ in track_correlations], axis=0)
        
        # Apply low-pass filter (exponential smoothing)
        # position = alpha * raw + (1-alpha) * previous
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
        """
        Update this object's importance map based on track importance.
        Tracks in track_pool are important (positive), others are unimportant (negative).
        
        Args:
            all_tracks: All alive tracks to categorize
            valuable_score: Positive score for important tracks
            unimportant_score: Negative score for unimportant tracks
            use_prediction: If True, paint predicted future positions
            prediction_steps: Number of future frames to predict (default 3)
        """
        if self.importance_map is None:
            return
        
        # Get track IDs in pool for quick lookup
        pool_track_ids = {t.id for t in self.track_pool}
        
        # Categorize tracks
        important_tracks = [t for t in all_tracks if t.id in pool_track_ids]
        unimportant_tracks = [t for t in all_tracks if t.id not in pool_track_ids]
        
        # Update the importance map with current positions
        self.importance_map.update(important_tracks, unimportant_tracks, 
                                   valuable_score, unimportant_score)
        
        # PREDICTIVE IMPORTANCE: Paint predicted future positions for fast-moving objects
        if use_prediction:
            object_speed = self.get_speed()
            
            # Use prediction if object is moving (speed > 10 px/frame)
            if object_speed > 10:
                # 1. Paint predicted OBJECT position trajectory (to prediction layer)
                for step in range(1, prediction_steps + 1):
                    # Predict where the object will be
                    predicted_obj_pos = self.position + self.velocity * step
                    
                    # Paint with gradually decreasing strength
                    prediction_strength = valuable_score * 2.0 * (1.0 - step * 0.15)  # Strong, visible
                    prediction_radius = 35 - step * 5  # 30, 25, 20 pixels
                    
                    self.importance_map.add_prediction(
                        predicted_obj_pos,
                        prediction_strength,
                        radius=prediction_radius
                    )
                
                # 2. Paint predicted TRACK positions (more detailed predictions to prediction layer)
                for track in important_tracks:
                    track_speed = track.get_speed()
                    
                    # Only predict for moving tracks
                    if track_speed > 5:
                        for step in range(1, prediction_steps + 1):
                            # Predict position: current + velocity * time_steps
                            predicted_track_pos = track.position + track.velocity * step
                            
                            # Smaller influence, more focused, but still visible
                            track_prediction_strength = valuable_score * 1.5 * (1.0 - step * 0.2)
                            track_prediction_radius = max(12, 25 - step * 4)  # 21, 17, 13 pixels
                            
                            self.importance_map.add_prediction(
                                predicted_track_pos,
                                track_prediction_strength,
                                radius=track_prediction_radius
                            )
    
    def __repr__(self):
        return f"Object(id={self.id}, pos={self.position}, tracks={len(self.track_pool)})"


def calculate_track_object_score(track, obj, frame_id):
    """
    Calculate matching score between a track and an object.
    Higher score = better match. Heavily favors high-certainty tracks.
    """
    score = 0.0
    
    # 0. Track certainty multiplier (CRITICAL - prevents jumping to unstable tracks)
    certainty = track.get_certainty()
    if certainty < 0.2:  # Very uncertain tracks get heavily penalized (relaxed from 0.3)
        return 0.0
    
    # 1. Distance score
    predicted_pos = obj.predict_position()
    distance = np.linalg.norm(track.position - predicted_pos)
    max_distance = 100  # pixels - objects can be larger
    
    if distance > max_distance:
        return 0.0
    
    distance_score = (max_distance - distance) / max_distance * 40
    score += distance_score
    
    # 2. Velocity correlation
    if np.linalg.norm(track.velocity) > 0.1 and np.linalg.norm(obj.velocity) > 0.1:
        track_vel_norm = track.velocity / (np.linalg.norm(track.velocity) + 1e-6)
        obj_vel_norm = obj.velocity / (np.linalg.norm(obj.velocity) + 1e-6)
        vel_correlation = np.dot(track_vel_norm, obj_vel_norm)
        vel_score = (vel_correlation + 1) / 2 * 20
        score += vel_score
    
    # 3. Recency bonus
    frames_since = frame_id - obj.last_seen_frame
    if frames_since == 0:
        recency_score = 10
    elif frames_since == 1:
        recency_score = 8
    elif frames_since == 2:
        recency_score = 5
    else:
        recency_score = 2
    score += recency_score
    
    # 4. CERTAINTY MULTIPLIER - Scales the entire score by certainty
    # High certainty tracks get proportionally higher total scores
    # Instead of flat bonus, multiply base score by (1 + certainty)
    # This gives 0-100% boost based on certainty level
    score *= (1.0 + certainty)
    
    return score


def match_tracks_to_objects(tracks, objects, frame_id, min_score=15):
    """
    Match tracks to objects using scoring system.
    Multiple tracks can be assigned to the same object.
    Returns: (object_assignments, unmatched_tracks)
    """
    if not objects or not tracks:
        return {}, tracks
    
    # Calculate scores for all track-object pairs
    scores = np.zeros((len(tracks), len(objects)))
    for i, track in enumerate(tracks):
        for j, obj in enumerate(objects):
            scores[i, j] = calculate_track_object_score(track, obj, frame_id)
    
    # Modified assignment: allow multiple tracks per object
    object_assignments = {obj.id: [] for obj in objects}
    assigned_tracks = set()
    
    while True:
        max_score = np.max(scores)
        if max_score < min_score:
            break
        
        track_idx, obj_idx = np.unravel_index(np.argmax(scores), scores.shape)
        
        # Assign track to object
        object_assignments[objects[obj_idx].id].append(tracks[track_idx])
        assigned_tracks.add(track_idx)
        
        # Zero out ONLY this track (not the object column!)
        # This allows multiple tracks to match the same object
        scores[track_idx, :] = 0
    
    # Collect unmatched tracks
    unmatched_tracks = [track for i, track in enumerate(tracks) if i not in assigned_tracks]
    
    return object_assignments, unmatched_tracks


def calculate_blob_track_score(blob, track, frame_id):
    """
    Calculate matching score between a blob and a track.
    Higher score = better match.
    
    Scoring factors:
    - Distance to predicted position
    - Velocity correlation
    - Size similarity
    - Recency of track updates
    """
    score = 0.0
    
    # 1. Distance score (closer = higher score)
    predicted_pos = track.predict_position()
    distance = np.linalg.norm(blob.pos - predicted_pos)
    max_distance = 80  # pixels - permissive for measurement-style tracking
    
    # No score if too far away
    if distance > max_distance:
        return 0.0
    
    distance_score = (max_distance - distance) / max_distance * 50
    score += distance_score
    
    # 2. Velocity correlation (if blob has velocity info from previous frame)
    if np.linalg.norm(blob.velocity) > 0.1 and np.linalg.norm(track.velocity) > 0.1:
        # Normalize velocities
        blob_vel_norm = blob.velocity / (np.linalg.norm(blob.velocity) + 1e-6)
        track_vel_norm = track.velocity / (np.linalg.norm(track.velocity) + 1e-6)
        
        # Dot product gives cosine similarity (-1 to 1)
        vel_correlation = np.dot(blob_vel_norm, track_vel_norm)
        vel_score = (vel_correlation + 1) / 2 * 30  # Scale to 0-30
        score += vel_score
    
    # 3. Recency bonus (recently updated tracks get priority)
    frames_since = frame_id - track.last_seen_frame
    if frames_since == 0:
        recency_score = 20
    elif frames_since == 1:
        recency_score = 15
    elif frames_since == 2:
        recency_score = 10
    else:
        recency_score = max(0, 10 - frames_since)
    score += recency_score
    
    return score


def match_blobs_to_tracks(blobs, tracks, frame_id, min_score=15):
    """
    Match detected blobs to existing tracks using scoring system.
    Returns: (track_assignments, unmatched_blobs)
    
    min_score: minimum score required for a match (permissive for raw measurements)
    """
    if not tracks or not blobs:
        return {}, blobs
    
    # Calculate scores for all blob-track pairs
    scores = np.zeros((len(blobs), len(tracks)))
    for i, blob in enumerate(blobs):
        for j, track in enumerate(tracks):
            scores[i, j] = calculate_blob_track_score(blob, track, frame_id)
    
    # Greedy assignment: repeatedly assign best scoring pairs
    track_assignments = {track.id: [] for track in tracks}
    assigned_blobs = set()
    
    while True:
        # Find highest score
        max_score = np.max(scores)
        if max_score < min_score:  # Must meet minimum threshold
            break
        
        blob_idx, track_idx = np.unravel_index(np.argmax(scores), scores.shape)
        
        # Assign blob to track
        track_assignments[tracks[track_idx].id].append(blobs[blob_idx])
        assigned_blobs.add(blob_idx)
        
        # Zero out this blob and track from further matching
        scores[blob_idx, :] = 0
        scores[:, track_idx] = 0
    
    # Collect unmatched blobs
    unmatched_blobs = [blob for i, blob in enumerate(blobs) if i not in assigned_blobs]
    
    return track_assignments, unmatched_blobs


def find_persistent_blobs(unmatched_blobs, recent_unmatched_history, min_persistence=3, max_distance=50):
    """
    Find unmatched blobs that have persisted across multiple frames.
    Returns a list of blobs that should become new tracks.
    """
    if len(unmatched_blobs) == 0 or len(recent_unmatched_history) < min_persistence:
        return []
    
    persistent_blobs = []
    
    # For each current unmatched blob, check if it has persisted
    for current_blob in unmatched_blobs:
        persistence_count = 1  # Current frame counts as 1
        
        # Look backwards through history
        last_position = current_blob.pos
        for hist_blobs in reversed(list(recent_unmatched_history)[-min_persistence:]):
            if len(hist_blobs) == 0:
                break
            
            # Find closest blob in this historical frame
            closest_dist = float('inf')
            closest_blob = None
            for hist_blob in hist_blobs:
                dist = np.linalg.norm(last_position - hist_blob.pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_blob = hist_blob
            
            # If we found a close blob, this is part of the persistent sequence
            if closest_blob and closest_dist < max_distance:
                persistence_count += 1
                last_position = closest_blob.pos  # Track the blob's movement
            else:
                break  # Sequence broken
        
        # If blob persisted long enough, mark it for track creation
        if persistence_count >= min_persistence:
            persistent_blobs.append(current_blob)
    
    return persistent_blobs


def setup_blob_detector():
    """Configure SimpleBlobDetector with parameters optimized for motion detection."""
    params = cv.SimpleBlobDetector_Params()
    
    # Filter by area - lowered min to track smaller targets
    params.filterByArea = True
    params.minArea = 25  # Reduced from 50 for even smaller targets
    params.maxArea = 2000
    
    # Filter by circularity (0=line, 1=circle)
    params.filterByCircularity = True
    params.minCircularity = 0.1
    params.maxCircularity = 1.0
    
    # Filter by convexity
    params.filterByConvexity = False
    
    # Filter by inertia (elongation)
    params.filterByInertia = False
    
    # Color (detect bright blobs on dark background)
    params.filterByColor = False
    
    return cv.SimpleBlobDetector_create(params)


class ImportanceMap:
    """
    Spatial importance map with three separate layers:
    - Positive layer (valuable tracks) - fast decay
    - Negative layer (noise) - persistent slow decay
    - Prediction layer (future positions) - very fast decay, always visible
    """
    
    def __init__(self, width, height, positive_decay=0.85, negative_decay=0.98, prediction_decay=0.75):
        """
        Args:
            width: Frame width
            height: Frame height
            positive_decay: Fast decay for positive scores (0.85 = 15% decay per frame)
            negative_decay: Slow decay for negative scores (0.98 = 2% decay per frame)
            prediction_decay: Very fast decay for predictions (0.75 = 25% decay per frame)
        """
        self.positive_map = np.zeros((height, width), dtype=np.float32)
        self.negative_map = np.zeros((height, width), dtype=np.float32)
        self.prediction_map = np.zeros((height, width), dtype=np.float32)  # NEW: separate prediction layer
        self.positive_decay = positive_decay
        self.negative_decay = negative_decay
        self.prediction_decay = prediction_decay
        self.width = width
        self.height = height
    
    def add_gaussian_influence(self, center, value, radius=30):
        """
        Add a Gaussian-weighted influence around a point.
        Automatically routes to positive or negative map.
        
        Args:
            center: (x, y) position
            value: Amount to add (positive for good, negative for bad)
            radius: Radius of influence in pixels
        """
        x, y = int(center[0]), int(center[1])
        
        # Create coordinate grids
        y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
        
        # Gaussian formula: exp(-(x^2 + y^2) / (2 * sigma^2))
        sigma = radius / 3.0  # 3-sigma rule
        gaussian = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        
        # Calculate the region bounds
        x_min = max(0, x - radius)
        x_max = min(self.width, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(self.height, y + radius + 1)
        
        # Calculate corresponding gaussian region
        g_x_min = max(0, radius - x)
        g_x_max = g_x_min + (x_max - x_min)
        g_y_min = max(0, radius - y)
        g_y_max = g_y_min + (y_max - y_min)
        
        influence = gaussian[g_y_min:g_y_max, g_x_min:g_x_max] * value
        
        # Route to appropriate map
        if value > 0:
            # Positive: add to positive map
            self.positive_map[y_min:y_max, x_min:x_max] += influence
        else:
            # Negative: add to negative map (store as positive values)
            # Only add negative if not already very negative (prevent stacking)
            region = self.negative_map[y_min:y_max, x_min:x_max]
            self.negative_map[y_min:y_max, x_min:x_max] = np.maximum(region, -influence)
    
    def update(self, valuable_tracks, noise_tracks, valuable_score=5.0, noise_score=-3.0):
        """
        Update importance map based on current tracks.
        
        Args:
            valuable_tracks: List of tracks being used by objects
            noise_tracks: List of tracks not being used (noise)
            valuable_score: Positive value to add around good tracks
            noise_score: Negative value to add around bad tracks
        """
        # Apply different decay rates
        self.positive_map *= self.positive_decay  # Fast decay (15% per frame)
        self.negative_map *= self.negative_decay  # Slow decay (2% per frame)
        self.prediction_map *= self.prediction_decay  # Very fast decay (25% per frame)
        
        # Add positive influence around valuable tracks
        for track in valuable_tracks:
            self.add_gaussian_influence(track.position, valuable_score, radius=30)
        
        # Add negative influence around noise tracks
        for track in noise_tracks:
            self.add_gaussian_influence(track.position, noise_score, radius=25)
    
    def add_prediction(self, center, value, radius=20):
        """
        Add prediction influence to the separate prediction layer.
        This layer decays very fast to create visible corridors.
        
        Args:
            center: (x, y) predicted position
            value: Importance value (typically positive)
            radius: Radius of influence
        """
        x, y = int(center[0]), int(center[1])
        
        # Create coordinate grids
        y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
        
        # Gaussian formula
        sigma = radius / 3.0
        gaussian = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        
        # Calculate bounds
        x_min = max(0, x - radius)
        x_max = min(self.width, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(self.height, y + radius + 1)
        
        # Calculate corresponding gaussian region
        g_x_min = max(0, radius - x)
        g_x_max = g_x_min + (x_max - x_min)
        g_y_min = max(0, radius - y)
        g_y_max = g_y_min + (y_max - y_min)
        
        # Add to prediction layer only
        influence = gaussian[g_y_min:g_y_max, g_x_min:g_x_max] * value
        self.prediction_map[y_min:y_max, x_min:x_max] += influence
    
    def get_score(self, position):
        """Get combined importance score at a position."""
        x, y = int(position[0]), int(position[1])
        if 0 <= x < self.width and 0 <= y < self.height:
            # Combine all three layers
            positive_val = self.positive_map[y, x]
            negative_val = self.negative_map[y, x]
            prediction_val = self.prediction_map[y, x]
            
            # Combined score: positive + prediction - negative
            return positive_val + prediction_val - negative_val
        return 0.0
    
    def get_visualization(self):
        """Get visualization of the combined importance map."""
        # Combine all three layers
        combined = self.positive_map + self.prediction_map - self.negative_map
        
        # Clip extreme values for better visualization
        combined = np.clip(combined, -20, 20)
        
        # Normalize to 0-255
        map_min = combined.min()
        map_max = combined.max()
        if map_max > map_min:
            map_vis = ((combined - map_min) / (map_max - map_min) * 255).astype(np.uint8)
        else:
            map_vis = np.zeros_like(combined, dtype=np.uint8)
        
        # Apply colormap (blue=negative, green=neutral, red=positive)
        return cv.applyColorMap(map_vis, cv.COLORMAP_JET)
    
    def reset(self):
        """Reset all three maps to zeros."""
        self.positive_map.fill(0)
        self.negative_map.fill(0)
        self.prediction_map.fill(0)


def main():
    camera_index = 0
    cap = cv.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        return
    
    print(f"Camera {camera_index} opened successfully!")
    
    # Scene configuration
    MAX_OBJECTS = 2  # Fixed number of objects in the scene
    print(f"\nScene Configuration: {MAX_OBJECTS} objects")
    
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save current diff_map")
    print("  'd' - Toggle debug views")
    print("  'c' - Clear/reset")
    print("  'b' - Toggle blob detection overlay")
    print("  't' - Toggle track visualization")
    print("  'o' - Toggle object visualization")
    print("  'i' - Toggle importance map visualization")
    print("  'p' - Toggle predictive importance mapping")
    print("  '1-9' - Set number of objects in scene")
    print("  '+/-' - Increase/decrease prediction steps")
    print()
    
    prev_gray = None
    prev_blobs = []
    show_debug = True
    show_blobs = True
    show_tracks = True
    show_importance = False  # Toggle for importance map visualization
    use_prediction = True  # Toggle for predictive importance
    prediction_steps = 3  # Number of future frames to predict
    frame_count = 0
    saved_count = 0
    max_objects = MAX_OBJECTS
    
    # Setup blob detector
    blob_detector = setup_blob_detector()
    
    # Track management
    tracks = []
    unmatched_history = deque(maxlen=5)
    last_frame_time = time.time()
    
    # Object management (higher level grouping of tracks)
    objects = []
    unmatched_tracks_history = deque(maxlen=3)
    show_objects = True
    
    # Temporal smoothing for diff_map to reduce jumpiness
    accumulated_diff = None
    diff_alpha = 0.9  # Blending factor for temporal smoothing
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        frame_count += 1
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        if prev_gray is not None:
            # Calculate difference map
            diff_map = mf.picture_diference(gray, prev_gray)
            
            # Process diff_map: normalize first
            diff_map = cv.normalize(diff_map, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            
            # Apply temporal smoothing to reduce jumpiness from lighting/camera changes
            if accumulated_diff is None:
                accumulated_diff = diff_map.astype(np.float32)
            else:
                # Blend current diff with accumulated: new = alpha*current + (1-alpha)*previous
                accumulated_diff = diff_alpha * diff_map.astype(np.float32) + (1 - diff_alpha) * accumulated_diff
            
            # Use the smoothed diff map
            diff_map_smoothed = accumulated_diff.astype(np.uint8)
            
            # Lighter blur to preserve detail
            diff_map_smoothed = cv.GaussianBlur(diff_map_smoothed, (5, 5), 2)
            
            # Lower threshold to capture more subtle changes
            diff_map_final = cv.threshold(diff_map_smoothed, 20, 255, cv.THRESH_BINARY)[1]
            
            # Detect blobs in the thresholded diff_map
            keypoints = blob_detector.detect(diff_map_final)
            
            # Convert keypoints to BlobInfo objects with velocity calculation
            current_time = time.time()
            dt = current_time - last_frame_time
            last_frame_time = current_time
            
            blobs = []
            for kp in keypoints:
                blob = BlobInfo(kp, frame_count)
                
                # Calculate velocity by finding closest previous blob
                if prev_blobs:
                    closest_dist = float('inf')
                    closest_blob = None
                    for prev_blob in prev_blobs:
                        dist = np.linalg.norm(blob.pos - prev_blob.pos)
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_blob = prev_blob
                    
                    # Only calculate velocity if close enough (< 50 pixels)
                    if closest_blob and closest_dist < 50:
                        if dt > 0:
                            blob.velocity = (blob.pos - closest_blob.pos) / dt
                
                blobs.append(blob)
            
            # Match blobs to existing tracks
            track_assignments, unmatched_blobs = match_blobs_to_tracks(blobs, tracks, frame_count)
            
            # Update tracks with matched blobs
            for track in tracks:
                if track.id in track_assignments:
                    track.update(track_assignments[track.id], frame_count)
            
            # Update activity status for all tracks based on final diff_map
            for track in tracks:
                track.update_activity_status(diff_map_final)
            
            # Add unmatched blobs to history
            unmatched_history.append(unmatched_blobs)
            
            # Find persistent unmatched blobs that should become new tracks
            # Keep permissive - tracks are like measurements, we'll filter later
            persistent_blobs = find_persistent_blobs(unmatched_blobs, unmatched_history, 
                                                     min_persistence=3, max_distance=50)
            
            # Create new tracks for persistent blobs
            for persistent_blob in persistent_blobs:
                # Don't create track if it's extremely close to existing one
                too_close_to_existing = False
                for existing_track in tracks:
                    dist = np.linalg.norm(persistent_blob.pos - existing_track.position)
                    if dist < 30:  # Only reject if very close
                        too_close_to_existing = True
                        break
                
                if not too_close_to_existing:
                    new_track = Track(persistent_blob, frame_count)
                    tracks.append(new_track)
                    print(f"Created new track #{new_track.id}")
            
            # Remove dead tracks
            alive_tracks = [t for t in tracks if t.is_alive(frame_count)]
            
            # Initialize valuable_track_ids as empty
            valuable_track_ids = set()
            
            # === OBJECT LEVEL TRACKING (Fixed number of objects) ===
            # Initialize objects if needed (first tracks appear)
            if len(objects) < max_objects and len(alive_tracks) > 0:
                # Create objects from first tracks seen
                for track in alive_tracks:
                    if len(objects) >= max_objects:
                        break
                    
                    # Check if this track is far from existing objects
                    too_close = False
                    for obj in objects:
                        if np.linalg.norm(track.position - obj.position) < 100:
                            too_close = True
                            break
                    
                    if not too_close:
                        new_object = Object(track, frame_count)
                        # Initialize object's importance map with frame dimensions
                        height, width = gray.shape
                        # Fast decay for positive (85%), slow for negative (98%), very fast for predictions (75%)
                        new_object.importance_map = ImportanceMap(width, height, 
                                                                   positive_decay=0.85, 
                                                                   negative_decay=0.98,
                                                                   prediction_decay=0.75)
                        objects.append(new_object)
                        print(f"Initialized object #{new_object.id} at pos {track.position}")
            
            # Match all tracks to the fixed set of objects
            if alive_tracks and objects:
                object_assignments, unmatched_tracks = match_tracks_to_objects(
                    alive_tracks, objects, frame_count, min_score=20
                )
                
                # Update objects with matched tracks (multi-track fusion)
                for obj in objects:
                    if obj.id in object_assignments and object_assignments[obj.id]:
                        # Pass all candidate tracks to object for correlation-based fusion
                        obj.update(object_assignments[obj.id], frame_count, correlation_threshold=0.35)
                    else:
                        # Object had no matching tracks this frame
                        obj.update([], frame_count)
                
                # Update categorization: valuable = in any object's track pool
                valuable_track_ids.clear()
                for obj in objects:
                    # Include all tracks actively contributing to this object
                    valuable_track_ids.update([t.id for t in obj.track_pool])
                    
                    # Update this object's importance map with prediction
                    obj.update_importance_map(alive_tracks, 
                                             valuable_score=5.0, 
                                             unimportant_score=-3.0,
                                             use_prediction=use_prediction,
                                             prediction_steps=prediction_steps)
            
            # Objects never get removed - they represent the fixed scene entities
            
            tracks = alive_tracks
            
            # Store blobs for next frame
            prev_blobs = blobs
            
            # Display smoothed diff_map (final processed version)
            cv.imshow("Diff Map", diff_map_final)
            
            # Create visualization frame
            vis_frame = frame.copy()
            
            # Draw blobs if enabled
            if show_blobs and keypoints:
                for kp in keypoints:
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    size = int(kp.size)
                    cv.circle(vis_frame, (x, y), size, (0, 255, 0), 2)
                    cv.circle(vis_frame, (x, y), 3, (0, 0, 255), -1)
            
            # Draw tracks if enabled (with categorization and certainty)
            if show_tracks and tracks:
                # Pre-calculate track scores for their assigned objects
                track_scores = {}
                if objects:
                    for obj in objects:
                        for track in obj.track_pool:
                            score = calculate_track_object_score(track, obj, frame_count)
                            track_scores[track.id] = score
                
                for track in tracks:
                    # Check if track is valuable (assigned to an object) or noise
                    is_valuable = track.id in valuable_track_ids if 'valuable_track_ids' in locals() else False
                    
                    # Get certainty for visual feedback
                    certainty = track.get_certainty()
                    
                    # Get score if track is assigned to an object
                    track_score = track_scores.get(track.id, 0) if is_valuable else 0
                    
                    # Color: GREEN only if valuable (in object), otherwise GRAY
                    if is_valuable:
                        # Pure green for valuable tracks
                        ring_color = (0, 255, 0)
                    else:
                        # Gray for noise tracks
                        ring_color = (100, 100, 100)
                    
                    # Use original track color for center dot
                    track_color = track.color if is_valuable else (80, 80, 80)
                    
                    # Draw track history trail (thickness based on certainty)
                    if len(track.position_history) > 1:
                        # position_history stores (pos, frame_id) tuples
                        points = [tuple(map(int, pos_frame[0])) for pos_frame in track.position_history]
                        line_thickness = max(1, int(certainty * 3)) if is_valuable else 1
                        for i in range(len(points) - 1):
                            cv.line(vis_frame, points[i], points[i + 1], track_color, line_thickness)
                    
                    # Draw current position (fixed small size)
                    pos = tuple(map(int, track.position))
                    circle_size = 4 if is_valuable else 3
                    cv.circle(vis_frame, pos, circle_size, track_color, -1)
                    
                    # Outer ring: size based on SCORE, thickness based on CERTAINTY, color based on valuable/noise
                    if is_valuable:
                        # Ring radius scales with score (normalized to 0-100 range, giving 2-12 pixel ring)
                        normalized_score = min(track_score / 100.0, 1.0)  # Normalize assuming max ~100
                        ring_radius = circle_size + int(2 + normalized_score * 10)
                        ring_thickness = max(1, int(certainty * 3))
                    else:
                        ring_radius = circle_size + 2
                        ring_thickness = 1
                    
                    cv.circle(vis_frame, pos, ring_radius, ring_color, ring_thickness)
                    
                    # Draw velocity vector if significant (only for valuable tracks)
                    if is_valuable:
                        speed = track.get_speed()
                        if speed > 5:  # Only draw if moving
                            end_pos = tuple(map(int, track.position + track.velocity * 0.3))
                            cv.arrowedLine(vis_frame, pos, end_pos, track_color, 2, tipLength=0.3)
                    
                    # Draw track ID (smaller for noise)
                    font_scale = 0.5 if is_valuable else 0.3
                    cv.putText(vis_frame, f"#{track.id}", 
                              (pos[0] + 15, pos[1] - 10),
                              cv.FONT_HERSHEY_SIMPLEX, font_scale, track_color, 2 if is_valuable else 1)
            
            # Draw objects if enabled (multi-track fusion visualization)
            if show_objects and objects:
                for obj in objects:
                    # Draw lines connecting object to its active track pool
                    obj_pos = tuple(map(int, obj.position))
                    for track in obj.track_pool:
                        track_pos = tuple(map(int, track.position))
                        # Line thickness based on track certainty
                        certainty = track.get_certainty()
                        correlation = obj.calculate_track_correlation(track)
                        line_thickness = max(1, int(correlation * certainty * 3))
                        cv.line(vis_frame, obj_pos, track_pos, obj.color, line_thickness)
                    
                    # Draw large circle representing the fused object position
                    cv.circle(vis_frame, obj_pos, 35, obj.color, 4)
                    
                    # Draw velocity vector
                    speed = obj.get_speed()
                    if speed > 3:
                        end_pos = tuple(map(int, obj.position + obj.velocity * 0.5))
                        cv.arrowedLine(vis_frame, obj_pos, end_pos, obj.color, 4, tipLength=0.3)
                    
                    # Draw object ID and active track pool size
                    active_tracks = len(obj.track_pool)
                    cv.putText(vis_frame, f"OBJ #{obj.id} [{active_tracks} tracks]", 
                              (obj_pos[0] - 50, obj_pos[1] - 45),
                              cv.FONT_HERSHEY_SIMPLEX, 0.7, obj.color, 2)
                    
                    # Show track IDs in pool (for debugging)
                    if active_tracks > 0:
                        track_ids = ",".join([f"#{t.id}" for t in obj.track_pool[:5]])  # Show first 5
                        cv.putText(vis_frame, track_ids, 
                                  (obj_pos[0] - 50, obj_pos[1] + 50),
                                  cv.FONT_HERSHEY_SIMPLEX, 0.4, obj.color, 1)
            
            # Debug views
            if show_debug:
                colored_diff = cv.applyColorMap(diff_map_final, cv.COLORMAP_JET)
                cv.imshow("Diff Map (Colored)", colored_diff)
                
                overlay = cv.addWeighted(frame, 0.3, colored_diff, 0.7, 0)
                cv.imshow("Overlay", overlay)
            
            # Importance map visualization (show combined map of all objects)
            if show_importance and objects:
                # Create combined maps from all objects (three layers)
                height, width = gray.shape
                combined_positive = np.zeros((height, width), dtype=np.float32)
                combined_negative = np.zeros((height, width), dtype=np.float32)
                combined_prediction = np.zeros((height, width), dtype=np.float32)
                
                for obj in objects:
                    if obj.importance_map:
                        combined_positive += obj.importance_map.positive_map
                        combined_negative += obj.importance_map.negative_map
                        combined_prediction += obj.importance_map.prediction_map
                
                # Create full combined view (positive + prediction - negative)
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
                
                # Show PREDICTION layer separately if prediction is enabled
                if use_prediction and np.max(combined_prediction) > 0.1:
                    pred_vis = np.clip(combined_prediction, 0, 20)
                    if pred_vis.max() > 0:
                        pred_vis = (pred_vis / pred_vis.max() * 255).astype(np.uint8)
                    else:
                        pred_vis = np.zeros_like(pred_vis, dtype=np.uint8)
                    
                    # Use HOT colormap for predictions (black -> red -> yellow -> white)
                    pred_colored = cv.applyColorMap(pred_vis, cv.COLORMAP_HOT)
                    pred_overlay = cv.addWeighted(frame, 0.5, pred_colored, 0.5, 0)
                    
                    cv.putText(pred_overlay, f"Prediction Corridors ({prediction_steps} steps)", 
                              (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv.putText(pred_overlay, "Red/Yellow = Predicted positions", 
                              (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    cv.imshow("Prediction Corridors", pred_overlay)
                else:
                    try:
                        cv.destroyWindow("Prediction Corridors")
                    except:
                        pass
                    
            elif not show_importance:
                # Close importance map windows if toggled off
                try:
                    cv.destroyWindow("Combined Importance Map")
                    cv.destroyWindow("Prediction Corridors")
                except:
                    pass
            
            # Statistics
            mean_diff = np.mean(diff_map_final)
            max_diff = np.max(diff_map_final)
            num_blobs = len(keypoints)
            
            # Info text
            cv.putText(vis_frame, f"Frame: {frame_count}", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv.putText(vis_frame, f"Blobs: {num_blobs}", (10, 60), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Count valuable vs noise tracks
            num_valuable = len(valuable_track_ids) if 'valuable_track_ids' in locals() else 0
            num_noise = len(tracks) - num_valuable
            
            cv.putText(vis_frame, f"Tracks: {len(tracks)} ({num_valuable} valuable, {num_noise} noise)", (10, 90), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv.putText(vis_frame, f"Objects: {len(objects)}/{max_objects}", (10, 120), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv.putText(vis_frame, f"Mean: {mean_diff:.1f} Max: {max_diff:.1f}", (10, 150), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv.putText(vis_frame, f"Blobs: {'ON' if show_blobs else 'OFF'} (b)", (10, 180), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv.putText(vis_frame, f"Tracks: {'ON' if show_tracks else 'OFF'} (t)", (10, 210), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv.putText(vis_frame, f"Objects: {'ON' if show_objects else 'OFF'} (o)", (10, 240), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv.putText(vis_frame, f"Importance: {'ON' if show_importance else 'OFF'} (i)", (10, 270), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv.putText(vis_frame, f"Prediction: {'ON' if use_prediction else 'OFF'} (p) Steps: {prediction_steps} (+/-)", (10, 300), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv.imshow("Frame with Blobs", vis_frame)
        else:
            # First frame
            cv.putText(frame, "Capturing first frame...", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv.imshow("Frame with Blobs", frame)
        
        prev_gray = gray.copy()
        
        # Keyboard input
        key = cv.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            if 'diff_map_final' in locals():
                filename = f"diff_map_{saved_count:03d}.png"
                cv.imwrite(filename, diff_map_final)
                print(f"Saved: {filename}")
                saved_count += 1
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"Debug views: {'ON' if show_debug else 'OFF'}")
            if not show_debug:
                cv.destroyWindow("Diff Map (Colored)")
                cv.destroyWindow("Overlay")
        elif key == ord('b'):
            show_blobs = not show_blobs
            print(f"Blob detection: {'ON' if show_blobs else 'OFF'}")
        elif key == ord('t'):
            show_tracks = not show_tracks
            print(f"Track visualization: {'ON' if show_tracks else 'OFF'}")
        elif key == ord('o'):
            show_objects = not show_objects
            print(f"Object visualization: {'ON' if show_objects else 'OFF'}")
        elif key == ord('i'):
            show_importance = not show_importance
            print(f"Importance map: {'ON' if show_importance else 'OFF'}")
        elif key == ord('p'):
            use_prediction = not use_prediction
            print(f"Predictive importance: {'ON' if use_prediction else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            prediction_steps = min(10, prediction_steps + 1)
            print(f"Prediction steps: {prediction_steps}")
        elif key == ord('-') or key == ord('_'):
            prediction_steps = max(1, prediction_steps - 1)
            print(f"Prediction steps: {prediction_steps}")
        elif key >= ord('1') and key <= ord('9'):
            # Change max number of objects in scene
            new_max = key - ord('0')
            if new_max != max_objects:
                max_objects = new_max
                # Remove excess objects if reducing
                if len(objects) > max_objects:
                    objects = objects[:max_objects]
                print(f"Max objects set to: {max_objects}")
        elif key == ord('c'):
            prev_gray = None
            tracks.clear()
            prev_blobs = []
            unmatched_history.clear()
            unmatched_tracks_history.clear()
            accumulated_diff = None  # Reset temporal smoothing
            Track.next_id = 0
            # Reset all object importance maps before clearing
            for obj in objects:
                if obj.importance_map:
                    obj.importance_map.reset()
            objects.clear()
            Object.next_id = 0
            print("Reset - cleared previous frame, all tracks, objects, importance maps, and temporal buffer")
    
    cap.release()
    cv.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames")
    print(f"Saved {saved_count} diff_maps")
    print(f"Total tracks created: {Track.next_id}")
    print(f"Total objects created: {Object.next_id}")


if __name__ == "__main__":
    print("=" * 60)
    print("Diff Map Demo")
    print("=" * 60)
    main()
