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
        """Update track with matched blobs."""
        if not matched_blobs:
            self.frames_since_update += 1
            return
        
        # Calculate new position as weighted average of blob positions
        blob_positions = np.array([blob.pos for blob in matched_blobs])
        new_position = np.mean(blob_positions, axis=0)
        
        # Update velocity if we have history
        if len(self.position_history) >= 2:
            prev_pos, prev_frame = self.position_history[-1]
            dt = max(frame_id - prev_frame, 1)
            self.velocity = (new_position - prev_pos) / dt
        
        # Update position and history
        self.position = new_position
        self.position_history.append((new_position.copy(), frame_id))
        
        # Update blob tracking
        self.blobs = matched_blobs
        self.blob_history.append(len(matched_blobs))
        
        self.last_seen_frame = frame_id
        self.frames_since_update = 0
        self.total_updates += 1
    
    def is_alive(self, current_frame, max_frames_lost=30):
        """Check if track should be kept alive - increased lifetime for better persistence."""
        frames_lost = current_frame - self.last_seen_frame
        
        # Need minimum number of updates to be considered established
        if self.total_updates < 3:
            return frames_lost < 10  # Increased from 5
        
        return frames_lost < max_frames_lost  # Increased from 10
    
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
        
        # Track pool: stores track objects (not just IDs) for active fusion
        self.track_pool = [initial_track]  # Active tracks contributing to this object
        self.track_history = deque(maxlen=10)  # Historical track IDs
        self.track_history.append(initial_track.id)
        
        self.last_seen_frame = frame_id
        self.creation_frame = frame_id
        self.frames_since_update = 0
        self.total_updates = 1
        
        self.color = tuple(np.random.randint(100, 255, 3).tolist())
    
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
            # Weighted average of positions
            weighted_positions = [track.position * weight 
                                 for (track, _), weight in zip(track_correlations, weights)]
            self.position = sum(weighted_positions) / total_weight
            
            # Weighted average of velocities
            weighted_velocities = [track.velocity * weight 
                                  for (track, _), weight in zip(track_correlations, weights)]
            self.velocity = sum(weighted_velocities) / total_weight
        else:
            # Fallback to simple average
            self.position = np.mean([t.position for t, _ in track_correlations], axis=0)
            self.velocity = np.mean([t.velocity for t, _ in track_correlations], axis=0)
        
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
    
    def __repr__(self):
        return f"Object(id={self.id}, pos={self.position}, tracks={len(self.tracks)})"


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
    
    # 4. CERTAINTY BONUS - This is the key to preventing jumps!
    # High certainty tracks get massive bonus (up to 30 points)
    certainty_bonus = certainty * 30
    score += certainty_bonus
    
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
    print("  '1-9' - Set number of objects in scene")
    print()
    
    prev_gray = None
    prev_blobs = []
    show_debug = True
    show_blobs = True
    show_tracks = True
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
            
            # Process diff_map: normalize, blur, threshold
            diff_map = cv.normalize(diff_map, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            diff_map = cv.GaussianBlur(diff_map, (9, 9), 4)
            diff_map = cv.threshold(diff_map, 30, 255, cv.THRESH_BINARY)[1]
            
            # Detect blobs in the thresholded diff_map
            keypoints = blob_detector.detect(diff_map)
            
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
                
                # Categorize tracks: valuable = in any object's track pool
                valuable_track_ids = set()
                for obj in objects:
                    # Include all tracks actively contributing to this object
                    valuable_track_ids.update([t.id for t in obj.track_pool])
            
            # Objects never get removed - they represent the fixed scene entities
            
            tracks = alive_tracks
            
            # Store blobs for next frame
            prev_blobs = blobs
            
            # Display diff_map
            cv.imshow("Diff Map", diff_map)
            
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
                for track in tracks:
                    # Check if track is valuable (assigned to an object) or noise
                    is_valuable = track.id in valuable_track_ids if 'valuable_track_ids' in locals() else False
                    
                    # Get certainty for visual feedback
                    certainty = track.get_certainty()
                    
                    # Color: blend based on certainty (red=uncertain, green=certain)
                    if is_valuable:
                        # Valuable tracks: color intensity based on certainty
                        color_scale = int(certainty * 255)
                        track_color = (track.color[0] * certainty, track.color[1] * certainty, track.color[2] * certainty)
                        track_color = tuple(map(int, track_color))
                    else:
                        # Noise tracks: gray
                        track_color = (100, 100, 100)
                    
                    # Draw track history trail (thickness based on certainty)
                    if len(track.position_history) > 1:
                        # position_history stores (pos, frame_id) tuples
                        points = [tuple(map(int, pos_frame[0])) for pos_frame in track.position_history]
                        line_thickness = max(1, int(certainty * 3)) if is_valuable else 1
                        for i in range(len(points) - 1):
                            cv.line(vis_frame, points[i], points[i + 1], track_color, line_thickness)
                    
                    # Draw current position (size based on certainty)
                    pos = tuple(map(int, track.position))
                    circle_size = int(4 + certainty * 6) if is_valuable else 3
                    cv.circle(vis_frame, pos, circle_size, track_color, -1)
                    
                    # Outer ring color indicates certainty level
                    ring_color = (0, int(certainty * 255), int((1-certainty) * 255)) if is_valuable else (100, 100, 100)
                    cv.circle(vis_frame, pos, circle_size + 2, ring_color, 2 if is_valuable else 1)
                    
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
                colored_diff = cv.applyColorMap(diff_map, cv.COLORMAP_JET)
                cv.imshow("Diff Map (Colored)", colored_diff)
                
                overlay = cv.addWeighted(frame, 0.3, colored_diff, 0.7, 0)
                cv.imshow("Overlay", overlay)
            
            # Statistics
            mean_diff = np.mean(diff_map)
            max_diff = np.max(diff_map)
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
            if 'diff_map' in locals():
                filename = f"diff_map_{saved_count:03d}.png"
                cv.imwrite(filename, diff_map)
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
            objects.clear()
            prev_blobs = []
            unmatched_history.clear()
            unmatched_tracks_history.clear()
            Track.next_id = 0
            Object.next_id = 0
            print("Reset - cleared previous frame, all tracks and objects")
    
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
