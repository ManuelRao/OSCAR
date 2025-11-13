"""
Multi-object tracking library for video analysis.

Main exports:
- MultiObjectTracker: Main tracking engine
- TrackingConfig: Configuration class
- TrackingVisualizer: Visualization and debugging tools
- Data classes: BlobInfo, Track, Object, ImportanceMap
"""

from .multi_object_tracker import (
    MultiObjectTracker,
    TrackingConfig,
    BlobInfo,
    Track,
    Object,
    ImportanceMap
)

from .visualization import TrackingVisualizer

__all__ = [
    'MultiObjectTracker',
    'TrackingConfig',
    'TrackingVisualizer',
    'BlobInfo',
    'Track',
    'Object',
    'ImportanceMap'
]
