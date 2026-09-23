"""SparseTrack: whole-movie germination and tube-length analysis for sparse fields.

The project's movies are x264 exports with a keyframe every 12 frames at the
encoder's quality floor, so frames between keyframes are near-copies of the last
keyframe. Every stage here therefore reads keyframes only and works on
registered averages of consecutive keyframes ("bins").
"""

__version__ = "0.2.0"
