# VideoProcessor/video_analyzer.py
"""
Video file analysis module for extracting basic video properties
"""

import os
import cv2
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime

from .utils import log_time, safe_float_conversion, safe_int_conversion, format_error_message


class VideoAnalyzer:
    """
    Handles video file analysis and property extraction
    """
    
    def __init__(self):
        """Initialize VideoAnalyzer"""
        pass
    
    def analyze_video_file(self, video_path: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Get detailed information about a video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            (success, video_info_dict, error_message)
        """
        try:
            log_time(f"Analyzing video file: {video_path}")
            
            # Check if file exists
            if not os.path.exists(video_path):
                return False, None, f"Video file not found: {video_path}"
            
            # Get file size
            file_size_bytes = os.path.getsize(video_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            log_time(f"File exists - Size: {file_size_mb:.2f} MB")
            
            # Open video with OpenCV
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return False, None, f"Cannot open video with OpenCV: {video_path}"
            
            # Extract video properties
            fps = safe_float_conversion(cap.get(cv2.CAP_PROP_FPS))
            frame_count = safe_int_conversion(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = safe_int_conversion(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = safe_int_conversion(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate duration
            duration = frame_count / fps if fps > 0 else 0
            
            # Get codec information
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            codec = self._fourcc_to_string(fourcc) if fourcc else "Unknown"
            
            # Release the video capture
            cap.release()
            
            # Validate extracted properties
            if not self._validate_video_properties(width, height, fps, frame_count, duration):
                return False, None, "Invalid video properties detected - video may be corrupted"
            
            # Create video info dictionary
            video_info = {
                # Basic file info
                'filepath': os.path.abspath(video_path),
                'filename': os.path.basename(video_path),
                'file_size_bytes': file_size_bytes,
                'file_size_mb': file_size_mb,
                
                # Video properties
                'width': width,
                'height': height,
                'fps': fps,
                'duration': duration,  # Keep this for backward compatibility
                'duration_seconds': duration,  # Database field name
                'total_frames': frame_count,
                'frames': frame_count,  # Keep this for backward compatibility
                'size_mb': file_size_mb,  # Keep this for backward compatibility
                
                # Additional metadata
                'codec': codec,
                'aspect_ratio': width / height if height > 0 else 0,
                'resolution': f"{width}x{height}",
                'bitrate_estimate': (file_size_bytes * 8) / duration if duration > 0 else 0,
                
                # Analysis metadata
                'analyzed_at': datetime.now().isoformat(),
                'analyzer_version': '1.0'
            }
            
            # Log results
            log_time(f"Video analysis completed successfully")
            self._log_video_properties(video_info)
            
            return True, video_info, ""
            
        except Exception as e:
            error_msg = format_error_message("Video analysis", e)
            return False, None, error_msg
    
    def _fourcc_to_string(self, fourcc: float) -> str:
        """
        Convert FourCC code to readable string
        
        Args:
            fourcc: FourCC code from OpenCV
            
        Returns:
            Readable codec string
        """
        try:
            fourcc_int = int(fourcc)
            return "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        except:
            return "Unknown"
    
    def _validate_video_properties(self, width: int, height: int, fps: float, 
                                 frame_count: int, duration: float) -> bool:
        """
        Validate that extracted video properties are reasonable
        
        Args:
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            frame_count: Total number of frames
            duration: Video duration in seconds
            
        Returns:
            True if properties are valid, False otherwise
        """
        # Check minimum dimensions
        if width < 1 or height < 1:
            return False
        
        # Check reasonable dimensions (not too large)
        if width > 10000 or height > 10000:
            return False
        
        # Check FPS range
        if fps <= 0 or fps > 240:  # Reasonable FPS range
            return False
        
        # Check frame count
        if frame_count <= 0:
            return False
        
        # Check duration
        if duration <= 0 or duration > 86400:  # Max 24 hours
            return False
        
        # Check consistency between fps, frame_count, and duration
        expected_duration = frame_count / fps
        duration_diff = abs(duration - expected_duration)
        
        # Allow 10% difference for rounding errors
        if duration_diff > (expected_duration * 0.1):
            return False
        
        return True
    
    def _log_video_properties(self, video_info: Dict[str, Any]) -> None:
        """
        Log video properties in a formatted way
        
        Args:
            video_info: Video information dictionary
        """
        print(f"\n📊 Video Properties:")
        print(f"   Resolution: {video_info['width']} x {video_info['height']}")
        print(f"   FPS: {video_info['fps']:.2f}")
        print(f"   Duration: {video_info['duration']:.2f} seconds")
        print(f"   Total Frames: {video_info['total_frames']:,}")
        print(f"   File Size: {video_info['file_size_mb']:.2f} MB")
        print(f"   Codec: {video_info['codec']}")
        print(f"   Aspect Ratio: {video_info['aspect_ratio']:.2f}")
        
        if video_info['bitrate_estimate'] > 0:
            bitrate_kbps = video_info['bitrate_estimate'] / 1000
            print(f"   Estimated Bitrate: {bitrate_kbps:.0f} kbps")
    
    def get_video_frame_at_time(self, video_path: str, timestamp: float) -> Tuple[bool, Optional[Any], str]:
        """
        Extract a single frame from video at specified timestamp
        
        Args:
            video_path: Path to video file
            timestamp: Time in seconds to extract frame
            
        Returns:
            (success, frame_data, error_message)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return False, None, f"Cannot open video: {video_path}"
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Validate timestamp
            if timestamp < 0 or timestamp > duration:
                cap.release()
                return False, None, f"Timestamp {timestamp:.2f}s out of range (0-{duration:.2f}s)"
            
            # Calculate frame number
            frame_number = int(timestamp * fps)
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return False, None, f"Failed to read frame at timestamp {timestamp:.2f}s"
            
            # Convert BGR to RGB for consistency
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            return True, frame_rgb, ""
            
        except Exception as e:
            error_msg = format_error_message("Frame extraction", e)
            return False, None, error_msg
    
    def validate_video_for_processing(self, video_path: str) -> Tuple[bool, str]:
        """
        Validate if video is suitable for processing
        
        Args:
            video_path: Path to video file
            
        Returns:
            (is_suitable, error_message)
        """
        try:
            # Basic file validation
            if not os.path.exists(video_path):
                return False, f"Video file not found: {video_path}"
            
            # Analyze video
            success, video_info, error = self.analyze_video_file(video_path)
            
            if not success:
                return False, f"Video analysis failed: {error}"
            
            # Check if video meets processing requirements
            if video_info['duration'] < 1.0:
                return False, "Video too short (minimum 1 second required)"
            
            if video_info['duration'] > 7200:  # 2 hours
                return False, "Video too long (maximum 2 hours supported)"
            
            if video_info['width'] < 100 or video_info['height'] < 100:
                return False, "Video resolution too low (minimum 100x100 required)"
            
            if video_info['fps'] < 1:
                return False, "Invalid frame rate"
            
            if video_info['total_frames'] < 10:
                return False, "Video has too few frames for analysis"
            
            return True, ""
            
        except Exception as e:
            error_msg = format_error_message("Video validation", e)
            return False, error_msg


# Export the VideoAnalyzer class
__all__ = ['VideoAnalyzer']