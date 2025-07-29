# VideoProcessor/utils.py
"""
Utility functions for video processing operations
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path


def log_time(message: str) -> None:
    """Print message with timestamp for performance tracking"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")


def validate_video_file(video_path: str) -> Tuple[bool, str]:
    """
    Validate if video file exists and is accessible
    
    Args:
        video_path: Path to video file
        
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(video_path, str) or not video_path.strip():
        return False, "Video path must be a non-empty string"
    
    if not os.path.exists(video_path):
        return False, f"Video file not found: {video_path}"
    
    if not os.path.isfile(video_path):
        return False, f"Path is not a file: {video_path}"
    
    # Check file extension
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    file_ext = Path(video_path).suffix.lower()
    
    if file_ext not in valid_extensions:
        return False, f"Unsupported video format: {file_ext}. Supported: {valid_extensions}"
    
    # Check file size (basic check)
    try:
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            return False, "Video file is empty"
        if file_size > 10 * 1024 * 1024 * 1024:  # 10GB limit
            return False, "Video file too large (>10GB)"
    except OSError as e:
        return False, f"Cannot access video file: {str(e)}"
    
    return True, ""


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def calculate_processing_speed(video_duration: float, processing_time: float) -> str:
    """
    Calculate and format processing speed relative to real-time
    
    Args:
        video_duration: Duration of video in seconds
        processing_time: Time taken to process in seconds
        
    Returns:
        Formatted speed string
    """
    if processing_time <= 0:
        return "∞x real-time"
    
    speed_ratio = video_duration / processing_time
    
    if speed_ratio >= 1:
        return f"{speed_ratio:.1f}x real-time"
    else:
        return f"{1/speed_ratio:.1f}x slower than real-time"


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float with fallback
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int_conversion(value: Any, default: int = 0) -> int:
    """
    Safely convert value to int with fallback
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Integer value or default
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def create_processing_summary(video_info: Dict[str, Any], 
                            scenes_count: int,
                            keyframes_count: int, 
                            analysis_count: int,
                            total_processing_time: float) -> Dict[str, Any]:
    """
    Create a comprehensive summary of video processing results
    
    Args:
        video_info: Basic video information
        scenes_count: Number of scenes detected
        keyframes_count: Number of keyframes extracted
        analysis_count: Number of analysis records created
        total_processing_time: Total processing time in seconds
        
    Returns:
        Processing summary dictionary
    """
    video_duration = video_info.get('duration', 0)
    
    summary = {
        'video_file': video_info.get('filename', 'Unknown'),
        'video_duration': video_duration,
        'video_duration_formatted': format_duration(video_duration),
        'resolution': f"{video_info.get('width', 0)}x{video_info.get('height', 0)}",
        'fps': video_info.get('fps', 0),
        'file_size_mb': video_info.get('size_mb', 0),
        
        'processing_results': {
            'scenes_detected': scenes_count,
            'keyframes_extracted': keyframes_count,
            'analysis_records': analysis_count,
            'avg_scene_duration': video_duration / scenes_count if scenes_count > 0 else 0,
            'keyframes_per_scene': keyframes_count / scenes_count if scenes_count > 0 else 0
        },
        
        'performance_metrics': {
            'total_processing_time': total_processing_time,
            'processing_time_formatted': format_duration(total_processing_time),
            'processing_speed': calculate_processing_speed(video_duration, total_processing_time),
            'time_per_scene': total_processing_time / scenes_count if scenes_count > 0 else 0,
            'time_per_keyframe': total_processing_time / keyframes_count if keyframes_count > 0 else 0
        },
        
        'timestamp': datetime.now().isoformat()
    }
    
    return summary


def validate_processing_config(config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate VideoProcessor configuration parameters
    
    Args:
        config: Configuration dictionary
        
    Returns:
        (is_valid, error_message)
    """
    # Validate threshold
    threshold = config.get('threshold', 27.0)
    if not isinstance(threshold, (int, float)) or threshold <= 0 or threshold > 100:
        return False, "Threshold must be a number between 0 and 100"
    
    # Validate model name
    model = config.get('model', 'ViT-B-32')
    valid_models = ['ViT-B-32', 'ViT-B-16', 'ViT-L-14']  # Add more as needed
    if model not in valid_models:
        return False, f"Model must be one of: {valid_models}"
    
    # Validate confidence threshold
    confidence_threshold = config.get('confidence_threshold', 0.15)
    if not isinstance(confidence_threshold, (int, float)) or confidence_threshold < 0 or confidence_threshold > 1:
        return False, "Confidence threshold must be between 0.0 and 1.0"
    
    # Validate frame position
    frame_position = config.get('frame_position', 'middle')
    valid_positions = ['start', 'middle', 'end', 'all']
    if frame_position not in valid_positions:
        return False, f"Frame position must be one of: {valid_positions}"
    
    return True, ""


def format_error_message(operation: str, error: Exception) -> str:
    """
    Format error message consistently
    
    Args:
        operation: Name of the operation that failed
        error: Exception that occurred
        
    Returns:
        Formatted error message
    """
    return f"{operation} failed: {type(error).__name__}: {str(error)}"


def print_processing_summary(summary: Dict[str, Any]) -> None:
    """
    Print a formatted processing summary
    
    Args:
        summary: Processing summary from create_processing_summary()
    """
    print("\n" + "="*60)
    print("🎬 VIDEO PROCESSING SUMMARY")
    print("="*60)
    
    print(f"📄 File: {summary['video_file']}")
    print(f"⏱️  Duration: {summary['video_duration_formatted']}")
    print(f"📐 Resolution: {summary['resolution']}")
    print(f"🎯 FPS: {summary['fps']:.2f}")
    print(f"💾 Size: {summary['file_size_mb']:.2f} MB")
    
    print(f"\n📊 Processing Results:")
    results = summary['processing_results']
    print(f"   • Scenes detected: {results['scenes_detected']}")
    print(f"   • Keyframes extracted: {results['keyframes_extracted']}")
    print(f"   • Analysis records: {results['analysis_records']}")
    print(f"   • Avg scene duration: {results['avg_scene_duration']:.2f}s")
    
    print(f"\n⚡ Performance:")
    perf = summary['performance_metrics']
    print(f"   • Processing time: {perf['processing_time_formatted']}")
    print(f"   • Speed: {perf['processing_speed']}")
    print(f"   • Time per scene: {perf['time_per_scene']:.2f}s")
    
    print("="*60)


# Export all utility functions
__all__ = [
    'log_time',
    'validate_video_file', 
    'format_duration',
    'calculate_processing_speed',
    'safe_float_conversion',
    'safe_int_conversion',
    'create_processing_summary',
    'validate_processing_config',
    'format_error_message',
    'print_processing_summary'
]