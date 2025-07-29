# database/utils.py
"""
Utility functions for SQLAlchemy ORM operations and data validation
"""

import os
import json
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

# Import ORM models for type hints and validation
from .schemas import Video, Scene, Keyframe, ClipAnalysis, EmbeddingMetadata


def create_directories(db_path: str) -> Tuple[bool, str]:
    """
    Create database directory if it doesn't exist
    
    Returns:
        (success, error_message)
    """
    try:
        directory = os.path.dirname(db_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        return True, ""
    except Exception as e:
        return False, f"Failed to create directory: {str(e)}"


# ============================================================================
# ORM MODEL VALIDATION FUNCTIONS
# ============================================================================

def validate_video_data(video_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate video data for ORM Video model
    
    Args:
        video_data: Dictionary with video metadata
        
    Returns:
        (is_valid, error_message)
    """
    required_fields = [
        'filepath', 'filename', 'duration_seconds', 'fps', 
        'width', 'height', 'total_frames', 'file_size_mb'
    ]
    
    for field in required_fields:
        if field not in video_data:
            return False, f"Missing required field: {field}"
    
    # Type and value validations
    if not isinstance(video_data['filepath'], str) or not video_data['filepath'].strip():
        return False, "filepath must be a non-empty string"
    
    if not isinstance(video_data['filename'], str) or not video_data['filename'].strip():
        return False, "filename must be a non-empty string"
    
    # Check file path length (SQLAlchemy String(500) constraint)
    if len(video_data['filepath']) > 500:
        return False, "filepath too long (max 500 characters)"
    
    if len(video_data['filename']) > 255:
        return False, "filename too long (max 255 characters)"
    
    numeric_fields = ['duration_seconds', 'fps', 'file_size_mb']
    for field in numeric_fields:
        if not isinstance(video_data[field], (int, float)) or video_data[field] <= 0:
            return False, f"{field} must be a positive number"
    
    integer_fields = ['width', 'height', 'total_frames']
    for field in integer_fields:
        if not isinstance(video_data[field], int) or video_data[field] <= 0:
            return False, f"{field} must be a positive integer"
    
    # Validate status if provided
    if 'status' in video_data:
        valid_statuses = ['pending', 'processing', 'completed', 'error']
        if video_data['status'] not in valid_statuses:
            return False, f"status must be one of: {valid_statuses}"
    
    return True, ""


def validate_scene_data(scene_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate scene data for ORM Scene model
    """
    required_fields = [
        'video_id', 'scene_number', 'start_time', 'end_time', 
        'duration', 'detection_method', 'threshold_used'
    ]
    
    for field in required_fields:
        if field not in scene_data:
            return False, f"Missing required field: {field}"
    
    # Type validations
    if not isinstance(scene_data['video_id'], int) or scene_data['video_id'] <= 0:
        return False, "video_id must be a positive integer"
    
    if not isinstance(scene_data['scene_number'], int) or scene_data['scene_number'] <= 0:
        return False, "scene_number must be a positive integer"
    
    # Time validations
    numeric_fields = ['start_time', 'end_time', 'duration', 'threshold_used']
    for field in numeric_fields:
        if not isinstance(scene_data[field], (int, float)) or scene_data[field] < 0:
            return False, f"{field} must be a non-negative number"
    
    # Validate time relationships
    if scene_data['start_time'] >= scene_data['end_time']:
        return False, "start_time must be less than end_time"
    
    expected_duration = scene_data['end_time'] - scene_data['start_time']
    if abs(scene_data['duration'] - expected_duration) > 0.1:  # Allow small floating point errors
        return False, "duration doesn't match end_time - start_time"
    
    # String field validations
    if not isinstance(scene_data['detection_method'], str) or not scene_data['detection_method'].strip():
        return False, "detection_method must be a non-empty string"
    
    if len(scene_data['detection_method']) > 50:
        return False, "detection_method too long (max 50 characters)"
    
    return True, ""


def validate_keyframe_data(keyframe_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate keyframe data for ORM Keyframe model
    """
    required_fields = [
        'scene_id', 'frame_index', 'timestamp', 'frame_number', 
        'frame_position', 'extraction_method'
    ]
    
    for field in required_fields:
        if field not in keyframe_data:
            return False, f"Missing required field: {field}"
    
    # Type validations
    if not isinstance(keyframe_data['scene_id'], int) or keyframe_data['scene_id'] <= 0:
        return False, "scene_id must be a positive integer"
    
    if not isinstance(keyframe_data['frame_index'], int) or keyframe_data['frame_index'] <= 0:
        return False, "frame_index must be positive"
    
    if not isinstance(keyframe_data['frame_number'], int) or keyframe_data['frame_number'] < 0:
        return False, "frame_number must be non-negative"
    
    if not isinstance(keyframe_data['timestamp'], (int, float)) or keyframe_data['timestamp'] < 0:
        return False, "timestamp must be non-negative"
    
    # String validations
    valid_positions = ['start', 'middle', 'end', 'all']
    if keyframe_data['frame_position'] not in valid_positions:
        return False, f"frame_position must be one of: {valid_positions}"
    
    if not isinstance(keyframe_data['extraction_method'], str) or not keyframe_data['extraction_method'].strip():
        return False, "extraction_method must be a non-empty string"
    
    if len(keyframe_data['extraction_method']) > 20:
        return False, "extraction_method too long (max 20 characters)"
    
    # Optional field validations
    if 'frame_path' in keyframe_data and keyframe_data['frame_path']:
        if len(keyframe_data['frame_path']) > 500:
            return False, "frame_path too long (max 500 characters)"
    
    if 'embedding_id' in keyframe_data and keyframe_data['embedding_id'] is not None:
        if not isinstance(keyframe_data['embedding_id'], int) or keyframe_data['embedding_id'] < 0:
            return False, "embedding_id must be non-negative integer"
    
    return True, ""


def validate_analysis_data(analysis_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate CLIP analysis data for ORM ClipAnalysis model
    """
    required_fields = ['keyframe_id', 'category', 'label', 'confidence']
    
    for field in required_fields:
        if field not in analysis_data:
            return False, f"Missing required field: {field}"
    
    # Type validations
    if not isinstance(analysis_data['keyframe_id'], int) or analysis_data['keyframe_id'] <= 0:
        return False, "keyframe_id must be a positive integer"
    
    if not (0.0 <= analysis_data['confidence'] <= 1.0):
        return False, "confidence must be between 0.0 and 1.0"
    
    # String validations
    if not isinstance(analysis_data['category'], str) or not analysis_data['category'].strip():
        return False, "category cannot be empty"
    
    if not isinstance(analysis_data['label'], str) or not analysis_data['label'].strip():
        return False, "label cannot be empty"
    
    if len(analysis_data['category']) > 100:
        return False, "category too long (max 100 characters)"
    
    if len(analysis_data['label']) > 200:
        return False, "label too long (max 200 characters)"
    
    # Optional field validations
    if 'analysis_model' in analysis_data:
        if len(analysis_data['analysis_model']) > 50:
            return False, "analysis_model too long (max 50 characters)"
    
    if 'analysis_version' in analysis_data:
        if len(analysis_data['analysis_version']) > 20:
            return False, "analysis_version too long (max 20 characters)"
    
    return True, ""


def validate_embedding_metadata(embedding_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate embedding metadata for ORM EmbeddingMetadata model
    """
    required_fields = ['keyframe_id', 'faiss_index', 'embedding_model', 'embedding_dim']
    
    for field in required_fields:
        if field not in embedding_data:
            return False, f"Missing required field: {field}"
    
    # Type validations
    if not isinstance(embedding_data['keyframe_id'], int) or embedding_data['keyframe_id'] <= 0:
        return False, "keyframe_id must be a positive integer"
    
    if not isinstance(embedding_data['faiss_index'], int) or embedding_data['faiss_index'] < 0:
        return False, "faiss_index must be non-negative integer"
    
    if not isinstance(embedding_data['embedding_dim'], int) or embedding_data['embedding_dim'] <= 0:
        return False, "embedding_dim must be positive integer"
    
    # String validations
    if not isinstance(embedding_data['embedding_model'], str) or not embedding_data['embedding_model'].strip():
        return False, "embedding_model cannot be empty"
    
    if len(embedding_data['embedding_model']) > 50:
        return False, "embedding_model too long (max 50 characters)"
    
    return True, ""


# ============================================================================
# ORM OBJECT CONVERSION UTILITIES
# ============================================================================

def video_to_dict(video: Video) -> Dict[str, Any]:
    """
    Convert Video ORM object to dictionary
    
    Args:
        video: Video ORM instance
        
    Returns:
        Dictionary representation
    """
    return {
        'id': video.id,
        'filepath': video.filepath,
        'filename': video.filename,
        'duration_seconds': video.duration_seconds,
        'fps': video.fps,
        'width': video.width,
        'height': video.height,
        'total_frames': video.total_frames,
        'file_size_mb': video.file_size_mb,
        'status': video.status,
        'created_at': video.created_at,
        'processed_at': video.processed_at
    }


def scene_to_dict(scene: Scene) -> Dict[str, Any]:
    """
    Convert Scene ORM object to dictionary
    """
    return {
        'id': scene.id,
        'video_id': scene.video_id,
        'scene_number': scene.scene_number,
        'start_time': scene.start_time,
        'end_time': scene.end_time,
        'duration': scene.duration,
        'detection_method': scene.detection_method,
        'threshold_used': scene.threshold_used,
        'created_at': scene.created_at
    }


def keyframe_to_dict(keyframe: Keyframe) -> Dict[str, Any]:
    """
    Convert Keyframe ORM object to dictionary
    """
    return {
        'id': keyframe.id,
        'scene_id': keyframe.scene_id,
        'frame_index': keyframe.frame_index,
        'timestamp': keyframe.timestamp,
        'frame_number': keyframe.frame_number,
        'frame_position': keyframe.frame_position,
        'embedding_id': keyframe.embedding_id,
        'frame_path': keyframe.frame_path,
        'extraction_method': keyframe.extraction_method,
        'created_at': keyframe.created_at
    }


def analysis_to_dict(analysis: ClipAnalysis) -> Dict[str, Any]:
    """
    Convert ClipAnalysis ORM object to dictionary
    """
    return {
        'id': analysis.id,
        'keyframe_id': analysis.keyframe_id,
        'category': analysis.category,
        'label': analysis.label,
        'confidence': analysis.confidence,
        'is_high_confidence': analysis.is_high_confidence,
        'analysis_model': analysis.analysis_model,
        'analysis_version': analysis.analysis_version,
        'created_at': analysis.created_at
    }


def embedding_to_dict(embedding: EmbeddingMetadata) -> Dict[str, Any]:
    """
    Convert EmbeddingMetadata ORM object to dictionary
    """
    return {
        'id': embedding.id,
        'keyframe_id': embedding.keyframe_id,
        'faiss_index': embedding.faiss_index,
        'embedding_model': embedding.embedding_model,
        'embedding_dim': embedding.embedding_dim,
        'created_at': embedding.created_at
    }


# ============================================================================
# NOTEBOOK INTEGRATION UTILITIES
# ============================================================================

def convert_pyscenedetect_to_orm(scenes_list: List, video_id: int, 
                                threshold: float = 27.0, 
                                method: str = 'ContentDetector') -> List[Dict[str, Any]]:
    """
    Convert PySceneDetect scene list to ORM-compatible format
    
    Args:
        scenes_list: List of scenes from PySceneDetect
        video_id: Database video ID
        threshold: Detection threshold used
        method: Detection method used
        
    Returns:
        List of scene dictionaries ready for ORM
    """
    scenes_data = []
    for i, scene in enumerate(scenes_list):
        scene_data = {
            'video_id': video_id,
            'scene_number': i + 1,
            'start_time': scene[0].get_seconds(),
            'end_time': scene[1].get_seconds(),
            'duration': scene[1].get_seconds() - scene[0].get_seconds(),
            'detection_method': method,
            'threshold_used': threshold
        }
        scenes_data.append(scene_data)
    
    return scenes_data


def convert_keyframes_to_orm(keyframes_list: List[Dict[str, Any]], 
                           scene_id_mapping: Dict[int, int]) -> List[Dict[str, Any]]:
    """
    Convert extracted keyframes to ORM-compatible format
    
    Args:
        keyframes_list: List from extract_keyframes_from_scenes()
        scene_id_mapping: Mapping from scene_number to scene_id
        
    Returns:
        List of keyframe dictionaries ready for ORM
    """
    keyframes_data = []
    for keyframe in keyframes_list:
        scene_number = keyframe.get('scene_number', 1)
        scene_id = scene_id_mapping.get(scene_number)
        
        if scene_id is None:
            continue  # Skip if scene_id not found
        
        keyframe_data = {
            'scene_id': scene_id,
            'frame_index': keyframe.get('frame_index', 1),
            'timestamp': keyframe['timestamp'],
            'frame_number': keyframe['frame_number'],
            'frame_position': keyframe.get('frame_position', 'middle'),
            'frame_path': keyframe.get('frame_path'),
            'extraction_method': keyframe.get('extraction_method', 'middle')
        }
        keyframes_data.append(keyframe_data)
    
    return keyframes_data


def convert_clip_analysis_to_orm(analysis_results: List[Dict[str, Any]], 
                                keyframe_id_mapping: Dict[int, int]) -> List[Dict[str, Any]]:
    """
    Convert CLIP analysis results to ORM-compatible format
    
    Args:
        analysis_results: Results from analyze_frame_with_fallback()
        keyframe_id_mapping: Mapping from keyframe_index to keyframe_id
        
    Returns:
        List of analysis dictionaries ready for ORM
    """
    analysis_data = []
    
    for keyframe_index, analysis in enumerate(analysis_results):
        keyframe_id = keyframe_id_mapping.get(keyframe_index)
        
        if keyframe_id is None:
            continue  # Skip if keyframe_id not found
        
        if isinstance(analysis, dict):
            # Handle comprehensive analysis format
            for category, items in analysis.items():
                if category in ['analysis_metadata', 'fallback_classification']:
                    continue
                
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict) and 'label' in item and 'confidence' in item:
                            analysis_item = {
                                'keyframe_id': keyframe_id,
                                'category': category,
                                'label': item['label'],
                                'confidence': item['confidence'],
                                'analysis_model': 'ViT-B-32',
                                'analysis_version': '1.0'
                            }
                            analysis_data.append(analysis_item)
    
    return analysis_data


# ============================================================================
# FAISS UTILITIES
# ============================================================================

def save_embedding_mapping(mapping_path: str, mapping_data: Dict[int, int]) -> Tuple[bool, str]:
    """
    Save FAISS index to keyframe_id mapping to JSON file
    
    Args:
        mapping_path: Path to save mapping file
        mapping_data: Dict mapping faiss_index -> keyframe_id
        
    Returns:
        (success, error_message)
    """
    try:
        # Convert keys to strings for JSON serialization
        string_mapping = {str(k): v for k, v in mapping_data.items()}
        
        with open(mapping_path, 'w') as f:
            json.dump(string_mapping, f, indent=2)
        
        return True, ""
    except Exception as e:
        return False, f"Failed to save embedding mapping: {str(e)}"


def load_embedding_mapping(mapping_path: str) -> Tuple[bool, Dict[int, int], str]:
    """
    Load FAISS index to keyframe_id mapping from JSON file
    
    Returns:
        (success, mapping_data, error_message)
    """
    try:
        if not os.path.exists(mapping_path):
            return True, {}, ""  # Empty mapping is valid for new database
        
        with open(mapping_path, 'r') as f:
            string_mapping = json.load(f)
        
        # Convert keys back to integers
        mapping_data = {int(k): v for k, v in string_mapping.items()}
        
        return True, mapping_data, ""
    except Exception as e:
        return False, {}, f"Failed to load embedding mapping: {str(e)}"


# ============================================================================
# SEARCH RESULT FORMATTING
# ============================================================================

def format_search_results(results: List[Dict[str, Any]], include_metadata: bool = True) -> Dict[str, Any]:
    """
    Format search results for consistent return structure
    
    Args:
        results: List of result dictionaries
        include_metadata: Whether to include search metadata
        
    Returns:
        Formatted results dictionary
    """
    formatted = {
        'results': results,
        'count': len(results)
    }
    
    if include_metadata:
        formatted['metadata'] = {
            'search_timestamp': datetime.now().isoformat(),
            'result_count': len(results),
            'has_results': len(results) > 0
        }
    
    return formatted


def format_video_summary_for_display(summary: Dict[str, Any]) -> str:
    """
    Format video summary for notebook display
    
    Args:
        summary: Video summary dictionary from get_video_summary()
        
    Returns:
        Formatted string for display
    """
    return f"""
🎬 Video Summary: {summary.get('filename', 'Unknown')}
   📄 File: {summary.get('filepath', 'Unknown')}
   ⏱️  Duration: {summary.get('duration_seconds', 0):.1f}s
   📐 Resolution: {summary.get('width', 0)}x{summary.get('height', 0)}
   🎯 FPS: {summary.get('fps', 0):.2f}
   💾 Size: {summary.get('file_size_mb', 0):.2f}MB
   📊 Status: {summary.get('status', 'unknown')}
   
📈 Analysis Counts:
   • Scenes: {summary.get('scene_count', 0)}
   • Keyframes: {summary.get('keyframe_count', 0)}
   • Analysis records: {summary.get('analysis_count', 0)}
   • Embeddings: {summary.get('embedding_count', 0)}
   
🕒 Timestamps:
   • Created: {summary.get('created_at', 'Unknown')}
   • Processed: {summary.get('processed_at', 'Not processed')}
    """.strip()


# ============================================================================
# FILE AND PATH UTILITIES
# ============================================================================

def get_file_extension(filepath: str) -> str:
    """
    Get file extension from filepath
    """
    return os.path.splitext(filepath)[1].lower()


def is_video_file(filepath: str) -> bool:
    """
    Check if file is a supported video format
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    return get_file_extension(filepath) in video_extensions


def generate_frame_filename(video_filename: str, scene_number: int, frame_index: int) -> str:
    """
    Generate consistent filename for extracted frames
    
    Args:
        video_filename: Original video filename
        scene_number: Scene number
        frame_index: Frame index within scene
        
    Returns:
        Generated filename
    """
    base_name = os.path.splitext(video_filename)[0]
    return f"{base_name}_scene{scene_number:03d}_frame{frame_index:02d}.jpg"


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float with fallback
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int_conversion(value: Any, default: int = 0) -> int:
    """
    Safely convert value to int with fallback
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def calculate_confidence_category(confidence: float) -> str:
    """
    Categorize confidence scores for analysis
    """
    if confidence >= 0.8:
        return "very_high"
    elif confidence >= 0.6:
        return "high" 
    elif confidence >= 0.4:
        return "medium"
    elif confidence >= 0.2:
        return "low"
    else:
        return "very_low"


# Export all utility functions
__all__ = [
    # Directory utilities
    'create_directories',
    
    # Validation functions
    'validate_video_data',
    'validate_scene_data', 
    'validate_keyframe_data',
    'validate_analysis_data',
    'validate_embedding_metadata',
    
    # ORM conversion utilities
    'video_to_dict',
    'scene_to_dict',
    'keyframe_to_dict',
    'analysis_to_dict',
    'embedding_to_dict',
    
    # Notebook integration
    'convert_pyscenedetect_to_orm',
    'convert_keyframes_to_orm',
    'convert_clip_analysis_to_orm',
    
    # FAISS utilities
    'save_embedding_mapping',
    'load_embedding_mapping',
    
    # Formatting utilities
    'format_search_results',
    'format_video_summary_for_display',
    
    # File utilities
    'get_file_extension',
    'is_video_file',
    'generate_frame_filename',
    'safe_float_conversion',
    'safe_int_conversion',
    'calculate_confidence_category'
]