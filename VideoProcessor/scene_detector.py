# VideoProcessor/scene_detector.py
"""
Scene detection module using PySceneDetect
"""

import time
from typing import Dict, List, Any, Tuple, Optional

try:
    from scenedetect import detect, ContentDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False

from .utils import log_time, format_error_message, calculate_processing_speed


class SceneDetector:
    """
    Handles video scene detection using PySceneDetect
    """
    
    def __init__(self, threshold: float = 27.0, detection_method: str = 'ContentDetector'):
        """
        Initialize SceneDetector
        
        Args:
            threshold: Detection threshold (default: 27.0)
            detection_method: Detection method to use (default: 'ContentDetector')
        """
        self.threshold = threshold
        self.detection_method = detection_method
        
        if not SCENEDETECT_AVAILABLE:
            raise ImportError("PySceneDetect not available. Install with: pip install scenedetect[opencv]")
    
    def detect_scenes(self, video_path: str, video_duration: float = None) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Detect scenes in video and return detailed analysis
        
        Args:
            video_path: Path to video file
            video_duration: Video duration in seconds (for performance calculation)
            
        Returns:
            (success, scenes_list, error_message)
        """
        try:
            log_time(f"Starting scene detection (threshold={self.threshold})")
            
            start_time = time.time()
            
            # Detect scenes using ContentDetector
            scene_list = detect(video_path, ContentDetector(threshold=self.threshold))
            
            detection_time = time.time() - start_time
            log_time(f"Scene detection completed in {detection_time:.2f}s")
            
            # Analyze results
            total_scenes = len(scene_list)
            log_time(f"Found {total_scenes} scenes")
            
            if total_scenes == 0:
                log_time("⚠️  No scenes detected - video might be too uniform")
                return True, [], "No scenes detected"
            
            # Convert PySceneDetect results to our format
            scenes_data = self._convert_scenes_to_data(scene_list)
            
            # Calculate and log statistics
            self._log_scene_statistics(scenes_data, detection_time, video_duration)
            
            return True, scenes_data, ""
            
        except Exception as e:
            error_msg = format_error_message("Scene detection", e)
            return False, [], error_msg
    
    def _convert_scenes_to_data(self, scene_list: List) -> List[Dict[str, Any]]:
        """
        Convert PySceneDetect scene list to our data format
        
        Args:
            scene_list: PySceneDetect scene list
            
        Returns:
            List of scene dictionaries
        """
        scenes_data = []
        
        for i, scene in enumerate(scene_list):
            start_sec = scene[0].get_seconds()
            end_sec = scene[1].get_seconds()
            duration = end_sec - start_sec
            
            scene_data = {
                'scene_number': i + 1,
                'start_time': start_sec,
                'end_time': end_sec,
                'duration': duration,
                'detection_method': self.detection_method,
                'threshold_used': self.threshold,
                
                # Additional metadata
                'start_frame': scene[0].get_frames(),
                'end_frame': scene[1].get_frames(),
                'frame_count': scene[1].get_frames() - scene[0].get_frames(),
            }
            
            scenes_data.append(scene_data)
        
        return scenes_data
    
    def _log_scene_statistics(self, scenes_data: List[Dict[str, Any]], 
                            detection_time: float, video_duration: float = None) -> None:
        """
        Log detailed scene detection statistics
        
        Args:
            scenes_data: List of scene dictionaries
            detection_time: Time taken for detection
            video_duration: Video duration for speed calculation
        """
        if not scenes_data:
            return
        
        # Calculate statistics
        durations = [scene['duration'] for scene in scenes_data]
        total_scenes = len(scenes_data)
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        # Log scene analysis table
        print(f"\n📊 Scene Analysis:")
        print(f"{'Scene':<8} {'Start':<8} {'End':<8} {'Duration':<10}")
        print("-" * 35)
        
        # Show first 10 scenes to avoid overwhelming output
        scenes_to_show = min(10, total_scenes)
        for i in range(scenes_to_show):
            scene = scenes_data[i]
            print(f"{scene['scene_number']:<8} {scene['start_time']:<8.2f} "
                  f"{scene['end_time']:<8.2f} {scene['duration']:<10.2f}")
        
        if total_scenes > 10:
            print(f"... and {total_scenes - 10} more scenes")
        
        # Log statistics
        print(f"\n📈 Scene Statistics:")
        print(f"   Total scenes: {total_scenes}")
        print(f"   Average duration: {avg_duration:.2f}s")
        print(f"   Shortest scene: {min_duration:.2f}s")
        print(f"   Longest scene: {max_duration:.2f}s")
        
        # Show processing speed if video duration provided
        if video_duration and video_duration > 0:
            speed_info = calculate_processing_speed(video_duration, detection_time)
            print(f"   Processing speed: {speed_info}")
    
    def analyze_scene_distribution(self, scenes_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the distribution of scene lengths and patterns
        
        Args:
            scenes_data: List of scene dictionaries
            
        Returns:
            Scene distribution analysis
        """
        if not scenes_data:
            return {}
        
        durations = [scene['duration'] for scene in scenes_data]
        
        # Basic statistics
        total_scenes = len(scenes_data)
        total_duration = sum(durations)
        avg_duration = total_duration / total_scenes
        
        # Duration categories
        short_scenes = len([d for d in durations if d < 1.0])  # < 1 second
        medium_scenes = len([d for d in durations if 1.0 <= d < 5.0])  # 1-5 seconds
        long_scenes = len([d for d in durations if d >= 5.0])  # >= 5 seconds
        
        # Variation analysis
        import statistics
        duration_std = statistics.stdev(durations) if len(durations) > 1 else 0
        duration_median = statistics.median(durations)
        
        analysis = {
            'total_scenes': total_scenes,
            'total_duration': total_duration,
            'average_duration': avg_duration,
            'median_duration': duration_median,
            'duration_std_dev': duration_std,
            'min_duration': min(durations),
            'max_duration': max(durations),
            
            'scene_categories': {
                'short_scenes': {'count': short_scenes, 'percentage': short_scenes / total_scenes * 100},
                'medium_scenes': {'count': medium_scenes, 'percentage': medium_scenes / total_scenes * 100},
                'long_scenes': {'count': long_scenes, 'percentage': long_scenes / total_scenes * 100}
            },
            
            'consistency_metrics': {
                'coefficient_of_variation': duration_std / avg_duration if avg_duration > 0 else 0,
                'uniformity_score': 1 - (duration_std / avg_duration) if avg_duration > 0 else 0
            }
        }
        
        return analysis
    
    def test_multiple_thresholds(self, video_path: str, 
                                thresholds: List[float] = None) -> Tuple[bool, Dict[str, Any], str]:
        """
        Test scene detection with multiple thresholds
        
        Args:
            video_path: Path to video file
            thresholds: List of thresholds to test (default: [15.0, 20.0, 27.0, 35.0, 45.0])
            
        Returns:
            (success, results_dict, error_message)
        """
        if thresholds is None:
            thresholds = [15.0, 20.0, 27.0, 35.0, 45.0]
        
        try:
            log_time("=== Testing Multiple Thresholds ===")
            
            results = {}
            original_threshold = self.threshold
            
            for threshold in thresholds:
                log_time(f"Testing threshold: {threshold}")
                
                # Update threshold
                self.threshold = threshold
                
                # Detect scenes
                start_time = time.time()
                scene_list = detect(video_path, ContentDetector(threshold=threshold))
                detection_time = time.time() - start_time
                
                # Convert and analyze
                scenes_data = self._convert_scenes_to_data(scene_list)
                analysis = self.analyze_scene_distribution(scenes_data)
                
                results[threshold] = {
                    'scene_count': len(scenes_data),
                    'detection_time': detection_time,
                    'average_duration': analysis.get('average_duration', 0),
                    'scenes_data': scenes_data,
                    'analysis': analysis
                }
            
            # Restore original threshold
            self.threshold = original_threshold
            
            # Log comparison table
            self._log_threshold_comparison(results)
            
            return True, results, ""
            
        except Exception as e:
            # Restore original threshold on error
            self.threshold = original_threshold
            error_msg = format_error_message("Multiple threshold testing", e)
            return False, {}, error_msg
    
    def _log_threshold_comparison(self, results: Dict[str, Any]) -> None:
        """
        Log threshold comparison table
        
        Args:
            results: Results from test_multiple_thresholds
        """
        print(f"\n📊 Threshold Comparison:")
        print(f"{'Threshold':<12} {'Scenes':<8} {'Avg Duration':<12} {'Time':<8}")
        print("-" * 45)
        
        for threshold, data in results.items():
            print(f"{threshold:<12} {data['scene_count']:<8} "
                  f"{data['average_duration']:<12.2f} {data['detection_time']:<8.2f}s")
    
    def get_recommended_threshold(self, video_path: str) -> Tuple[bool, float, str]:
        """
        Analyze video and recommend optimal threshold
        
        Args:
            video_path: Path to video file
            
        Returns:
            (success, recommended_threshold, error_message)
        """
        try:
            # Test multiple thresholds
            success, results, error = self.test_multiple_thresholds(video_path)
            
            if not success:
                return False, 27.0, error
            
            # Analyze results to find optimal threshold
            best_threshold = 27.0
            best_score = 0
            
            for threshold, data in results.items():
                analysis = data['analysis']
                scene_count = data['scene_count']
                
                # Calculate score based on:
                # 1. Reasonable scene count (not too many, not too few)
                # 2. Good scene distribution
                # 3. Consistent scene lengths
                
                if scene_count == 0:
                    score = 0
                else:
                    # Prefer 10-50 scenes for typical videos
                    scene_count_score = 1.0 if 10 <= scene_count <= 50 else 0.5
                    
                    # Prefer uniform scene lengths
                    uniformity_score = analysis.get('consistency_metrics', {}).get('uniformity_score', 0)
                    
                    # Combine scores
                    score = scene_count_score * 0.6 + uniformity_score * 0.4
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            log_time(f"Recommended threshold: {best_threshold} (score: {best_score:.3f})")
            
            return True, best_threshold, ""
            
        except Exception as e:
            error_msg = format_error_message("Threshold recommendation", e)
            return False, 27.0, error_msg


# Export the SceneDetector class
__all__ = ['SceneDetector']