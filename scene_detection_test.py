#!/usr/bin/env python3
"""
Scene detection test with timing analysis
"""
import time
from datetime import datetime
from scenedetect import detect, ContentDetector

def log_time(message):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

def detect_scenes_in_video(video_path):
    """Detect scenes and print results with timing"""
    log_time(f"Starting scene detection for: {video_path}")
    
    start_time = time.time()
    
    # Detect scenes using ContentDetector
    scene_list = detect(video_path, ContentDetector())
    
    detection_time = time.time() - start_time
    log_time(f"Scene detection completed in {detection_time:.2f}s")
    
    # Print results
    log_time(f"Found {len(scene_list)} scenes:")
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        duration = end_time - start_time
        print(f"  Scene {i+1}: {start_time:.2f}s - {end_time:.2f}s (duration: {duration:.2f}s)")
    
    return scene_list

if __name__ == "__main__":
    video_file = "Samples_Video-Images/Giant_Oarfish.mp4"
    print(f"Video file: {video_file}")
    log_time("=== Starting Scene Detection Test ===")
    log_time("Importing scenedetect...")
    start = time.time()

    scenes = detect_scenes_in_video(video_file)