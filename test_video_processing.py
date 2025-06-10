#!/usr/bin/env python3
"""
Test script for video scene detection with time logging
"""
import time
from datetime import datetime

def log_time(message):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

if __name__ == "__main__":
    log_time("=== Starting Video Processing Test ===")
    
    log_time("Importing cv2...")
    start = time.time()
    import cv2
    log_time(f"✅ cv2 imported in {time.time() - start:.2f}s")
    
    log_time("Importing scenedetect...")
    start = time.time()
    from scenedetect import detect, ContentDetector
    log_time(f"✅ scenedetect imported in {time.time() - start:.2f}s")
    
    log_time("Testing OpenCV...")
    start = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] OpenCV version: {cv2.__version__}")
    log_time(f"✅ OpenCV tested in {time.time() - start:.2f}s")
    
    log_time("🎉 All tests completed!")