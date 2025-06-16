# Smart Media Analyzer - Frugal Architecture

> **Local-first Media AI for M1 Pro 16GB** - Open-source video content understanding and automatic metadata generation

## Overview

An intelligent media archive system that runs entirely personal laptop, using open-source AI models to analyze video content, detect scenes, and generate searchable metadata. Built with a "frugal architecture" approach - local-first, cost-conscious, and educational.

## Tech Stack

- **Python 3.11** (UV package management)
- **Scene Detection**: PySceneDetect + OpenCV
- **Vision-Language**: OpenCLIP (planned)
- **Audio Processing**: faster-whisper (planned)

## Project Phases

### Phase 1: Video Processing Foundation (Current)
**Goal**: Basic video scene detection and analysis

**Key Features**:
- Automatic scene boundary detection
- Video metadata extraction (resolution, fps, duration)
- Performance optimization for M1 Mac
- Threshold sensitivity analysis
- Visual scene change inspection

---

### Phase 2: Intelligence Layer (Next)
**Goal**: Semantic understanding of video content

**Planned Features**:
- Visual content description using OpenCLIP
- Natural language search ("find videos with people outdoors")
- Content similarity matching
- Automatic tagging and categorization

---

### Phase 3: Advanced Analytics (Future)
**Goal**: Cross-modal content intelligence

**Planned Features**:
- Audio transcription and voice analysis
- Face recognition and speaker identification
- Story detection
- Duplicate content finder
- Automatic video summaries

### Interactive Exploration
**[Video Processing Exploration Notebook](notebooks/01_video_processing_exploration.ipynb)**

Jupyter notebook with:
- Step-by-step video analysis
- Performance benchmarking
- Threshold sensitivity testing
- Visual scene change inspection

## License

MIT License - Built for learning and experimentation