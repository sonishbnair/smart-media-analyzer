# database/__init__.py
"""
Media Archive Database Module - SQLAlchemy ORM Version

Provides persistent storage for video analysis results including:
- Video metadata and scene detection using SQLAlchemy ORM
- Keyframe extraction and OpenCLIP analysis with relationships
- Vector embeddings with FAISS similarity search
- Batch operations and notebook integration utilities

Usage:
    from database import MediaArchiveDB
    
    # Initialize database (auto-creates schema)
    db = MediaArchiveDB()
    
    # Add video from your notebook analysis
    success, video_id, error = db.add_video(video_data)
    
    # Quick integration with existing notebook workflow
    success, video_id, error = db.quick_add_video_from_analysis(
        video_info, scenes, keyframes, analysis_results
    )
    
    # Search operations
    success, results, error = db.search_by_text("large fish")
    success, results, error = db.search_similar_embeddings(embedding_vector)
    
    # Get statistics
    print(db.get_simple_stats())

Features:
    - SQLAlchemy ORM with automatic relationship management
    - FAISS vector similarity search with lazy loading
    - Comprehensive validation and error handling
    - Batch processing for efficient data insertion
    - Context managers for safe session handling
    - Notebook-friendly convenience methods
    - Beautiful formatted output for analysis results
"""

from .media_db import MediaArchiveDB

# Import ORM models for advanced usage
from .schemas import (
    Base, Video, Scene, Keyframe, ClipAnalysis, EmbeddingMetadata,
    get_database_url, create_all_tables, drop_all_tables
)

# Import query operations for advanced usage
from .queries import DatabaseQueries

# Import utility functions for notebook integration
from .utils import (
    # Validation functions
    validate_video_data, validate_scene_data, validate_keyframe_data,
    validate_analysis_data, validate_embedding_metadata,
    
    # ORM conversion utilities
    video_to_dict, scene_to_dict, keyframe_to_dict, 
    analysis_to_dict, embedding_to_dict,
    
    # Notebook integration helpers
    convert_pyscenedetect_to_orm, convert_keyframes_to_orm, 
    convert_clip_analysis_to_orm,
    
    # Display formatting
    format_search_results, format_video_summary_for_display,
    
    # File utilities
    is_video_file, generate_frame_filename
)

__version__ = "2.0.0"
__orm_version__ = "SQLAlchemy 2.x compatible"

# Main exports for simple usage
__all__ = [
    # Primary class
    "MediaArchiveDB",
    
    # ORM models (for advanced usage)
    "Base", "Video", "Scene", "Keyframe", "ClipAnalysis", "EmbeddingMetadata",
    
    # Query operations (for advanced usage)
    "DatabaseQueries",
    
    # Essential utilities
    "validate_video_data", "validate_scene_data", "validate_keyframe_data",
    "convert_pyscenedetect_to_orm", "convert_keyframes_to_orm", "convert_clip_analysis_to_orm",
    "format_video_summary_for_display", "is_video_file"
]

# Quick start examples for documentation
EXAMPLES = {
    "basic_usage": """
# Basic database operations
from database import MediaArchiveDB

db = MediaArchiveDB()

# Add video
video_data = {
    'filepath': 'path/to/video.mp4',
    'filename': 'video.mp4',
    'duration_seconds': 120.5,
    'fps': 30.0,
    'width': 1920,
    'height': 1080,
    'total_frames': 3615,
    'file_size_mb': 25.3
}

success, video_id, error = db.add_video(video_data)
if success:
    print(f"Video added with ID: {video_id}")
""",

    "notebook_integration": """
# Integrate with existing notebook workflow
from database import MediaArchiveDB

db = MediaArchiveDB()

# Use your existing variables from notebook
success, video_id, error = db.quick_add_video_from_analysis(
    video_info,      # From analyze_video_file()
    scenes,          # From detect_scenes_with_analysis()
    oarfish_frames,  # From extract_keyframes_from_scenes()
    analysis_results # From OpenCLIP analysis
)

if success:
    print(f"Complete analysis stored for video {video_id}")
    print(db.get_simple_stats())
""",

    "search_operations": """
# Search operations
from database import MediaArchiveDB

db = MediaArchiveDB()

# Text search in CLIP analysis
success, results, error = db.search_by_text("large fish", confidence_threshold=0.5)
if success:
    print(f"Found {results['count']} matches")
    for result in results['results'][:3]:
        print(f"- {result['label']} (confidence: {result['confidence']:.3f})")

# Vector similarity search (requires embeddings)
import numpy as np
query_embedding = np.random.random(512)  # Your actual embedding
success, similar, error = db.search_similar_embeddings(query_embedding, top_k=5)
""",

    "advanced_orm_usage": """
# Advanced ORM usage with direct session access
from database import MediaArchiveDB, Video, Scene

db = MediaArchiveDB()

# Direct ORM queries
with db.get_session() as session:
    # Get video with all related data
    video = session.query(Video).filter(Video.filename.like('%fish%')).first()
    
    if video:
        print(f"Video: {video.filename}")
        print(f"Scenes: {len(video.scenes)}")
        
        for scene in video.scenes[:3]:
            print(f"  Scene {scene.scene_number}: {scene.duration:.2f}s")
            print(f"    Keyframes: {len(scene.keyframes)}")
"""
}

def get_example(example_name: str) -> str:
    """
    Get example code for common usage patterns
    
    Args:
        example_name: One of 'basic_usage', 'notebook_integration', 
                     'search_operations', 'advanced_orm_usage'
    
    Returns:
        Example code as string
    """
    return EXAMPLES.get(example_name, "Example not found. Available: " + ", ".join(EXAMPLES.keys()))


def print_quick_start():
    """
    Print quick start guide for new users
    """
    print("""
🚀 Media Archive Database - Quick Start Guide

1️⃣ Basic Setup:
   from database import MediaArchiveDB
   db = MediaArchiveDB()  # Auto-creates database

2️⃣ Add Video Analysis:
   success, video_id, error = db.add_video(video_data)

3️⃣ Notebook Integration:
   success, video_id, error = db.quick_add_video_from_analysis(
       video_info, scenes, keyframes, analysis
   )

4️⃣ Search Content:
   success, results, error = db.search_by_text("large fish")

5️⃣ View Statistics:
   print(db.get_simple_stats())

📚 For more examples:
   from database import get_example
   print(get_example('notebook_integration'))

🔧 Advanced Usage:
   - Direct ORM access: Video, Scene, Keyframe models
   - Custom queries: DatabaseQueries class
   - Validation: validate_video_data(), etc.
   - Utilities: convert_pyscenedetect_to_orm(), etc.
    """)


def check_dependencies():
    """
    Check if required dependencies are available
    
    Returns:
        Dict with availability status of optional dependencies
    """
    dependencies = {
        'sqlalchemy': False,
        'faiss': False,
        'numpy': False
    }
    
    try:
        import sqlalchemy
        dependencies['sqlalchemy'] = True
    except ImportError:
        pass
    
    try:
        import faiss
        dependencies['faiss'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        pass
    
    return dependencies


def print_dependency_status():
    """
    Print status of required and optional dependencies
    """
    deps = check_dependencies()
    
    print("📦 Dependency Status:")
    print(f"   SQLAlchemy: {'✅ Available' if deps['sqlalchemy'] else '❌ Missing (required)'}")
    print(f"   FAISS:      {'✅ Available' if deps['faiss'] else '❌ Missing (optional - for embeddings)'}")
    print(f"   NumPy:      {'✅ Available' if deps['numpy'] else '❌ Missing (required for embeddings)'}")
    
    if not deps['sqlalchemy']:
        print("\n⚠️  Install SQLAlchemy: pip install sqlalchemy")
    
    if not deps['faiss']:
        print("\n💡 For embedding search, install FAISS: pip install faiss-cpu")


# Module-level convenience functions
def create_database(db_path: str = "./databases/media_archive.db") -> MediaArchiveDB:
    """
    Convenience function to create and return database instance
    
    Args:
        db_path: Path for database file
        
    Returns:
        MediaArchiveDB instance
    """
    return MediaArchiveDB(db_path=db_path)


def quick_setup() -> MediaArchiveDB:
    """
    Quick setup with default configuration for notebook usage
    
    Returns:
        Configured MediaArchiveDB instance
    """
    print("🔧 Setting up Media Archive Database...")
    
    # Check dependencies
    deps = check_dependencies()
    if not deps['sqlalchemy']:
        raise ImportError("SQLAlchemy is required. Install with: pip install sqlalchemy")
    
    # Create database
    db = MediaArchiveDB()
    
    print("✅ Database initialized successfully!")
    print(f"📁 Database location: {db.db_path}")
    print(f"📊 Embedding dimension: {db.embedding_dim}")
    
    if deps['faiss']:
        print("🔍 FAISS vector search: Available")
    else:
        print("⚠️  FAISS not available - install with: pip install faiss-cpu")
    
    print("\n" + "="*50)
    print(db.get_simple_stats())
    print("="*50)
    
    return db


# Export convenience functions
__all__ += [
    "get_example", "print_quick_start", "check_dependencies", 
    "print_dependency_status", "create_database", "quick_setup"
]