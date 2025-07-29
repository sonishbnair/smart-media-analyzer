# database/media_db.py
"""
Main database class for media archive system using SQLAlchemy ORM
"""

import os
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from contextlib import contextmanager

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from .schemas import (
    Base, Video, Scene, Keyframe, ClipAnalysis, EmbeddingMetadata,
    get_database_url, create_all_tables
)
from .queries import DatabaseQueries
from .utils import (
    create_directories, validate_video_data, validate_scene_data,
    validate_keyframe_data, validate_analysis_data, save_embedding_mapping,
    load_embedding_mapping, format_search_results
)


class MediaArchiveDB:
    """
    Main database class for media archive with SQLAlchemy ORM and FAISS vector search
    
    Features:
    - SQLAlchemy ORM for all database operations
    - FAISS for vector similarity search
    - Auto-initialization and lazy loading
    - Session management with context managers
    - Comprehensive error handling
    - Batch operations for efficiency
    """
    
    def __init__(self, db_path: str = "../database/databases/media_archive.db", 
                 embedding_dim: int = 512, 
                 faiss_index_path: str = "../database/databases/embeddings.index",
                 mapping_path: str = "../database/databases/embedding_mapping.json"):
        """
        Initialize database connection and setup
        
        Args:
            db_path: Path to SQLite database file
            embedding_dim: Dimension of embedding vectors (default: 512 for ViT-B-32)
            faiss_index_path: Path to FAISS index file
            mapping_path: Path to embedding mapping JSON file
        """
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.faiss_index_path = faiss_index_path
        self.mapping_path = mapping_path
        
        # SQLAlchemy components
        self.engine = None
        self.SessionLocal = None
        self.queries = None
        
        # FAISS components (lazy-loaded)
        self.faiss_index = None
        self.embedding_mapping = {}  # faiss_index -> keyframe_id
        self.reverse_mapping = {}    # keyframe_id -> faiss_index
        self._faiss_loaded = False
        
        # Initialize database
        success, data, error = self.initialize_database()
        if not success:
            raise Exception(f"Database initialization failed: {error}")
    
    def initialize_database(self) -> Tuple[bool, Optional[str], str]:
        """
        Initialize SQLAlchemy engine, session factory, and database schema
        
        Returns:
            (success, data, error_message)
        """
        try:
            # Create directories
            success, error = create_directories(self.db_path)
            if not success:
                return False, None, error
            
            # Create SQLAlchemy engine
            database_url = get_database_url(self.db_path)
            self.engine = create_engine(
                database_url,
                echo=False,  # Set to True for SQL debugging
                pool_pre_ping=True,  # Verify connections before use
                connect_args={"check_same_thread": False}  # SQLite specific
            )
            
            # Create session factory
            self.SessionLocal = scoped_session(
                sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            )
            
            # Create all tables
            create_all_tables(self.engine)
            
            # Initialize query helper
            self.queries = DatabaseQueries(self.SessionLocal)
            
            return True, "Database initialized successfully", ""
            
        except Exception as e:
            return False, None, f"Database initialization failed: {str(e)}"
    
    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions with automatic cleanup
        
        Usage:
            with db.get_session() as session:
                # database operations
                session.commit()
        """
        session = self.SessionLocal()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def _load_faiss_index(self) -> Tuple[bool, Optional[str], str]:
        """
        Lazy-load FAISS index and mapping (called when first needed)
        
        Returns:
            (success, data, error_message)
        """
        if self._faiss_loaded:
            return True, "FAISS already loaded", ""
        
        if not FAISS_AVAILABLE:
            return False, None, "FAISS not installed. Run: pip install faiss-cpu"
        
        try:
            # Load or create FAISS index
            if os.path.exists(self.faiss_index_path):
                self.faiss_index = faiss.read_index(self.faiss_index_path)
            else:
                # Create new FlatL2 index
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                # Save empty index
                faiss.write_index(self.faiss_index, self.faiss_index_path)
            
            # Load mapping
            success, mapping_data, error = load_embedding_mapping(self.mapping_path)
            if not success:
                return False, None, error
            
            self.embedding_mapping = mapping_data
            # Create reverse mapping
            self.reverse_mapping = {v: k for k, v in mapping_data.items()}
            
            self._faiss_loaded = True
            return True, f"FAISS loaded with {self.faiss_index.ntotal} embeddings", ""
            
        except Exception as e:
            return False, None, f"FAISS loading failed: {str(e)}"
    
    def _save_faiss_state(self) -> Tuple[bool, Optional[str], str]:
        """
        Save FAISS index and mapping to disk
        """
        try:
            if self.faiss_index is not None:
                faiss.write_index(self.faiss_index, self.faiss_index_path)
            
            success, error = save_embedding_mapping(self.mapping_path, self.embedding_mapping)
            if not success:
                return False, None, error
            
            return True, "FAISS state saved", ""
            
        except Exception as e:
            return False, None, f"FAISS save failed: {str(e)}"
    
    # ============================================================================
    # VIDEO OPERATIONS
    # ============================================================================
    
    def add_video(self, video_data: Dict[str, Any]) -> Tuple[bool, Optional[int], str]:
        """
        Add new video to database using ORM
        
        Args:
            video_data: Dictionary with video metadata
            
        Returns:
            (success, video_id, error_message)
        """
        # Validate input
        is_valid, error = validate_video_data(video_data)
        if not is_valid:
            return False, None, f"Validation error: {error}"
        
        try:
            with self.get_session() as session:
                video = self.queries.create_video(session, video_data)
                session.commit()
                return True, video.id, ""
                
        except IntegrityError as e:
            return False, None, f"Video already exists or constraint violation: {str(e)}"
        except SQLAlchemyError as e:
            return False, None, f"Database error: {str(e)}"
    
    def get_video(self, video_id: int) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Get video by ID
        
        Returns:
            (success, video_data, error_message)
        """
        try:
            with self.get_session() as session:
                video = self.queries.get_video_by_id(session, video_id)
                
                if video:
                    # Convert ORM object to dictionary
                    video_dict = {
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
                    return True, video_dict, ""
                else:
                    return False, None, f"Video with ID {video_id} not found"
                    
        except SQLAlchemyError as e:
            return False, None, f"Database error: {str(e)}"
    
    def get_video_by_filepath(self, filepath: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Get video by file path
        """
        try:
            with self.get_session() as session:
                video = self.queries.get_video_by_filepath(session, filepath)
                
                if video:
                    video_dict = {
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
                    return True, video_dict, ""
                else:
                    return False, None, f"Video with path {filepath} not found"
                    
        except SQLAlchemyError as e:
            return False, None, f"Database error: {str(e)}"
    
    def update_video_status(self, video_id: int, status: str) -> Tuple[bool, Optional[str], str]:
        """
        Update video processing status
        
        Args:
            video_id: Video ID
            status: New status ('pending', 'processing', 'completed', 'error')
        """
        valid_statuses = ['pending', 'processing', 'completed', 'error']
        if status not in valid_statuses:
            return False, None, f"Invalid status. Must be one of: {valid_statuses}"
        
        try:
            with self.get_session() as session:
                success = self.queries.update_video_status(session, video_id, status)
                
                if success:
                    session.commit()
                    return True, f"Video status updated to {status}", ""
                else:
                    return False, None, f"Video with ID {video_id} not found"
                    
        except SQLAlchemyError as e:
            return False, None, f"Database error: {str(e)}"
    
    def delete_video(self, video_id: int) -> Tuple[bool, Optional[str], str]:
        """
        Delete video and all related data (cascading delete)
        """
        try:
            with self.get_session() as session:
                success = self.queries.delete_video(session, video_id)
                
                if success:
                    session.commit()
                    return True, f"Video {video_id} and related data deleted", ""
                else:
                    return False, None, f"Video with ID {video_id} not found"
                    
        except SQLAlchemyError as e:
            return False, None, f"Database error: {str(e)}"
    
    def get_all_videos(self) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Get all videos
        """
        try:
            with self.get_session() as session:
                videos = self.queries.get_all_videos(session)
                
                videos_list = []
                for video in videos:
                    video_dict = {
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
                    videos_list.append(video_dict)
                
                return True, videos_list, ""
                
        except SQLAlchemyError as e:
            return False, [], f"Database error: {str(e)}"
    
    # ============================================================================
    # SCENE OPERATIONS
    # ============================================================================
    
    def add_scene(self, scene_data: Dict[str, Any]) -> Tuple[bool, Optional[int], str]:
        """
        Add scene to database
        """
        is_valid, error = validate_scene_data(scene_data)
        if not is_valid:
            return False, None, f"Validation error: {error}"
        
        try:
            with self.get_session() as session:
                scene = self.queries.create_scene(session, scene_data)
                session.commit()
                return True, scene.id, ""
                
        except IntegrityError as e:
            return False, None, f"Scene constraint violation: {str(e)}"
        except SQLAlchemyError as e:
            return False, None, f"Database error: {str(e)}"
    
    def add_multiple_scenes(self, scenes_data: List[Dict[str, Any]]) -> Tuple[bool, List[int], str]:
        """
        Add multiple scenes efficiently
        """
        # Validate all scenes first
        for i, scene_data in enumerate(scenes_data):
            is_valid, error = validate_scene_data(scene_data)
            if not is_valid:
                return False, [], f"Validation error in scene {i}: {error}"
        
        try:
            with self.get_session() as session:
                scenes = self.queries.create_multiple_scenes(session, scenes_data)
                session.commit()
                
                scene_ids = [scene.id for scene in scenes]
                return True, scene_ids, ""
                
        except SQLAlchemyError as e:
            return False, [], f"Batch scene insert failed: {str(e)}"
    
    def get_scenes_by_video(self, video_id: int) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Get all scenes for a video
        """
        try:
            with self.get_session() as session:
                scenes = self.queries.get_scenes_by_video(session, video_id)
                
                scenes_list = []
                for scene in scenes:
                    scene_dict = {
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
                    scenes_list.append(scene_dict)
                
                return True, scenes_list, ""
                
        except SQLAlchemyError as e:
            return False, [], f"Database error: {str(e)}"
    
    # ============================================================================
    # KEYFRAME OPERATIONS
    # ============================================================================
    
    def add_keyframe(self, keyframe_data: Dict[str, Any]) -> Tuple[bool, Optional[int], str]:
        """
        Add keyframe to database
        """
        is_valid, error = validate_keyframe_data(keyframe_data)
        if not is_valid:
            return False, None, f"Validation error: {error}"
        
        try:
            with self.get_session() as session:
                keyframe = self.queries.create_keyframe(session, keyframe_data)
                session.commit()
                return True, keyframe.id, ""
                
        except SQLAlchemyError as e:
            return False, None, f"Database error: {str(e)}"
    
    def get_keyframes_by_scene(self, scene_id: int) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Get keyframes for a scene
        """
        try:
            with self.get_session() as session:
                keyframes = self.queries.get_keyframes_by_scene(session, scene_id)
                
                keyframes_list = []
                for keyframe in keyframes:
                    keyframe_dict = {
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
                    keyframes_list.append(keyframe_dict)
                
                return True, keyframes_list, ""
                
        except SQLAlchemyError as e:
            return False, [], f"Database error: {str(e)}"
    
    def get_keyframes_by_video(self, video_id: int) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Get all keyframes for a video
        """
        try:
            with self.get_session() as session:
                keyframes = self.queries.get_keyframes_by_video(session, video_id)
                
                keyframes_list = []
                for keyframe in keyframes:
                    keyframe_dict = {
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
                    keyframes_list.append(keyframe_dict)
                
                return True, keyframes_list, ""
                
        except SQLAlchemyError as e:
            return False, [], f"Database error: {str(e)}"
    
    # ============================================================================
    # CLIP ANALYSIS OPERATIONS
    # ============================================================================
    
    def add_clip_analysis(self, analysis_data: Dict[str, Any]) -> Tuple[bool, Optional[int], str]:
        """
        Add CLIP analysis result
        """
        is_valid, error = validate_analysis_data(analysis_data)
        if not is_valid:
            return False, None, f"Validation error: {error}"
        
        try:
            with self.get_session() as session:
                analysis = self.queries.create_clip_analysis(session, analysis_data)
                session.commit()
                return True, analysis.id, ""
                
        except SQLAlchemyError as e:
            return False, None, f"Database error: {str(e)}"
    
    def add_multiple_clip_analysis(self, analysis_list: List[Dict[str, Any]]) -> Tuple[bool, List[int], str]:
        """
        Add multiple CLIP analysis results efficiently
        """
        # Validate all analysis data
        for i, analysis_data in enumerate(analysis_list):
            is_valid, error = validate_analysis_data(analysis_data)
            if not is_valid:
                return False, [], f"Validation error in analysis {i}: {error}"
        
        try:
            with self.get_session() as session:
                analyses = self.queries.create_multiple_clip_analysis(session, analysis_list)
                session.commit()
                
                analysis_ids = [analysis.id for analysis in analyses]
                return True, analysis_ids, ""
                
        except SQLAlchemyError as e:
            return False, [], f"Batch analysis insert failed: {str(e)}"
    
    def get_analysis_by_keyframe(self, keyframe_id: int, high_confidence_only: bool = False) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Get CLIP analysis for a keyframe
        """
        try:
            with self.get_session() as session:
                analyses = self.queries.get_analysis_by_keyframe(session, keyframe_id, high_confidence_only)
                
                analyses_list = []
                for analysis in analyses:
                    analysis_dict = {
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
                    analyses_list.append(analysis_dict)
                
                return True, analyses_list, ""
                
        except SQLAlchemyError as e:
            return False, [], f"Database error: {str(e)}"
    
    # ============================================================================
    # EMBEDDING OPERATIONS
    # ============================================================================
    
    def add_embedding(self, keyframe_id: int, embedding_vector: np.ndarray, 
                     model_name: str = "ViT-B-32") -> Tuple[bool, Optional[int], str]:
        """
        Add embedding vector to FAISS index and metadata to database
        
        Args:
            keyframe_id: ID of the keyframe
            embedding_vector: Numpy array with embedding
            model_name: Name of the model used
            
        Returns:
            (success, faiss_index_position, error_message)
        """
        # Load FAISS if not already loaded
        success, _, error = self._load_faiss_index()
        if not success:
            return False, None, error
        
        try:
            # Validate embedding dimensions
            if embedding_vector.shape[0] != self.embedding_dim:
                return False, None, f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding_vector.shape[0]}"
            
            # Add to FAISS index
            embedding_vector = embedding_vector.reshape(1, -1).astype('float32')
            current_index = self.faiss_index.ntotal
            self.faiss_index.add(embedding_vector)
            
            # Update mappings
            self.embedding_mapping[current_index] = keyframe_id
            self.reverse_mapping[keyframe_id] = current_index
            
            # Add metadata to database
            with self.get_session() as session:
                embedding_data = {
                    'keyframe_id': keyframe_id,
                    'faiss_index': current_index,
                    'embedding_model': model_name,
                    'embedding_dim': self.embedding_dim
                }
                
                embedding = self.queries.create_embedding_metadata(session, embedding_data)
                
                # Update keyframe with embedding_id
                self.queries.update_keyframe_embedding_id(session, keyframe_id, current_index)
                
                session.commit()
            
            # Save FAISS state
            save_success, _, save_error = self._save_faiss_state()
            if not save_success:
                return False, None, f"Embedding added but save failed: {save_error}"
            
            return True, current_index, ""
            
        except Exception as e:
            return False, None, f"Embedding add failed: {str(e)}"
    
    def search_similar_embeddings(self, query_embedding: np.ndarray, 
                                top_k: int = 5) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Search for similar embeddings using FAISS
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            (success, results_with_metadata, error_message)
        """
        # Load FAISS if not already loaded
        success, _, error = self._load_faiss_index()
        if not success:
            return False, [], error
        
        try:
            if self.faiss_index.ntotal == 0:
                return True, [], "No embeddings in index"
            
            # Validate dimensions
            if query_embedding.shape[0] != self.embedding_dim:
                return False, [], f"Query embedding dimension mismatch"
            
            # Search FAISS
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            distances, indices = self.faiss_index.search(query_embedding, min(top_k, self.faiss_index.ntotal))
            
            # Get keyframe details for results
            results = []
            
            with self.get_session() as session:
                for i, (distance, faiss_idx) in enumerate(zip(distances[0], indices[0])):
                    if faiss_idx == -1:  # FAISS returns -1 for invalid results
                        continue
                    
                    # Get keyframe details
                    keyframe = self.queries.get_keyframe_by_faiss_index(session, int(faiss_idx))
                    
                    if keyframe:
                        result = {
                            'keyframe_id': keyframe.id,
                            'scene_id': keyframe.scene_id,
                            'timestamp': keyframe.timestamp,
                            'frame_path': keyframe.frame_path,
                            'video_id': keyframe.scene.video_id,
                            'video_filename': keyframe.scene.video.filename,
                            'faiss_index': int(faiss_idx),
                            'similarity_distance': float(distance),
                            'similarity_rank': i + 1
                        }
                        results.append(result)
            
            return True, results, ""
            
        except Exception as e:
            return False, [], f"Embedding search failed: {str(e)}"
    
    # ============================================================================
    # SEARCH OPERATIONS
    # ============================================================================
    
    def search_by_text(self, query_text: str, confidence_threshold: float = 0.3, 
                      top_k: int = 10) -> Tuple[bool, Dict[str, Any], str]:
        """
        Search by text in CLIP analysis labels
        
        Args:
            query_text: Text to search for
            confidence_threshold: Minimum confidence for results
            top_k: Maximum number of results
            
        Returns:
            (success, formatted_results, error_message)
        """
        try:
            with self.get_session() as session:
                analyses = self.queries.search_by_category_label(
                    session, 
                    label_pattern=query_text,
                    min_confidence=confidence_threshold,
                    limit=top_k
                )
                
                results = []
                for analysis in analyses:
                    result = {
                        'analysis_id': analysis.id,
                        'keyframe_id': analysis.keyframe_id,
                        'category': analysis.category,
                        'label': analysis.label,
                        'confidence': analysis.confidence,
                        'timestamp': analysis.keyframe.timestamp,
                        'frame_path': analysis.keyframe.frame_path,
                        'scene_number': analysis.keyframe.scene.scene_number,
                        'video_id': analysis.keyframe.scene.video_id,
                        'video_filename': analysis.keyframe.scene.video.filename
                    }
                    results.append(result)
                
                return True, format_search_results(results), ""
                
        except SQLAlchemyError as e:
            return False, {}, f"Text search failed: {str(e)}"
    
    def search_by_metadata(self, filters: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
        """
        Search by metadata filters
        
        Args:
            filters: Dictionary with filter criteria
                    - min_duration, max_duration: Video duration filters
                    - video_id: Specific video
                    - status: Video status
                    - filename_pattern: Pattern to match in filename
        """
        try:
            with self.get_session() as session:
                videos = self.queries.search_videos_with_filters(session, filters)
                
                results = []
                for video in videos:
                    result = {
                        'video_id': video.id,
                        'filepath': video.filepath,
                        'filename': video.filename,
                        'duration_seconds': video.duration_seconds,
                        'status': video.status,
                        'created_at': video.created_at,
                        'processed_at': video.processed_at
                    }
                    results.append(result)
                
                return True, format_search_results(results), ""
                
        except SQLAlchemyError as e:
            return False, {}, f"Metadata search failed: {str(e)}"
    
    # ============================================================================
    # UTILITY AND SUMMARY OPERATIONS
    # ============================================================================
    
    def get_video_summary(self, video_id: int) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Get comprehensive video summary with counts
        """
        try:
            with self.get_session() as session:
                summary = self.queries.get_video_summary(session, video_id)
                
                if summary:
                    return True, summary, ""
                else:
                    return False, None, f"Video with ID {video_id} not found"
                    
        except SQLAlchemyError as e:
            return False, None, f"Database error: {str(e)}"
    
    def get_database_stats(self) -> Tuple[bool, Dict[str, Any], str]:
        """
        Get overall database statistics
        """
        try:
            with self.get_session() as session:
                stats = self.queries.get_database_stats(session)
                
                # Add FAISS stats if loaded
                if self._faiss_loaded and self.faiss_index:
                    stats['faiss_embeddings'] = self.faiss_index.ntotal
                    stats['embedding_dimension'] = self.embedding_dim
                
                return True, stats, ""
                
        except SQLAlchemyError as e:
            return False, {}, f"Stats query failed: {str(e)}"
    
    def close(self) -> None:
        """
        Close database connections and save state
        """
        if self._faiss_loaded:
            self._save_faiss_state()
        
        if self.SessionLocal:
            self.SessionLocal.remove()
        
        if self.engine:
            self.engine.dispose()
    
    def __enter__(self):
        """
        Context manager entry
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit with cleanup
        """
        self.close()
    
    # ============================================================================
    # BATCH PROCESSING OPERATIONS
    # ============================================================================
    
    def process_video_analysis_batch(self, video_id: int, scenes_data: List[Dict[str, Any]], 
                                   keyframes_data: List[Dict[str, Any]], 
                                   analysis_data: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any], str]:
        """
        Process complete video analysis in a single transaction
        
        Args:
            video_id: ID of the video
            scenes_data: List of scene data dictionaries
            keyframes_data: List of keyframe data dictionaries  
            analysis_data: List of CLIP analysis data dictionaries
            
        Returns:
            (success, result_summary, error_message)
        """
        try:
            with self.get_session() as session:
                # Add scenes
                scenes = self.queries.create_multiple_scenes(session, scenes_data)
                
                # Map scene numbers to scene IDs for keyframes
                scene_id_map = {scene.scene_number: scene.id for scene in scenes}
                
                # Update keyframe data with correct scene IDs
                for keyframe_data in keyframes_data:
                    scene_number = keyframe_data.pop('scene_number', 1)  # Remove scene_number, use scene_id
                    keyframe_data['scene_id'] = scene_id_map.get(scene_number, scenes[0].id)
                
                # Add keyframes
                keyframes = []
                for keyframe_data in keyframes_data:
                    keyframe = self.queries.create_keyframe(session, keyframe_data)
                    keyframes.append(keyframe)
                
                # Map keyframe indices to keyframe IDs for analysis
                keyframe_id_map = {i: keyframe.id for i, keyframe in enumerate(keyframes)}
                
                # Update analysis data with correct keyframe IDs
                for analysis_item in analysis_data:
                    keyframe_index = analysis_item.pop('keyframe_index', 0)  # Remove keyframe_index
                    analysis_item['keyframe_id'] = keyframe_id_map.get(keyframe_index, keyframes[0].id)
                
                # Add CLIP analysis
                analyses = self.queries.create_multiple_clip_analysis(session, analysis_data)
                
                # Update video status to completed
                self.queries.update_video_status(session, video_id, 'completed')
                
                # Commit all changes
                session.commit()
                
                summary = {
                    'video_id': video_id,
                    'scenes_added': len(scenes),
                    'keyframes_added': len(keyframes),
                    'analyses_added': len(analyses),
                    'scene_ids': [scene.id for scene in scenes],
                    'keyframe_ids': [keyframe.id for keyframe in keyframes]
                }
                
                return True, summary, ""
                
        except SQLAlchemyError as e:
            return False, {}, f"Batch processing failed: {str(e)}"
    
    # ============================================================================
    # CONVENIENCE METHODS FOR NOTEBOOK USAGE
    # ============================================================================
    
    def quick_add_video_from_analysis(self, video_info: Dict[str, Any], 
                                     scenes_list: List, 
                                     keyframes_list: List[Dict[str, Any]], 
                                     clip_analysis: List[Dict[str, Any]]) -> Tuple[bool, int, str]:
        """
        Convenience method to add complete video analysis from notebook processing
        
        This method is designed to work with your existing notebook variables:
        - video_info: From analyze_video_file()
        - scenes_list: From detect_scenes_with_analysis() 
        - keyframes_list: From extract_keyframes_from_scenes()
        - clip_analysis: From analyze_frame_with_fallback()
        
        Args:
            video_info: Video metadata dictionary
            scenes_list: PySceneDetect scene list
            keyframes_list: Extracted keyframes list
            clip_analysis: CLIP analysis results
            
        Returns:
            (success, video_id, error_message)
        """
        try:
            # Step 1: Add video
            success, video_id, error = self.add_video(video_info)
            if not success:
                return False, 0, f"Failed to add video: {error}"
            
            # Step 2: Convert PySceneDetect scenes to database format
            scenes_data = []
            for i, scene in enumerate(scenes_list):
                scene_data = {
                    'video_id': video_id,
                    'scene_number': i + 1,
                    'start_time': scene[0].get_seconds(),
                    'end_time': scene[1].get_seconds(),
                    'duration': scene[1].get_seconds() - scene[0].get_seconds(),
                    'detection_method': 'ContentDetector',
                    'threshold_used': 27.0  # Default threshold
                }
                scenes_data.append(scene_data)
            
            # Step 3: Convert keyframes to database format
            keyframes_data = []
            for keyframe in keyframes_list:
                keyframe_data = {
                    'scene_number': keyframe['scene_number'],  # Will be converted to scene_id
                    'frame_index': keyframe.get('frame_index', 1),
                    'timestamp': keyframe['timestamp'],
                    'frame_number': keyframe['frame_number'],
                    'frame_position': 'middle',  # Default
                    'extraction_method': 'middle'
                }
                keyframes_data.append(keyframe_data)
            
            # Step 4: Convert CLIP analysis to database format
            analysis_data = []
            for i, analysis in enumerate(clip_analysis):
                if isinstance(analysis, dict) and 'results' in analysis:
                    # Handle comprehensive analysis format
                    for category, items in analysis['results'].items():
                        if category not in ['analysis_metadata', 'fallback_classification']:
                            for item in items:
                                analysis_item = {
                                    'keyframe_index': i,  # Will be converted to keyframe_id
                                    'category': category,
                                    'label': item['label'],
                                    'confidence': item['confidence'],
                                    'analysis_model': 'ViT-B-32',
                                    'analysis_version': '1.0'
                                }
                                analysis_data.append(analysis_item)
            
            # Step 5: Process everything in batch
            success, summary, error = self.process_video_analysis_batch(
                video_id, scenes_data, keyframes_data, analysis_data
            )
            
            if success:
                return True, video_id, f"Successfully processed video with {summary['scenes_added']} scenes"
            else:
                return False, video_id, error
                
        except Exception as e:
            return False, 0, f"Quick add failed: {str(e)}"
    
    def get_simple_stats(self) -> str:
        """
        Get simple database statistics as formatted string for notebook display
        """
        success, stats, error = self.get_database_stats()
        if success:
            return f"""
📊 Database Statistics:
   • Videos: {stats.get('total_videos', 0)}
   • Scenes: {stats.get('total_scenes', 0)}  
   • Keyframes: {stats.get('total_keyframes', 0)}
   • Analysis records: {stats.get('total_analysis', 0)}
   • Embeddings: {stats.get('total_embeddings', 0)}
   • Avg video duration: {stats.get('avg_video_duration', 0):.1f}s
   • FAISS embeddings: {stats.get('faiss_embeddings', 0)}
            """.strip()
        else:
            return f"❌ Error getting stats: {error}"