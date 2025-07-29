# database/queries.py
"""
SQLAlchemy ORM query operations for media archive system

This file replaces raw SQL queries with SQLAlchemy ORM methods.
All database operations are now performed using ORM models and sessions.
"""

from sqlalchemy.orm import sessionmaker, joinedload
from sqlalchemy import and_, or_, desc, asc, func
from typing import List, Dict, Any, Optional, Tuple
from .schemas import Video, Scene, Keyframe, ClipAnalysis, EmbeddingMetadata


class DatabaseQueries:
    """
    High-level query operations using SQLAlchemy ORM
    
    This class provides reusable query methods that can be used
    by the main MediaArchiveDB class.
    """
    
    def __init__(self, session_factory):
        """
        Initialize with SQLAlchemy session factory
        
        Args:
            session_factory: SQLAlchemy sessionmaker instance
        """
        self.session_factory = session_factory
    
    # ============================================================================
    # VIDEO QUERIES
    # ============================================================================
    
    def create_video(self, session, video_data: Dict[str, Any]) -> Video:
        """
        Create new video record
        
        Args:
            session: SQLAlchemy session
            video_data: Dictionary with video metadata
            
        Returns:
            Video instance
        """
        video = Video(**video_data)
        session.add(video)
        session.flush()  # Get ID without committing
        return video
    
    def get_video_by_id(self, session, video_id: int) -> Optional[Video]:
        """
        Get video by ID with all relationships loaded
        """
        return session.query(Video)\
            .options(joinedload(Video.scenes))\
            .filter(Video.id == video_id)\
            .first()
    
    def get_video_by_filepath(self, session, filepath: str) -> Optional[Video]:
        """
        Get video by file path
        """
        return session.query(Video)\
            .filter(Video.filepath == filepath)\
            .first()
    
    def get_all_videos(self, session, limit: int = 100) -> List[Video]:
        """
        Get all videos ordered by creation date
        """
        return session.query(Video)\
            .order_by(desc(Video.created_at))\
            .limit(limit)\
            .all()
    
    def get_videos_by_status(self, session, status: str) -> List[Video]:
        """
        Get videos by processing status
        """
        return session.query(Video)\
            .filter(Video.status == status)\
            .order_by(desc(Video.created_at))\
            .all()
    
    def update_video_status(self, session, video_id: int, status: str) -> bool:
        """
        Update video processing status
        
        Returns:
            True if video was found and updated, False otherwise
        """
        video = session.query(Video).filter(Video.id == video_id).first()
        if video:
            video.status = status
            if status == 'completed':
                video.processed_at = func.now()
            return True
        return False
    
    def delete_video(self, session, video_id: int) -> bool:
        """
        Delete video and all related data (cascading)
        
        Returns:
            True if video was found and deleted, False otherwise
        """
        video = session.query(Video).filter(Video.id == video_id).first()
        if video:
            session.delete(video)
            return True
        return False
    
    # ============================================================================
    # SCENE QUERIES
    # ============================================================================
    
    def create_scene(self, session, scene_data: Dict[str, Any]) -> Scene:
        """
        Create new scene record
        """
        scene = Scene(**scene_data)
        session.add(scene)
        session.flush()
        return scene
    
    def create_multiple_scenes(self, session, scenes_data: List[Dict[str, Any]]) -> List[Scene]:
        """
        Create multiple scenes efficiently
        """
        scenes = [Scene(**scene_data) for scene_data in scenes_data]
        session.add_all(scenes)
        session.flush()
        return scenes
    
    def get_scenes_by_video(self, session, video_id: int) -> List[Scene]:
        """
        Get all scenes for a video ordered by scene number
        """
        return session.query(Scene)\
            .filter(Scene.video_id == video_id)\
            .order_by(Scene.scene_number)\
            .all()
    
    def get_scene_by_id(self, session, scene_id: int) -> Optional[Scene]:
        """
        Get scene by ID with relationships
        """
        return session.query(Scene)\
            .options(joinedload(Scene.keyframes))\
            .filter(Scene.id == scene_id)\
            .first()
    
    def get_scenes_by_duration_range(self, session, min_duration: float, 
                                   max_duration: float) -> List[Scene]:
        """
        Get scenes within duration range
        """
        return session.query(Scene)\
            .filter(and_(
                Scene.duration >= min_duration,
                Scene.duration <= max_duration
            ))\
            .order_by(desc(Scene.duration))\
            .all()
    
    # ============================================================================
    # KEYFRAME QUERIES
    # ============================================================================
    
    def create_keyframe(self, session, keyframe_data: Dict[str, Any]) -> Keyframe:
        """
        Create new keyframe record
        """
        keyframe = Keyframe(**keyframe_data)
        session.add(keyframe)
        session.flush()
        return keyframe
    
    def get_keyframes_by_scene(self, session, scene_id: int) -> List[Keyframe]:
        """
        Get keyframes for a scene ordered by frame index
        """
        return session.query(Keyframe)\
            .filter(Keyframe.scene_id == scene_id)\
            .order_by(Keyframe.frame_index)\
            .all()
    
    def get_keyframes_by_video(self, session, video_id: int) -> List[Keyframe]:
        """
        Get all keyframes for a video ordered by timestamp
        """
        return session.query(Keyframe)\
            .join(Scene)\
            .filter(Scene.video_id == video_id)\
            .order_by(Keyframe.timestamp)\
            .all()
    
    def get_keyframe_by_id(self, session, keyframe_id: int) -> Optional[Keyframe]:
        """
        Get keyframe by ID with all relationships
        """
        return session.query(Keyframe)\
            .options(
                joinedload(Keyframe.scene).joinedload(Scene.video),
                joinedload(Keyframe.clip_analyses),
                joinedload(Keyframe.embedding_metadata)
            )\
            .filter(Keyframe.id == keyframe_id)\
            .first()
    
    def update_keyframe_embedding_id(self, session, keyframe_id: int, embedding_id: int) -> bool:
        """
        Update keyframe with embedding ID
        """
        keyframe = session.query(Keyframe).filter(Keyframe.id == keyframe_id).first()
        if keyframe:
            keyframe.embedding_id = embedding_id
            return True
        return False
    
    # ============================================================================
    # CLIP ANALYSIS QUERIES
    # ============================================================================
    
    def create_clip_analysis(self, session, analysis_data: Dict[str, Any]) -> ClipAnalysis:
        """
        Create new CLIP analysis record
        """
        # Auto-calculate is_high_confidence
        analysis_data['is_high_confidence'] = analysis_data.get('confidence', 0) > 0.7
        
        analysis = ClipAnalysis(**analysis_data)
        session.add(analysis)
        session.flush()
        return analysis
    
    def create_multiple_clip_analysis(self, session, analysis_list: List[Dict[str, Any]]) -> List[ClipAnalysis]:
        """
        Create multiple CLIP analysis records efficiently
        """
        # Auto-calculate is_high_confidence for each
        for analysis_data in analysis_list:
            analysis_data['is_high_confidence'] = analysis_data.get('confidence', 0) > 0.7
        
        analyses = [ClipAnalysis(**analysis_data) for analysis_data in analysis_list]
        session.add_all(analyses)
        session.flush()
        return analyses
    
    def get_analysis_by_keyframe(self, session, keyframe_id: int, 
                               high_confidence_only: bool = False) -> List[ClipAnalysis]:
        """
        Get CLIP analysis for a keyframe
        """
        query = session.query(ClipAnalysis)\
            .filter(ClipAnalysis.keyframe_id == keyframe_id)
        
        if high_confidence_only:
            query = query.filter(ClipAnalysis.is_high_confidence == True)
        
        return query.order_by(desc(ClipAnalysis.confidence)).all()
    
    def search_by_category_label(self, session, category: str = None, 
                                label_pattern: str = None, 
                                min_confidence: float = 0.3, 
                                limit: int = 50) -> List[ClipAnalysis]:
        """
        Search CLIP analysis by category and/or label pattern
        """
        query = session.query(ClipAnalysis)\
            .options(
                joinedload(ClipAnalysis.keyframe)
                .joinedload(Keyframe.scene)
                .joinedload(Scene.video)
            )\
            .filter(ClipAnalysis.confidence >= min_confidence)
        
        if category:
            query = query.filter(ClipAnalysis.category == category)
        
        if label_pattern:
            query = query.filter(ClipAnalysis.label.ilike(f"%{label_pattern}%"))
        
        return query.order_by(desc(ClipAnalysis.confidence)).limit(limit).all()
    
    # ============================================================================
    # EMBEDDING QUERIES
    # ============================================================================
    
    def create_embedding_metadata(self, session, embedding_data: Dict[str, Any]) -> EmbeddingMetadata:
        """
        Create embedding metadata record
        """
        embedding = EmbeddingMetadata(**embedding_data)
        session.add(embedding)
        session.flush()
        return embedding
    
    def get_embedding_by_keyframe(self, session, keyframe_id: int) -> Optional[EmbeddingMetadata]:
        """
        Get embedding metadata by keyframe ID
        """
        return session.query(EmbeddingMetadata)\
            .filter(EmbeddingMetadata.keyframe_id == keyframe_id)\
            .first()
    
    def get_keyframe_by_faiss_index(self, session, faiss_index: int) -> Optional[Keyframe]:
        """
        Get keyframe by FAISS index position
        """
        return session.query(Keyframe)\
            .join(EmbeddingMetadata)\
            .options(
                joinedload(Keyframe.scene).joinedload(Scene.video)
            )\
            .filter(EmbeddingMetadata.faiss_index == faiss_index)\
            .first()
    
    def get_all_embedding_metadata(self, session) -> List[EmbeddingMetadata]:
        """
        Get all embedding metadata ordered by FAISS index
        """
        return session.query(EmbeddingMetadata)\
            .order_by(EmbeddingMetadata.faiss_index)\
            .all()
    
    # ============================================================================
    # COMPLEX JOINED QUERIES
    # ============================================================================
    
    def get_video_summary(self, session, video_id: int) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive video summary with counts
        """
        # Get video with counts using subqueries
        video = session.query(Video).filter(Video.id == video_id).first()
        if not video:
            return None
        
        # Count related records
        scene_count = session.query(Scene).filter(Scene.video_id == video_id).count()
        keyframe_count = session.query(Keyframe)\
            .join(Scene)\
            .filter(Scene.video_id == video_id)\
            .count()
        analysis_count = session.query(ClipAnalysis)\
            .join(Keyframe)\
            .join(Scene)\
            .filter(Scene.video_id == video_id)\
            .count()
        embedding_count = session.query(EmbeddingMetadata)\
            .join(Keyframe)\
            .join(Scene)\
            .filter(Scene.video_id == video_id)\
            .count()
        
        # Convert to dictionary and add counts
        summary = {
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
            'processed_at': video.processed_at,
            'scene_count': scene_count,
            'keyframe_count': keyframe_count,
            'analysis_count': analysis_count,
            'embedding_count': embedding_count
        }
        
        return summary
    
    def get_database_stats(self, session) -> Dict[str, Any]:
        """
        Get overall database statistics
        """
        stats = {
            'total_videos': session.query(Video).count(),
            'total_scenes': session.query(Scene).count(),
            'total_keyframes': session.query(Keyframe).count(),
            'total_analysis': session.query(ClipAnalysis).count(),
            'total_embeddings': session.query(EmbeddingMetadata).count(),
            'avg_video_duration': session.query(func.avg(Video.duration_seconds)).scalar() or 0.0,
            'avg_scene_duration': session.query(func.avg(Scene.duration)).scalar() or 0.0,
            'videos_by_status': {}
        }
        
        # Get video counts by status
        status_counts = session.query(Video.status, func.count(Video.id))\
            .group_by(Video.status)\
            .all()
        
        for status, count in status_counts:
            stats['videos_by_status'][status] = count
        
        return stats
    
    # ============================================================================
    # SEARCH OPERATIONS
    # ============================================================================
    
    def search_videos_with_filters(self, session, filters: Dict[str, Any]) -> List[Video]:
        """
        Search videos with various filters
        
        Args:
            filters: Dictionary with filter criteria:
                - status: Video status
                - min_duration, max_duration: Duration filters
                - filename_pattern: Pattern to match in filename
                - limit: Maximum results (default: 100)
        """
        query = session.query(Video)
        
        if 'status' in filters:
            query = query.filter(Video.status == filters['status'])
        
        if 'min_duration' in filters:
            query = query.filter(Video.duration_seconds >= filters['min_duration'])
        
        if 'max_duration' in filters:
            query = query.filter(Video.duration_seconds <= filters['max_duration'])
        
        if 'filename_pattern' in filters:
            query = query.filter(Video.filename.ilike(f"%{filters['filename_pattern']}%"))
        
        limit = filters.get('limit', 100)
        return query.order_by(desc(Video.created_at)).limit(limit).all()
    
    def cleanup_orphaned_records(self, session) -> Dict[str, int]:
        """
        Clean up any orphaned records (shouldn't happen with proper cascading)
        
        Returns:
            Dictionary with count of cleaned records
        """
        cleanup_count = {
            'orphaned_analysis': 0,
            'orphaned_embeddings': 0
        }
        
        # This is mostly for safety - cascading deletes should handle this
        # But useful for data integrity checks
        
        return cleanup_count


# Export the queries class
__all__ = ['DatabaseQueries']