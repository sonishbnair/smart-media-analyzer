# database/schemas.py
"""
SQLAlchemy ORM models for media archive system
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

# Base class for all models
Base = declarative_base()


class Video(Base):
    """
    Video metadata model
    """
    __tablename__ = 'videos'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filepath = Column(String(500), nullable=False, unique=True)
    filename = Column(String(255), nullable=False)
    duration_seconds = Column(Float, nullable=False)
    fps = Column(Float, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    total_frames = Column(Integer, nullable=False)
    file_size_mb = Column(Float, nullable=False)
    created_at = Column(DateTime, default=func.now())
    processed_at = Column(DateTime, nullable=True)
    status = Column(String(20), default='pending')
    
    # Relationships
    scenes = relationship("Scene", back_populates="video", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'processing', 'completed', 'error')", name='check_video_status'),
        CheckConstraint("duration_seconds > 0", name='check_duration_positive'),
        CheckConstraint("fps > 0", name='check_fps_positive'),
        CheckConstraint("width > 0", name='check_width_positive'),
        CheckConstraint("height > 0", name='check_height_positive'),
        CheckConstraint("total_frames > 0", name='check_frames_positive'),
        CheckConstraint("file_size_mb > 0", name='check_filesize_positive'),
    )
    
    def __repr__(self):
        return f"<Video(id={self.id}, filename='{self.filename}', duration={self.duration_seconds}s)>"


class Scene(Base):
    """
    Scene detection results model
    """
    __tablename__ = 'scenes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(Integer, ForeignKey('videos.id', ondelete='CASCADE'), nullable=False)
    scene_number = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)
    detection_method = Column(String(50), nullable=False, default='ContentDetector')
    threshold_used = Column(Float, nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    video = relationship("Video", back_populates="scenes")
    keyframes = relationship("Keyframe", back_populates="scene", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("start_time >= 0", name='check_start_time_positive'),
        CheckConstraint("end_time > start_time", name='check_end_after_start'),
        CheckConstraint("duration > 0", name='check_scene_duration_positive'),
        CheckConstraint("scene_number > 0", name='check_scene_number_positive'),
        CheckConstraint("threshold_used > 0", name='check_threshold_positive'),
    )
    
    def __repr__(self):
        return f"<Scene(id={self.id}, video_id={self.video_id}, scene={self.scene_number}, duration={self.duration}s)>"


class Keyframe(Base):
    """
    Extracted keyframe metadata model
    """
    __tablename__ = 'keyframes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    scene_id = Column(Integer, ForeignKey('scenes.id', ondelete='CASCADE'), nullable=False)
    frame_index = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)
    frame_number = Column(Integer, nullable=False)
    frame_position = Column(String(10), nullable=False)
    embedding_id = Column(Integer, nullable=True)
    frame_path = Column(String(500), nullable=True)
    extraction_method = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    scene = relationship("Scene", back_populates="keyframes")
    clip_analyses = relationship("ClipAnalysis", back_populates="keyframe", cascade="all, delete-orphan")
    embedding_metadata = relationship("EmbeddingMetadata", back_populates="keyframe", uselist=False, cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("frame_position IN ('start', 'middle', 'end', 'all')", name='check_frame_position'),
        CheckConstraint("frame_index > 0", name='check_frame_index_positive'),
        CheckConstraint("timestamp >= 0", name='check_timestamp_positive'),
        CheckConstraint("frame_number >= 0", name='check_frame_number_positive'),
    )
    
    def __repr__(self):
        return f"<Keyframe(id={self.id}, scene_id={self.scene_id}, timestamp={self.timestamp}s)>"


class ClipAnalysis(Base):
    """
    CLIP analysis results model
    """
    __tablename__ = 'clip_analysis'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    keyframe_id = Column(Integer, ForeignKey('keyframes.id', ondelete='CASCADE'), nullable=False)
    category = Column(String(100), nullable=False)
    label = Column(String(200), nullable=False)
    confidence = Column(Float, nullable=False)
    is_high_confidence = Column(Boolean, nullable=False)
    analysis_model = Column(String(50), nullable=False, default='ViT-B-32')
    analysis_version = Column(String(20), nullable=False, default='1.0')
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    keyframe = relationship("Keyframe", back_populates="clip_analyses")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name='check_confidence_range'),
    )
    
    def __repr__(self):
        return f"<ClipAnalysis(id={self.id}, keyframe_id={self.keyframe_id}, label='{self.label}', confidence={self.confidence})>"


class EmbeddingMetadata(Base):
    """
    FAISS embedding metadata model
    """
    __tablename__ = 'embeddings_metadata'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    keyframe_id = Column(Integer, ForeignKey('keyframes.id', ondelete='CASCADE'), nullable=False, unique=True)
    faiss_index = Column(Integer, nullable=False, unique=True)
    embedding_model = Column(String(50), nullable=False, default='ViT-B-32')
    embedding_dim = Column(Integer, nullable=False, default=512)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    keyframe = relationship("Keyframe", back_populates="embedding_metadata")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("faiss_index >= 0", name='check_faiss_index_positive'),
        CheckConstraint("embedding_dim > 0", name='check_embedding_dim_positive'),
    )
    
    def __repr__(self):
        return f"<EmbeddingMetadata(id={self.id}, keyframe_id={self.keyframe_id}, faiss_index={self.faiss_index})>"


# Utility functions for SQLAlchemy
def get_database_url(db_path: str = "./databases/media_archive.db") -> str:
    """
    Generate SQLAlchemy database URL for SQLite
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        SQLAlchemy database URL
    """
    return f"sqlite:///{db_path}"


def create_all_tables(engine):
    """
    Create all tables in the database
    
    Args:
        engine: SQLAlchemy engine instance
    """
    Base.metadata.create_all(engine)


def drop_all_tables(engine):
    """
    Drop all tables from the database (use with caution!)
    
    Args:
        engine: SQLAlchemy engine instance
    """
    Base.metadata.drop_all(engine)


def get_table_names():
    """
    Get list of all table names defined in models
    
    Returns:
        List of table names
    """
    return [table.name for table in Base.metadata.tables.values()]


def get_model_by_tablename(tablename: str):
    """
    Get model class by table name
    
    Args:
        tablename: Name of the table
        
    Returns:
        Model class or None if not found
    """
    for mapper in Base.registry.mappers:
        model = mapper.class_
        if hasattr(model, '__tablename__') and model.__tablename__ == tablename:
            return model
    return None


# Export all models for easy importing
__all__ = [
    'Base',
    'Video', 
    'Scene', 
    'Keyframe', 
    'ClipAnalysis', 
    'EmbeddingMetadata',
    'get_database_url',
    'create_all_tables',
    'drop_all_tables',
    'get_table_names',
    'get_model_by_tablename'
]