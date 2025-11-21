"""
Database Models and Configuration
SQLAlchemy ORM for conversation history and user data
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
import json

# Create Base class for models
Base = declarative_base()

class User(Base):
    """User model for tracking learners"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(100))
    email = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    total_queries = Column(Integer, default=0)
    
    # Relationships
    conversations = relationship('Conversation', back_populates='user', cascade='all, delete-orphan')
    progress = relationship('UserProgress', back_populates='user', cascade='all, delete-orphan')
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'name': self.name,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_active': self.last_active.isoformat() if self.last_active else None,
            'total_queries': self.total_queries
        }

class Conversation(Base):
    """Conversation history model"""
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    session_id = Column(String(100), nullable=False, index=True)
    query = Column(Text, nullable=False)
    intent = Column(String(50))
    confidence = Column(Float)
    response = Column(Text)
    model_used = Column(String(50))  # 'bert' or 'random_forest'
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    response_time = Column(Float)  # Response time in seconds
    feedback_rating = Column(Integer)  # 1-5 rating
    feedback_text = Column(Text)
    
    # Relationship
    user = relationship('User', back_populates='conversations')
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'query': self.query,
            'intent': self.intent,
            'confidence': self.confidence,
            'response': self.response,
            'model_used': self.model_used,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'response_time': self.response_time,
            'feedback_rating': self.feedback_rating,
            'feedback_text': self.feedback_text
        }

class UserProgress(Base):
    """Track user learning progress"""
    __tablename__ = 'user_progress'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    topic = Column(String(100), nullable=False)
    completed = Column(Boolean, default=False)
    score = Column(Float)
    attempts = Column(Integer, default=0)
    last_practiced = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)
    
    # Relationship
    user = relationship('User', back_populates='progress')
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'topic': self.topic,
            'completed': self.completed,
            'score': self.score,
            'attempts': self.attempts,
            'last_practiced': self.last_practiced.isoformat() if self.last_practiced else None,
            'notes': self.notes
        }

class TopicAnalytics(Base):
    """Analytics for popular topics and queries"""
    __tablename__ = 'topic_analytics'
    
    id = Column(Integer, primary_key=True)
    topic = Column(String(100), nullable=False, unique=True, index=True)
    query_count = Column(Integer, default=0)
    avg_confidence = Column(Float)
    last_queried = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'topic': self.topic,
            'query_count': self.query_count,
            'avg_confidence': self.avg_confidence,
            'last_queried': self.last_queried.isoformat() if self.last_queried else None
        }

class Database:
    """Database manager class"""
    
    def __init__(self, db_url=None):
        """
        Initialize database connection
        
        Args:
            db_url (str): Database URL. If None, uses SQLite with default path
        """
        if db_url is None:
            # Use SQLite by default
            db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ai_tutor.db')
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            db_url = f'sqlite:///{db_path}'
        
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create all tables
        Base.metadata.create_all(self.engine)
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def get_or_create_user(self, session_id, name=None, email=None, session=None):
        """
        Get existing user or create new one
        
        Args:
            session_id (str): Session ID
            name (str): User name (optional)
            email (str): User email (optional)
            
        Returns:
            User: User object
        """
        own_session = False 
        if session is None:
            session = self.get_session()
            own_session = True
        
        try:
            user = session.query(User).filter_by(session_id=session_id).first()
            
            if not user:
                user = User(
                    session_id=session_id,
                    name=name,
                    email=email
                )
                session.add(user)
                session.commit()
                # Refresh to get the ID
                session.refresh(user)
            else:
                # Update last active
                user.last_active = datetime.utcnow()
                session.commit()
            
            return user
        except Exception as e:
            session.rollback()
            raise e
        finally:
            if own_session:
                session.close()
    
    def save_conversation(self, session_id, query, intent, confidence, 
                         response, model_used='bert', response_time=None):
        """
        Save conversation to database
        
        Args:
            session_id (str): Session ID
            query (str): User query
            intent (str): Detected intent
            confidence (float): Confidence score
            response (str): Generated response
            model_used (str): Model used for prediction
            response_time (float): Response time in seconds
            
        Returns:
            Conversation: Saved conversation object
        """
        session = self.get_session()
        try:
            # Get or create user
            user = self.get_or_create_user(session_id, session=session)
            
            # Create conversation record
            conversation = Conversation(
                user_id=user.id,
                session_id=session_id,
                query=query,
                intent=intent,
                confidence=confidence,
                response=response,
                model_used=model_used,
                response_time=response_time
            )
            
            session.add(conversation)
            
            # Update user query count
            user.total_queries += 1
            session.add(user)
            
            session.commit()
            session.refresh(conversation)  # Refresh to get the ID
            return conversation
        except Exception as e:
            session.rollback()
            raise e
    
    def get_conversation_history(self, session_id, limit=50):
        """
        Get conversation history for a session
        
        Args:
            session_id (str): Session ID
            limit (int): Maximum number of records to return
            
        Returns:
            list: List of conversation dictionaries
        """
        session = self.get_session()
        try:
            conversations = session.query(Conversation).filter_by(
                session_id=session_id
            ).order_by(
                Conversation.timestamp.desc()
            ).limit(limit).all()
            
            return [conv.to_dict() for conv in conversations]
        finally:
            session.close()
    
    def save_feedback(self, conversation_id, rating, feedback_text=None):
        """
        Save user feedback for a conversation
        
        Args:
            conversation_id (int): Conversation ID
            rating (int): Rating (1-5)
            feedback_text (str): Optional feedback text
            
        Returns:
            bool: Success status
        """
        session = self.get_session()
        try:
            conversation = session.query(Conversation).filter_by(id=conversation_id).first()
            
            if conversation:
                conversation.feedback_rating = rating
                conversation.feedback_text = feedback_text
                session.commit()
                return True
            return False
        finally:
            session.close()
    
    def update_user_progress(self, session_id, topic, completed=False, 
                           score=None, notes=None):
        """
        Update user progress for a topic
        
        Args:
            session_id (str): Session ID
            topic (str): Topic name
            completed (bool): Whether topic is completed
            score (float): Score achieved
            notes (str): Additional notes
            
        Returns:
            UserProgress: Updated progress object
        """
        session = self.get_session()
        try:
            user = self.get_or_create_user(session_id)
            
            # Check if progress record exists
            progress = session.query(UserProgress).filter_by(
                user_id=user.id,
                topic=topic
            ).first()
            
            if not progress:
                progress = UserProgress(
                    user_id=user.id,
                    topic=topic
                )
                session.add(progress)
            
            # Update progress
            progress.completed = completed
            if score is not None:
                progress.score = score
            progress.attempts += 1
            progress.last_practiced = datetime.utcnow()
            if notes:
                progress.notes = notes
            
            session.commit()
            session.refresh(progress)
            return progress
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_user_progress(self, session_id):
        """
        Get all progress for a user
        
        Args:
            session_id (str): Session ID
            
        Returns:
            list: List of progress dictionaries
        """
        session = self.get_session()
        try:
            user = session.query(User).filter_by(session_id=session_id).first()
            if not user:
                return []
            
            progress_list = session.query(UserProgress).filter_by(
                user_id=user.id
            ).all()
            
            return [p.to_dict() for p in progress_list]
        finally:
            session.close()
    
    def update_topic_analytics(self, topic, confidence):
        """
        Update analytics for a topic
        
        Args:
            topic (str): Topic name
            confidence (float): Confidence score
        """
        session = self.get_session()
        try:
            analytics = session.query(TopicAnalytics).filter_by(topic=topic).first()
            
            if not analytics:
                analytics = TopicAnalytics(
                    topic=topic,
                    query_count=1,
                    avg_confidence=confidence
                )
                session.add(analytics)
            else:
                # Update running average
                total = analytics.query_count * analytics.avg_confidence
                analytics.query_count += 1
                analytics.avg_confidence = (total + confidence) / analytics.query_count
                analytics.last_queried = datetime.utcnow()
            
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_analytics(self):
        """
        Get overall analytics
        
        Returns:
            dict: Analytics data
        """
        session = self.get_session()
        try:
            total_users = session.query(User).count()
            total_conversations = session.query(Conversation).count()
            
            # Get popular topics
            popular_topics = session.query(TopicAnalytics).order_by(
                TopicAnalytics.query_count.desc()
            ).limit(10).all()
            
            # Get recent activity
            recent_conversations = session.query(Conversation).order_by(
                Conversation.timestamp.desc()
            ).limit(10).all()
            
            return {
                'total_users': total_users,
                'total_conversations': total_conversations,
                'popular_topics': [t.to_dict() for t in popular_topics],
                'recent_activity': [c.to_dict() for c in recent_conversations]
            }
        finally:
            session.close()
    
    def clear_old_sessions(self, days=30):
        """
        Clear sessions older than specified days
        
        Args:
            days (int): Number of days to keep
        """
        session = self.get_session()
        try:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Delete old conversations
            session.query(Conversation).filter(
                Conversation.timestamp < cutoff_date
            ).delete()
            
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

# Initialize global database instance
db = Database()

# Utility functions for easy access
def save_conversation(session_id, query, intent, confidence, response, model_used='bert', response_time=None):
    """Save conversation - shortcut function"""
    return db.save_conversation(session_id, query, intent, confidence, response, model_used, response_time)

def get_conversation_history(session_id, limit=50):
    """Get conversation history - shortcut function"""
    return db.get_conversation_history(session_id, limit)

def save_feedback(conversation_id, rating, feedback_text=None):
    """Save feedback - shortcut function"""
    return db.save_feedback(conversation_id, rating, feedback_text)

def get_user_progress(session_id):
    """Get user progress - shortcut function"""
    return db.get_user_progress(session_id)

def update_user_progress(session_id, topic, completed=False, score=None, notes=None):
    """Update user progress - shortcut function"""
    return db.update_user_progress(session_id, topic, completed, score, notes)

# Test database
if __name__ == "__main__":
    print("="*70)
    print("TESTING DATABASE")
    print("="*70)
    
    try:
        # Test user creation
        user = db.get_or_create_user('test_session_123', 'John Doe', 'john@example.com')
        print(f"\n✓ Created user: {user.to_dict()}")
        
        # Test conversation save
        conversation = db.save_conversation(
            session_id='test_session_123',
            query='What is machine learning?',
            intent='explanation',
            confidence=0.95,
            response='Machine learning is...',
            model_used='bert',
            response_time=0.5
        )
        print(f"\n✓ Saved conversation: {conversation.to_dict()}")
        
        # Test history retrieval
        history = db.get_conversation_history('test_session_123')
        print(f"\n✓ Retrieved {len(history)} conversations")
        
        # Test progress update
        progress = db.update_user_progress(
            session_id='test_session_123',
            topic='Machine Learning',
            completed=False,
            score=0.75
        )
        print(f"\n✓ Updated progress: {progress.to_dict()}")
        
        # Test analytics
        analytics = db.get_analytics()
        print(f"\n✓ Analytics: {analytics}")
        
        print("\n" + "="*70)
        print("✅ DATABASE TESTS COMPLETE!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()