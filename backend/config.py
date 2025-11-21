"""
Configuration Settings for AI Tutor Application
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Database Configuration
class DatabaseConfig:
    """Database configuration"""
    
    # SQLite (default for development)
    SQLITE_DB_PATH = os.path.join(BASE_DIR, 'data', 'ai_tutor.db')
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{SQLITE_DB_PATH}'
    
    # PostgreSQL (for production - uncomment and configure)
    # POSTGRES_USER = os.getenv('POSTGRES_USER', 'ai_tutor')
    # POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
    # POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    # POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
    # POSTGRES_DB = os.getenv('POSTGRES_DB', 'ai_tutor_db')
    # SQLALCHEMY_DATABASE_URI = f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class ModelConfig:
    """ML Model configuration"""
    
    # Model paths
    ML_MODELS_DIR = os.path.join(BASE_DIR, '..', 'ml_models')
    RF_MODEL_PATH = os.path.join(ML_MODELS_DIR, 'rf_intent_classifier.pkl')
    BERT_MODEL_PATH = os.path.join(ML_MODELS_DIR, 'bert_intent_model')
    
    # Model settings
    USE_BERT = True  # Set to False to use only Random Forest (faster)
    CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for BERT
    MAX_SEQUENCE_LENGTH = 128


class ChatGPTConfig:
    """ChatGPT API configuration (Optional Backup)"""
    
    # API Settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')  # Set your API key here or in environment
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', 500))
    OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', 0.7))
    
    # Feature flags
    USE_CHATGPT_BACKUP = os.getenv('USE_CHATGPT_BACKUP', 'False') == 'True'
    CHATGPT_TIMEOUT = int(os.getenv('CHATGPT_TIMEOUT', 10))  # seconds
    
    # Educational context for better responses
    SYSTEM_PROMPT = """You are an AI Tutor specializing in computer science, machine learning, 
    data science, and programming education. You provide clear, concise, and educational 
    responses to students. Focus on:
    - Explaining concepts in simple terms
    - Providing practical examples
    - Offering step-by-step guidance
    - Suggesting learning resources
    - Being encouraging and supportive
    
    Always structure your responses to be educational and helpful."""


class APIConfig:
    """API configuration"""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here-change-in-production')
    DEBUG = os.getenv('DEBUG', 'True') == 'True'
    PORT = int(os.getenv('PORT', 5000))
    HOST = os.getenv('HOST', '0.0.0.0')
    
    # CORS settings
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
    
    # Rate limiting
    RATE_LIMIT_ENABLED = os.getenv('RATE_LIMIT_ENABLED', 'False') == 'True'
    RATE_LIMIT = os.getenv('RATE_LIMIT', '100 per hour')


class AppConfig:
    """Application configuration"""
    
    # Session settings
    SESSION_TIMEOUT = 3600  # 1 hour in seconds
    MAX_HISTORY_LENGTH = 100  # Maximum conversation history to keep
    
    # Response settings
    MAX_RESPONSE_LENGTH = 2000  # Maximum characters in response
    DEFAULT_SUGGESTIONS_COUNT = 3
    
    # Analytics
    ANALYTICS_ENABLED = True
    CLEANUP_OLD_SESSIONS_DAYS = 30  # Delete sessions older than this
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.path.join(BASE_DIR, 'logs', 'ai_tutor.log')


class KnowledgeBaseConfig:
    """Knowledge base configuration"""
    
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    EDUCATIONAL_CONTENT_PATH = os.path.join(DATA_DIR, 'educational_content.json')
    
    # Topics
    DEFAULT_TOPICS = [
        'Machine Learning',
        'Deep Learning',
        'Neural Networks',
        'Natural Language Processing',
        'Computer Vision',
        'Data Science',
        'Python Programming',
        'Algorithms',
        'Mathematics for AI'
    ]


# Environment-specific configurations
class DevelopmentConfig:
    """Development environment configuration"""
    DEBUG = True
    DATABASE_URI = DatabaseConfig.SQLALCHEMY_DATABASE_URI
    USE_BERT = True
    USE_CHATGPT_BACKUP = False  # Disable in development by default


class ProductionConfig:
    """Production environment configuration"""
    DEBUG = False
    # Use PostgreSQL in production
    DATABASE_URI = DatabaseConfig.SQLALCHEMY_DATABASE_URI
    USE_BERT = True
    USE_CHATGPT_BACKUP = True  # Enable in production for reliability
    RATE_LIMIT_ENABLED = True


class TestingConfig:
    """Testing environment configuration"""
    TESTING = True
    DATABASE_URI = 'sqlite:///:memory:'  # In-memory database for testing
    USE_BERT = False  # Use faster RF model for testing
    USE_CHATGPT_BACKUP = False


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env=None):
    """
    Get configuration for specified environment
    
    Args:
        env (str): Environment name (development, production, testing)
        
    Returns:
        Configuration class
    """
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')
    
    return config.get(env, config['default'])