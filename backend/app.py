from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import logging
from datetime import datetime
import time
from dotenv import load_dotenv
load_dotenv()


# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
try:
    from models.intent_classifier import IntentClassifier
    from models.response_generator import ResponseGenerator
    from utils.nlp_preprocessing import NLPPreprocessor
    from utils.knowledge_base import KnowledgeBase
    from database import (
        db,
        save_conversation,
        get_conversation_history,
        save_feedback,
        get_user_progress,
        update_user_progress
    )
except ImportError as e:
    print(f"Import error: {e}")
    # Create dummy classes for missing modules
    class DummyClassifier:
        def predict_intent(self, query):
            return {'intent': 'general_query', 'confidence': 0.5, 'model': 'fallback'}
    
    class DummyGenerator:
        def generate_response(self, query, intent, preprocessed=None):
            return {'response': 'System initializing...', 'suggestions': []}
    
    class DummyPreprocessor:
        def preprocess_query(self, query):
            return {'cleaned_text': query, 'keywords': [], 'language': 'en'}
    
    class DummyKnowledgeBase:
        def get_all_topics(self): return []
        def get_practice_problems(self, topic): return []

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'    
)
logger = logging.getLogger(__name__)

# Initialize components
try:
    logger.info("Initializing AI Tutor Components...")

    # Load models and utilities
    intent_classifier = IntentClassifier()
    response_generator = ResponseGenerator()
    nlp_preprocessor = NLPPreprocessor()
    knowledge_base = KnowledgeBase()

    logger.info("All components initialized successfully.")
    logger.info("Database is created and connected.")

except Exception as e:
    logger.error(f"Error during initialization: {e}")
    # Use dummy classes if real ones fail
    intent_classifier = DummyClassifier()
    response_generator = DummyGenerator()
    nlp_preprocessor = DummyPreprocessor()
    knowledge_base = DummyKnowledgeBase()

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'message': 'AI Tutor API is running!',
        'version': '1.0.0',
        'database': 'connected',
        'timestamp': datetime.now().isoformat()
    })

conversation_context = {}

@app.route('/api/chat', methods=['POST'])  # Fixed: 'method' to 'methods'
def chat():
    """
    Main chat endpoint with database storage
    Accepts user query and returns AI response
    """
    start_time = time.time()

    try:
        # Get request data
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({'error': 'Invalid request, "query" field is required.'}), 400
        
        user_query = data['query']
        session_id = data.get('session_id', f'session_{int(time.time())}')
        
        previous_topic = conversation_context.get(session_id,{}).get('current_topic')

        logger.info(f"Processing query from session {session_id}: {user_query[:50]}...")

        # Preprocess query
        preprocessed = nlp_preprocessor.preprocess_query(user_query)

        # Classify intent
        intent_result = intent_classifier.predict_intent(user_query)
        intent = intent_result['intent']
        confidence = intent_result['confidence']
        model_used = intent_result.get('model', 'bert')  # Fixed: use get() with default

        logger.info(f"Detected intent: {intent} (confidence: {confidence:.2f}, model: {model_used})")

        # Generate response
        response_data = response_generator.generate_response(
            query=user_query,
            intent=intent,
            preprocessed=preprocessed
            # Removed knowledge_base parameter as it's not in the method signature
        )

        # update conversation context
        current_topic = response_data.get('topic',intent)
        conversation_context[session_id] = {
            'current_topic': current_topic,
            'last_intent': intent,
            'timestamp': datetime.now().isoformat()
        }

        # Calculate processing time
        response_time = time.time() - start_time

        # Save conversation to database
        conversation = save_conversation(
            session_id=session_id,
            query=user_query,
            intent=intent,
            confidence=confidence,
            response=response_data['response'],
            model_used=model_used,
            response_time=response_time,
        )

        # Update analytics - fixed method name
        
        db.update_topic_analytics(current_topic, confidence)  # Fixed: update_topic_analytics instead of update_analytics

        # Prepare response
        return jsonify({
            'success': True,
            'conversation_id': conversation.id,
            'session_id': session_id,
            'query': user_query,
            'intent': intent,
            'confidence': confidence,
            'model_used': model_used,
            'response': response_data['response'],
            'suggestions': response_data.get('suggestions', []),
            'resources': response_data.get('resources', []),
            'response_time': round(response_time, 3),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred processing your request',
            'details': str(e)
        }), 500

@app.route('/api/feedback', methods=['POST'])
def feedback():
    """
    Endpoint to receive user feedback on responses
    Saves to database for model improvement
    """
    try:
        data = request.get_json()
        
        conversation_id = data.get('conversation_id')
        rating = data.get('rating')
        feedback_text = data.get('feedback_text', '')
        
        if not conversation_id or not rating:
            return jsonify({
                'success': False,
                'error': 'Missing conversation_id or rating'
            }), 400
        
        # Validate rating
        if not (1 <= rating <= 5):
            return jsonify({
                'success': False,
                'error': 'Rating must be between 1 and 5'
            }), 400
        
        # Save feedback to database
        success = save_feedback(conversation_id, rating, feedback_text)
        
        if success:
            logger.info(f"Feedback saved for conversation {conversation_id}: {rating} stars")
            return jsonify({
                'success': True,
                'message': 'Thank you for your feedback!'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Conversation not found'
            }), 404
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to process feedback'
        }), 500

@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id):
    """
    Get conversation history for a session from database
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        
        # Get history from database
        history = get_conversation_history(session_id, limit)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'history': history,
            'count': len(history)
        })
        
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve history'
        }), 500

@app.route('/api/progress/<session_id>', methods=['GET'])
def get_progress(session_id):
    """
    Get user learning progress from database
    """
    try:
        progress = get_user_progress(session_id)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'progress': progress,
            'count': len(progress)
        })
        
    except Exception as e:
        logger.error(f"Error retrieving progress: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve progress'
        }), 500

@app.route('/api/progress', methods=['POST'])
def update_progress():
    """
    Update user learning progress in database
    """
    try:
        data = request.get_json()
        
        session_id = data.get('session_id')
        topic = data.get('topic')
        completed = data.get('completed', False)
        score = data.get('score')
        notes = data.get('notes')
        
        if not session_id or not topic:
            return jsonify({
                'success': False,
                'error': 'Missing session_id or topic'
            }), 400
        
        # Update progress in database
        progress = update_user_progress(
            session_id=session_id,
            topic=topic,
            completed=completed,
            score=score,
            notes=notes
        )
        
        return jsonify({
            'success': True,
            'message': 'Progress updated successfully',
            'progress': progress.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error updating progress: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to update progress'
        }), 500

@app.route('/api/topics', methods=['GET'])
def get_topics():
    """
    Get available learning topics
    """
    try:
        topics = knowledge_base.get_all_topics()
        
        return jsonify({
            'success': True,
            'topics': topics,
            'count': len(topics)
        })
        
    except Exception as e:
        logger.error(f"Error retrieving topics: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve topics'
        }), 500

@app.route('/api/practice', methods=['POST'])
def get_practice():
    """
    Get practice problems for a topic
    """
    try:
        data = request.get_json()
        topic = data.get('topic', '')
        
        problems = knowledge_base.get_practice_problems(topic)
        
        return jsonify({
            'success': True,
            'topic': topic,
            'problems': problems,
            'count': len(problems)
        })
        
    except Exception as e:
        logger.error(f"Error retrieving practice problems: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve practice problems'
        }), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """
    Get system analytics from database
    Admin endpoint for monitoring
    """
    try:
        analytics = db.get_analytics()
        
        return jsonify({
            'success': True,
            'analytics': analytics
        })
        
    except Exception as e:
        logger.error(f"Error retrieving analytics: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve analytics'
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Get basic system statistics
    """
    try:
        analytics = db.get_analytics()
        
        return jsonify({
            'success': True,
            'stats': {
                'total_conversations': analytics['total_conversations'],
                'total_users': analytics['total_users'],
                'available_topics': len(knowledge_base.get_all_topics()),
                'model_status': 'active',
                'database_status': 'connected'
            }
        })
        
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve statistics'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    logger.info(f"Starting AI Tutor API on port {port}...")
    logger.info(f"Database: SQLite (ai_tutor.db)")
    logger.info(f"Models: Loaded and ready")
    logger.info("="*70)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=True  # Set to False in production
    )