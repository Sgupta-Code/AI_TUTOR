"""
Setup and Test Script for AI Tutor Backend
Run this to verify everything is working correctly
"""

import os
import sys
import time

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def check_models():
    """Check if trained models exist"""
    print_header("CHECKING TRAINED MODELS")
    
    ml_models_dir = '../ml_models'
    
    # UPDATED: Correct file list based on your actual BERT folder structure
    required_files = [
        'rf_intent_classifier.pkl',
        'bert_intent_model/config.json',
        'bert_intent_model/model.safetensors',  # Changed from pytorch_model.bin
        'bert_intent_model/tokenizer_config.json',  # Added
        'bert_intent_model/vocab.txt',  # Added
        'bert_intent_model/special_tokens_map.json',  # Added
        'educational_intent_data.csv'
    ]
    
    all_exist = True
    for file in required_files:
        path = os.path.join(ml_models_dir, file)
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"{status} {file}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print("\n⚠ WARNING: Some model files are missing!")
        print("Please check your BERT model directory structure")
        return False
    
    print("\n✓ All model files found!")
    return True


def test_database():
    """Test database functionality"""
    print_header("TESTING DATABASE")
    
    try:
        from database import db, save_conversation, get_conversation_history
        
        # Test save
        print("Testing conversation save...")
        conversation = save_conversation(
            session_id='test_setup_session',
            query='What is machine learning?',
            intent='explanation',
            confidence=0.95,
            response='Machine learning is a subset of AI...',
            model_used='bert',
            response_time=0.5
        )
        print(f"✓ Saved conversation ID: {conversation.id}")
        
        # Test retrieve
        print("Testing conversation retrieval...")
        history = get_conversation_history('test_setup_session', limit=1)
        print(f"✓ Retrieved {len(history)} conversation(s)")
        
        print("\n✓ Database tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Database test failed: {e}")
        return False


def test_nlp_preprocessing():
    """Test NLP preprocessing"""
    print_header("TESTING NLP PREPROCESSING")
    
    try:
        from utils.nlp_preprocessing import NLPPreprocessor
        
        preprocessor = NLPPreprocessor()
        test_query = "What is machine learning and how does it work?"
        
        print(f"Test query: {test_query}")
        result = preprocessor.preprocess_query(test_query)
        
        print(f"✓ Cleaned text: {result['cleaned_text']}")
        print(f"✓ Keywords: {result['keywords']}")
        print(f"✓ Language: {result['language']}")
        
        print("\n✓ NLP preprocessing tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ NLP preprocessing test failed: {e}")
        return False


def test_intent_classifier():
    """Test intent classification"""
    print_header("TESTING INTENT CLASSIFIER")
    
    try:
        from models.intent_classifier import IntentClassifier
        
        classifier = IntentClassifier()
        
        test_queries = [
            "What is machine learning?",
            "How to implement a neural network?",
            "Show me an example"
        ]
        
        for query in test_queries:
            result = classifier.predict_intent(query)
            print(f"\nQuery: {query}")
            print(f"  Intent: {result['intent']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Model: {result['model']}")
        
        print("\n✓ Intent classifier tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Intent classifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_response_generator():
    """Test response generation"""
    print_header("TESTING RESPONSE GENERATOR")
    
    try:
        from models.response_generator import ResponseGenerator
        
        generator = ResponseGenerator()
        
        test_cases = [
            ("What is AI?", "explanation"),
            ("How to code?", "howto"),
            ("Show example", "example")
        ]
        
        for query, intent in test_cases:
            response_data = generator.generate_response(query, intent)
            print(f"\nQuery: {query} (Intent: {intent})")
            print(f"Response length: {len(response_data['response'])} chars")
            print(f"Suggestions: {len(response_data['suggestions'])}")
        
        print("\n✓ Response generator tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Response generator test failed: {e}")
        return False


def test_api_endpoints():
    """Test API endpoints"""
    print_header("TESTING API ENDPOINTS")
    
    try:
        import requests
        
        base_url = "http://localhost:5000"
        
        print("Testing health check...")
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✓ Health check passed")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
        
        print("\nTesting chat endpoint...")
        chat_data = {
            "query": "What is machine learning?",
            "session_id": "test_api_session"
        }
        response = requests.post(f"{base_url}/api/chat", json=chat_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Chat endpoint working")
            print(f"  Intent: {result['intent']}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Response time: {result['response_time']}s")
        else:
            print(f"✗ Chat endpoint failed: {response.status_code}")
            return False
        
        print("\n✓ API endpoint tests passed!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("\n⚠ Server not running. Start with: python app.py")
        print("Skipping API tests...")
        return None
    except Exception as e:
        print(f"\n✗ API test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print_header("AI TUTOR BACKEND - SETUP AND TESTING")
    
    results = []
    
    # Check models
    results.append(("Model Files", check_models()))
    
    # Test components (only if models exist)
    if results[0][1]:
        results.append(("Database", test_database()))
        results.append(("NLP Preprocessing", test_nlp_preprocessing()))
        results.append(("Intent Classifier", test_intent_classifier()))
        results.append(("Response Generator", test_response_generator()))
        results.append(("API Endpoints", test_api_endpoints()))
    
    # Print summary
    print_header("TEST SUMMARY")
    
    for test_name, result in results:
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⊘ SKIP"
        
        print(f"{status:8} - {test_name}")
    
    # Overall result
    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0 and passed > 0:
        print("\n🎉 All tests passed! Backend is ready to use!")
        print("\nNext steps:")
        print("1. Start the server: python app.py")
        print("2. Test API: http://localhost:5000")
        print("3. Continue with frontend development")
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")


if __name__ == "__main__":
    run_all_tests()