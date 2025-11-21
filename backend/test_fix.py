#!/usr/bin/env python3
"""
Quick test to verify the fixes
"""

import sys
import os

# Add the backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_components():
    print("🧪 Testing AI Tutor Components...")
    
    try:
        # Test NLP Preprocessing
        from utils.nlp_preprocessing import NLPPreprocessor
        preprocessor = NLPPreprocessor()
        
        test_queries = [
            "What is machine learning?",
            "Tell me about Elon Musk",
            "How does Python work?",
            "Explain quantum computing"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing: {query}")
            
            # Preprocess
            preprocessed = preprocessor.preprocess_query(query)
            print(f"   Keywords: {preprocessed['keywords']}")
            
            # Test intent classification
            from models.intent_classifier import IntentClassifier
            classifier = IntentClassifier()
            intent_result = classifier.predict_intent(query)
            print(f"   Intent: {intent_result['intent']} (confidence: {intent_result['confidence']:.2f})")
            
            # Test response generation
            from models.response_generator import ResponseGenerator
            generator = ResponseGenerator()
            response_data = generator.generate_response(query, intent_result['intent'], preprocessed)
            
            print(f"   Response length: {len(response_data['response'])} chars")
            print(f"   Topic: {response_data.get('topic', 'N/A')}")
            print(f"   Suggestions: {len(response_data['suggestions'])}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_components()