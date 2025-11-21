"""
Intent Classifier Module
Loads and uses the trained Random Forest and BERT models
"""

import os
import pickle
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Hybrid intent classifier using both Random Forest and BERT
    """
    
    def __init__(self, model_path='../ml_models'):
        """
        Initialize the intent classifier with trained models
        
        Args:
            model_path (str): Path to the ml_models directory
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self.rf_model = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.label_encoder = None
        
        self._load_models()
    
    def _load_models(self):
        """Load Random Forest and BERT models"""
        try:
            # Load Random Forest model
            rf_path = os.path.join(self.model_path, 'rf_intent_classifier.pkl')
            logger.info(f"Loading Random Forest model from {rf_path}")
            
            if not os.path.exists(rf_path):
                raise FileNotFoundError(f"Random Forest model not found at {rf_path}")
                
            with open(rf_path, 'rb') as f:
                rf_data = pickle.load(f)
                self.rf_model = rf_data['model']
                self.vectorizer = rf_data['vectorizer']
                self.label_encoder = rf_data['label_encoder']
            
            logger.info("✓ Random Forest model loaded successfully")
            
            # Load BERT model
            bert_path = os.path.join(self.model_path, 'bert_intent_model')
            logger.info(f"Loading BERT model from {bert_path}")
            
            if not os.path.exists(bert_path):
                raise FileNotFoundError(f"BERT model directory not found at {bert_path}")
            
            # Check for required BERT files
            required_files = ['config.json', 'model.safetensors', 'tokenizer_config.json', 'vocab.txt']
            for file in required_files:
                if not os.path.exists(os.path.join(bert_path, file)):
                    logger.warning(f"Missing BERT file: {file}")
            
            # Load tokenizer and model
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
            
            # Load model - this should automatically handle .safetensors files
            self.bert_model = BertForSequenceClassification.from_pretrained(
                bert_path,
                local_files_only=True
            )
            self.bert_model.to(self.device)
            self.bert_model.eval()
            
            # Load BERT label encoder
            bert_label_path = os.path.join(bert_path, 'label_encoder.pkl')
            if os.path.exists(bert_label_path):
                with open(bert_label_path, 'rb') as f:
                    self.bert_label_encoder = pickle.load(f)
            else:
                logger.warning("BERT label encoder not found, using RF label encoder")
                self.bert_label_encoder = self.label_encoder
            
            logger.info("✓ BERT model loaded successfully")
            logger.info(f"✓ Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Don't raise immediately, try to continue with available models
            if self.rf_model is None:
                raise
    
    def predict_intent_rf(self, query):
        """
        Predict intent using Random Forest
        
        Args:
            query (str): User query
            
        Returns:
            dict: Intent prediction with confidence
        """
        try:
            if self.rf_model is None:
                raise ValueError("Random Forest model not loaded")
                
            # Vectorize query
            query_vec = self.vectorizer.transform([query])
            
            # Predict
            prediction = self.rf_model.predict(query_vec)
            probabilities = self.rf_model.predict_proba(query_vec)[0]
            
            # Get intent and confidence
            intent = self.label_encoder.inverse_transform(prediction)[0]
            confidence = max(probabilities)
            
            # Get top 3 predictions
            top_indices = probabilities.argsort()[-3:][::-1]
            top_predictions = [
                {
                    'intent': self.label_encoder.inverse_transform([idx])[0],
                    'confidence': float(probabilities[idx])
                }
                for idx in top_indices
            ]
            
            return {
                'intent': intent,
                'confidence': float(confidence),
                'model': 'random_forest',
                'top_predictions': top_predictions
            }
            
        except Exception as e:
            logger.error(f"Error in Random Forest prediction: {e}")
            return {
                'intent': 'general_query',
                'confidence': 0.5,
                'model': 'random_forest',
                'error': str(e)
            }
    
    def predict_intent_bert(self, query):
        """
        Predict intent using BERT
        
        Args:
            query (str): User query
            
        Returns:
            dict: Intent prediction with confidence
        """
        try:
            if self.bert_model is None:
                raise ValueError("BERT model not loaded")
                
            # Tokenize
            encoding = self.bert_tokenizer(
                query,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.bert_model(**encoding)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
                confidence, predicted_class = torch.max(predictions, dim=1)
            
            # Get intent
            intent = self.bert_label_encoder.inverse_transform(
                [predicted_class.cpu().item()]
            )[0]
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(predictions, k=min(3, predictions.shape[1]))
            top_predictions = [
                {
                    'intent': self.bert_label_encoder.inverse_transform([idx.cpu().item()])[0],
                    'confidence': float(prob.cpu().item())
                }
                for prob, idx in zip(top_probs[0], top_indices[0])
            ]
            
            return {
                'intent': intent,
                'confidence': float(confidence.cpu().item()),
                'model': 'bert',
                'top_predictions': top_predictions
            }
            
        except Exception as e:
            logger.error(f"Error in BERT prediction: {e}")
            # Fall back to Random Forest if BERT fails
            logger.info("Falling back to Random Forest due to BERT error")
            return self.predict_intent_rf(query)
    
    def predict_intent(self, query, use_bert=True):
        """
        Predict intent using hybrid approach
        
        Args:
            query (str): User query
            use_bert (bool): Whether to use BERT (slower but more accurate)
            
        Returns:
            dict: Intent prediction with confidence
        """
        try:
            query_lower = query.lower()
            follow_up_patterns = {
            'example': ['show me example', 'give me example', 'can i see example', 'examples'],
            'howto': ['how to', 'how do i', 'steps to', 'guide me'],
            'practice': ['practice problem', 'exercise', 'problem', 'quiz', 'test me'],
            'explanation': ['explain', 'what is', 'tell me about', 'describe']
            }
            for intent, patterns in follow_up_patterns.items():
                for pattern in patterns:
                    if pattern in query_lower:
                        logger.info(f"Detected follow-up intent: {intent} for query: {query}")
                        return {
                            'intent': intent,
                            'confidence': 0.95,
                            'model': 'context_aware',
                            'note': 'Detected follow-up intent based on keywords'
                        }
                    
            if use_bert and self.bert_model is not None:
                # Use BERT for higher accuracy
                result = self.predict_intent_bert(query)
                
                # If confidence is low, also check Random Forest
                if result['confidence'] < 0.7 and self.rf_model is not None:
                    rf_result = self.predict_intent_rf(query)
                    
                    # Use RF if it has higher confidence
                    if rf_result['confidence'] > result['confidence']:
                        result = rf_result
                        result['note'] = 'Used Random Forest due to low BERT confidence'
            elif self.rf_model is not None:
                # Use Random Forest for speed or if BERT not available
                result = self.predict_intent_rf(query)
            else:
                # Fallback if no models are available
                result = {
                    'intent': 'general_query',
                    'confidence': 0.5,
                    'model': 'fallback',
                    'note': 'No models available, using fallback'
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in intent prediction: {e}")
            return {
                'intent': 'general_query',
                'confidence': 0.5,
                'error': str(e)
            }
    
    def get_intent_description(self, intent):
        """
        Get human-readable description of an intent
        
        Args:
            intent (str): Intent label
            
        Returns:
            str: Description of the intent
        """
        descriptions = {
            'explanation': 'User wants an explanation or definition',
            'howto': 'User wants step-by-step instructions',
            'example': 'User wants to see examples',
            'comparison': 'User wants to compare concepts',
            'calculation': 'User wants numerical calculations',
            'temporal': 'User wants temporal/time information',
            'location': 'User wants location information',
            'person': 'User wants information about people',
            'general_query': 'General educational question'
        }
        
        return descriptions.get(intent, 'Educational query')


# Test function
if __name__ == "__main__":
    # Test the classifier
    try:
        classifier = IntentClassifier()
        
        test_queries = [
            "What is machine learning?",
            "How to implement a neural network?",
            "Show me an example of supervised learning",
            "What's the difference between AI and ML?"
        ]
        
        print("="*70)
        print("INTENT CLASSIFIER TESTING")
        print("="*70)
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            
            # Hybrid prediction
            result = classifier.predict_intent(query)
            print(f"Result: {result['intent']} (conf: {result['confidence']:.3f}, model: {result['model']})")
            
            # Show top predictions if available
            if 'top_predictions' in result:
                print("Top predictions:")
                for pred in result['top_predictions']:
                    print(f"  - {pred['intent']}: {pred['confidence']:.3f}")
                    
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()