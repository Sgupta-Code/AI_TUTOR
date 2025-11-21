"""
NLP Preprocessing Pipeline for AI Tutor
Handles text cleaning, tokenization, and normalization
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
from langdetect import detect
from textblob import TextBlob

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class NLPPreprocessor:
    """Complete NLP preprocessing pipeline for educational queries"""
    
    def __init__(self, language='english'):
        self.language = language
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))
        
    def clean_text(self, text):
        """
        Clean and normalize input text
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep educational symbols
        text = re.sub(r'[^\w\s\+\-\=\*\/\(\)\[\]\{\}\?\!\.\,]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from token list
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Filtered tokens
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens):
        """
        Apply stemming to tokens
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens):
        """
        Apply lemmatization to tokens
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def detect_language(self, text):
        """
        Detect language of the text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Detected language code
        """
        try:
            return detect(text)
        except:
            return 'en'
    
    def extract_keywords(self, text, top_n=5):
        """
        Extract important keywords from text
        
        Args:
            text (str): Input text
            top_n (int): Number of top keywords to extract
            
        Returns:
            list: List of keywords
        """
        blob = TextBlob(text)
        word_freq = {}
        
        for word, tag in blob.tags:
            # Focus on nouns and verbs
            if tag.startswith('NN') or tag.startswith('VB'):
                word_freq[word.lower()] = word_freq.get(word.lower(), 0) + 1
        
        # Sort by frequency and return top N
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_n]]
    
    def preprocess_query(self, query, use_stemming=False, remove_stops=True):
        """
        Complete preprocessing pipeline for educational queries
        
        Args:
            query (str): User query
            use_stemming (bool): Whether to apply stemming
            remove_stops (bool): Whether to remove stopwords
            
        Returns:
            dict: Preprocessed query information
        """
        # Clean text
        cleaned_text = self.clean_text(query)
        
        # Tokenize
        tokens = self.tokenize(cleaned_text)
        
        # Remove stopwords if needed
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
        
        # Apply stemming or lemmatization
        if use_stemming:
            processed_tokens = self.stem_tokens(tokens)
        else:
            processed_tokens = self.lemmatize_tokens(tokens)
        
        # Detect language
        language = self.detect_language(query)
        
        # Extract keywords
        keywords = self.extract_keywords(cleaned_text)
        
        return {
            'original_query': query,
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'processed_tokens': processed_tokens,
            'language': language,
            'keywords': keywords,
            'token_count': len(tokens)
        }
    
    def extract_educational_intent(self, query):
        """
        Extract educational intent from query
        
        Args:
            query (str): User query
            
        Returns:
            dict: Intent information
        """
        query_lower = query.lower()
        
        # Educational intent patterns
        intent_patterns = {
            'explanation': ['explain', 'what is', 'define', 'describe', 'tell me about'],
            'example': ['example', 'show me', 'demonstrate', 'illustrate'],
            'howto': ['how to', 'how do', 'steps to', 'guide'],
            'comparison': ['difference between', 'compare', 'versus', 'vs'],
            'practice': ['practice', 'exercise', 'problem', 'question'],
            'summary': ['summarize', 'summary', 'brief', 'overview'],
            'clarification': ['confused', 'don\'t understand', 'unclear', 'doubt']
        }
        
        detected_intents = []
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    detected_intents.append(intent)
                    break
        
        return {
            'intents': detected_intents if detected_intents else ['general_query'],
            'primary_intent': detected_intents[0] if detected_intents else 'general_query'
        }


# Utility functions
def preprocess_text(text):
    """Quick preprocessing function"""
    preprocessor = NLPPreprocessor()
    return preprocessor.preprocess_query(text)


def extract_intent(query):
    """Quick intent extraction"""
    preprocessor = NLPPreprocessor()
    return preprocessor.extract_educational_intent(query)


# Testing function
if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = NLPPreprocessor()
    
    test_queries = [
        "What is machine learning and how does it work?",
        "Explain the difference between supervised and unsupervised learning",
        "Can you show me an example of a neural network?",
        "I'm confused about backpropagation, can you help?"
    ]
    
    print("=" * 60)
    print("NLP PREPROCESSOR TESTING")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nOriginal Query: {query}")
        
        # Preprocess
        result = preprocessor.preprocess_query(query)
        print(f"Cleaned Text: {result['cleaned_text']}")
        print(f"Keywords: {result['keywords']}")
        print(f"Language: {result['language']}")
        
        # Extract intent
        intent_result = preprocessor.extract_educational_intent(query)
        print(f"Intent: {intent_result['primary_intent']}")
        print("-" * 60)