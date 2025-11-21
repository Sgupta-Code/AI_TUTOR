"""
Knowledge Base Management System for AI Tutor
Manages educational content with auto-learning capabilities
"""

import json
import os
import logging
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    Manages educational content and provides intelligent retrieval
    Features:
    - Pre-loaded educational topics
    - Search by keywords
    - Auto-learning from conversations
    - Content recommendations
    """
    
    def __init__(self, data_path='data/educational_content.json', enable_web_research = True):
        self.data_path = data_path
        self.enable_web_research = enable_web_research
        self.knowledge = self._load_or_create_knowledge()
        self.search_cache = {}  # Cache for faster searches
        self.access_stats = defaultdict(int)  # Track popular topics
        
        if self.enable_web_research:
            try:
                from web_researcher import WebResearcher
                self.web_researcher = WebResearcher()
                logger.info("Web research enabled")

            except ImportError:
                logger.warning("WebResearcher module not found. Web research disabled.")
                self.enable_web_research = False
        logger.info("KnowledgeBase initialized")
    
    def _load_or_create_knowledge(self):
        """Load knowledge from file or create default"""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    knowledge = json.load(f)
                    logger.info(f"Loaded knowledge base from {self.data_path}")
                    return knowledge
            except Exception as e:
                logger.error(f"Error loading knowledge base: {e}")
                logger.info("Creating default knowledge base...")
        
        # Create default knowledge base
        return self._create_default_knowledge()
    
    def _create_default_knowledge(self):
        """Create comprehensive default educational knowledge base"""
        return {
            "machine_learning": {
                "title": "Machine Learning",
                "category": "AI/ML",
                "definition": "Machine Learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms and statistical models to analyze and draw inferences from patterns in data.",
                "key_concepts": [
                    "Supervised Learning",
                    "Unsupervised Learning",
                    "Reinforcement Learning",
                    "Deep Learning",
                    "Feature Engineering",
                    "Model Training",
                    "Overfitting and Underfitting"
                ],
                "examples": [
                    {
                        "name": "Linear Regression",
                        "description": "A supervised learning algorithm for predicting continuous values based on input features",
                        "code": "from sklearn.linear_model import LinearRegression\nimport numpy as np\n\n# Create sample data\nX = np.array([[1], [2], [3], [4], [5]])\ny = np.array([2, 4, 5, 4, 5])\n\n# Train model\nmodel = LinearRegression()\nmodel.fit(X, y)\n\n# Make predictions\npredictions = model.predict([[6], [7]])\nprint(predictions)",
                        "difficulty": "beginner"
                    },
                    {
                        "name": "Decision Tree Classification",
                        "description": "A tree-based model for making decisions based on asking a series of questions",
                        "code": "from sklearn.tree import DecisionTreeClassifier\nfrom sklearn.datasets import load_iris\n\n# Load data\niris = load_iris()\nX, y = iris.data, iris.target\n\n# Train classifier\nclf = DecisionTreeClassifier(max_depth=3)\nclf.fit(X, y)\n\n# Predict\npredictions = clf.predict(X[:5])\nprint(predictions)",
                        "difficulty": "intermediate"
                    }
                ],
                "practice_problems": [
                    {
                        "question": "What is the main difference between supervised and unsupervised learning?",
                        "answer": "Supervised learning uses labeled data with known outputs to train models, while unsupervised learning finds patterns in unlabeled data without predefined outcomes.",
                        "difficulty": "easy"
                    },
                    {
                        "question": "Implement a simple linear regression model using scikit-learn to predict house prices",
                        "answer": "Use LinearRegression() class, prepare your data with features (X) and target (y), fit the model with model.fit(X, y), and make predictions with model.predict(X_new)",
                        "difficulty": "medium"
                    }
                ],
                "resources": [
                    "https://scikit-learn.org/stable/tutorial/index.html",
                    "https://www.coursera.org/learn/machine-learning",
                    "https://www.kaggle.com/learn/intro-to-machine-learning"
                ],
                "prerequisites": ["Python Programming", "Basic Statistics", "Linear Algebra"],
                "related_topics": ["deep_learning", "neural_networks", "data_science"],
                "last_updated": datetime.now().isoformat()
            },
            
            
            "neural_networks": {
                "title": "Neural Networks",
                "category": "AI/ML",
                "definition": "Neural Networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that can learn to recognize patterns and make decisions based on input data.",
                "key_concepts": [
                    "Perceptron",
                    "Layers (Input, Hidden, Output)",
                    "Weights and Biases",
                    "Forward Propagation",
                    "Backpropagation",
                    "Gradient Descent",
                    "Activation Functions"
                ],
                "examples": [
                    {
                        "name": "Simple Perceptron",
                        "description": "The simplest form of a neural network with one layer",
                        "code": "import numpy as np\n\nclass Perceptron:\n    def __init__(self, input_size, learning_rate=0.01):\n        self.weights = np.random.randn(input_size)\n        self.bias = 0\n        self.lr = learning_rate\n    \n    def predict(self, x):\n        return 1 if np.dot(x, self.weights) + self.bias > 0 else 0\n    \n    def train(self, X, y, epochs=100):\n        for _ in range(epochs):\n            for xi, yi in zip(X, y):\n                prediction = self.predict(xi)\n                error = yi - prediction\n                self.weights += self.lr * error * xi\n                self.bias += self.lr * error\n\n# Usage\nperceptron = Perceptron(input_size=2)\n# perceptron.train(X_train, y_train)",
                        "difficulty": "intermediate"
                    },
                    {
                        "name": "Multi-Layer Neural Network",
                        "description": "A complete neural network with multiple layers",
                        "code": "import numpy as np\n\nclass NeuralNetwork:\n    def __init__(self, layers):\n        self.weights = []\n        self.biases = []\n        for i in range(len(layers)-1):\n            self.weights.append(np.random.randn(layers[i], layers[i+1]))\n            self.biases.append(np.zeros((1, layers[i+1])))\n    \n    def sigmoid(self, x):\n        return 1 / (1 + np.exp(-x))\n    \n    def forward(self, X):\n        self.activations = [X]\n        for w, b in zip(self.weights, self.biases):\n            X = self.sigmoid(np.dot(X, w) + b)\n            self.activations.append(X)\n        return X\n\n# Create a 3-layer network\nnn = NeuralNetwork([2, 4, 1])",
                        "difficulty": "advanced"
                    }
                ],
                "practice_problems": [
                    {
                        "question": "Explain backpropagation in simple terms",
                        "answer": "Backpropagation is the process of calculating gradients by propagating errors backward through the network layers. It computes how much each weight contributed to the error and adjusts weights to minimize future errors using the chain rule of calculus.",
                        "difficulty": "medium"
                    },
                    {
                        "question": "What is the vanishing gradient problem?",
                        "answer": "The vanishing gradient problem occurs when gradients become extremely small during backpropagation in deep networks, making it difficult for early layers to learn. This is often caused by activation functions like sigmoid or tanh.",
                        "difficulty": "hard"
                    }
                ],
                "resources": [
                    "http://neuralnetworksanddeeplearning.com/",
                    "https://playground.tensorflow.org/",
                    "https://www.3blue1brown.com/topics/neural-networks"
                ],
                "prerequisites": ["Basic Calculus", "Linear Algebra", "Python Programming"],
                "related_topics": ["deep_learning", "machine_learning", "backpropagation"],
                "last_updated": datetime.now().isoformat()
            },
            
            "python_programming": {
                "title": "Python Programming",
                "category": "Programming",
                "definition": "Python is a high-level, interpreted programming language known for its clear syntax and readability. It's widely used in data science, machine learning, web development, automation, and scientific computing.",
                "key_concepts": [
                    "Variables and Data Types",
                    "Control Flow (if, for, while)",
                    "Functions and Modules",
                    "Lists and Dictionaries",
                    "Object-Oriented Programming",
                    "File Handling",
                    "Exception Handling"
                ],
                "examples": [
                    {
                        "name": "Basic Python Functions",
                        "description": "Creating and using functions in Python",
                        "code": "# Define a function\ndef greet(name):\n    return f'Hello, {name}!'\n\n# Function with default parameters\ndef calculate_area(length, width=10):\n    return length * width\n\n# Lambda function\nsquare = lambda x: x ** 2\n\n# Usage\nprint(greet('Alice'))  # Hello, Alice!\nprint(calculate_area(5))  # 50\nprint(square(4))  # 16",
                        "difficulty": "beginner"
                    },
                    {
                        "name": "List Comprehensions",
                        "description": "Efficient way to create lists in Python",
                        "code": "# Basic list comprehension\nsquares = [x**2 for x in range(10)]\n\n# With condition\neven_squares = [x**2 for x in range(10) if x % 2 == 0]\n\n# Nested comprehension\nmatrix = [[i+j for j in range(3)] for i in range(3)]\n\n# Dictionary comprehension\nsquare_dict = {x: x**2 for x in range(5)}\n\nprint(squares)  # [0, 1, 4, 9, 16, ...]\nprint(even_squares)  # [0, 4, 16, 36, 64]",
                        "difficulty": "intermediate"
                    },
                    {
                        "name": "Classes and OOP",
                        "description": "Object-oriented programming in Python",
                        "code": "class Student:\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age\n        self.grades = []\n    \n    def add_grade(self, grade):\n        self.grades.append(grade)\n    \n    def get_average(self):\n        return sum(self.grades) / len(self.grades) if self.grades else 0\n    \n    def __str__(self):\n        return f'{self.name}, Age: {self.age}'\n\n# Usage\nstudent = Student('Alice', 20)\nstudent.add_grade(85)\nstudent.add_grade(90)\nprint(student)  # Alice, Age: 20\nprint(f'Average: {student.get_average()}')  # Average: 87.5",
                        "difficulty": "intermediate"
                    }
                ],
                "practice_problems": [
                    {
                        "question": "Write a function that takes a list of numbers and returns the sum of even numbers",
                        "answer": "def sum_even(numbers):\n    return sum(n for n in numbers if n % 2 == 0)\n\n# Or using filter:\ndef sum_even(numbers):\n    return sum(filter(lambda x: x % 2 == 0, numbers))",
                        "difficulty": "easy"
                    },
                    {
                        "question": "Create a class called 'BankAccount' with deposit and withdraw methods",
                        "answer": "class BankAccount:\n    def __init__(self, balance=0):\n        self.balance = balance\n    \n    def deposit(self, amount):\n        self.balance += amount\n        return self.balance\n    \n    def withdraw(self, amount):\n        if amount <= self.balance:\n            self.balance -= amount\n            return self.balance\n        return 'Insufficient funds'",
                        "difficulty": "medium"
                    }
                ],
                "resources": [
                    "https://docs.python.org/3/tutorial/",
                    "https://realpython.com/",
                    "https://www.learnpython.org/",
                    "https://www.pythoncheatsheet.org/"
                ],
                "prerequisites": ["Basic Computer Knowledge"],
                "related_topics": ["data_science", "machine_learning", "web_development"],
                "last_updated": datetime.now().isoformat()
            },
            
            "data_science": {
                "title": "Data Science",
                "category": "Data",
                "definition": "Data Science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It combines statistics, data analysis, machine learning, and domain expertise.",
                "key_concepts": [
                    "Data Collection and Cleaning",
                    "Exploratory Data Analysis (EDA)",
                    "Statistical Analysis",
                    "Data Visualization",
                    "Machine Learning",
                    "Feature Engineering",
                    "Big Data Technologies"
                ],
                "examples": [
                    {
                        "name": "Data Analysis with Pandas",
                        "description": "Basic data manipulation and analysis using pandas",
                        "code": "import pandas as pd\nimport numpy as np\n\n# Create DataFrame\ndata = {\n    'name': ['Alice', 'Bob', 'Charlie', 'David'],\n    'age': [25, 30, 35, 28],\n    'score': [85, 90, 95, 88],\n    'city': ['NY', 'LA', 'NY', 'LA']\n}\ndf = pd.DataFrame(data)\n\n# Basic operations\nprint(df.describe())  # Statistical summary\nprint(df[df['age'] > 25])  # Filter rows\nprint(df.groupby('city')['score'].mean())  # Group by\n\n# Add new column\ndf['grade'] = df['score'].apply(lambda x: 'A' if x >= 90 else 'B')",
                        "difficulty": "beginner"
                    },
                    {
                        "name": "Data Visualization with Matplotlib",
                        "description": "Creating visualizations for data analysis",
                        "code": "import matplotlib.pyplot as plt\nimport pandas as pd\n\n# Sample data\ndata = pd.DataFrame({\n    'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],\n    'sales': [100, 150, 200, 180, 250]\n})\n\n# Line plot\nplt.figure(figsize=(10, 6))\nplt.plot(data['month'], data['sales'], marker='o')\nplt.title('Monthly Sales')\nplt.xlabel('Month')\nplt.ylabel('Sales ($)')\nplt.grid(True)\nplt.show()\n\n# Bar plot\nplt.bar(data['month'], data['sales'])\nplt.title('Sales by Month')\nplt.show()",
                        "difficulty": "intermediate"
                    },
                    {
                        "name": "Statistical Analysis",
                        "description": "Performing statistical tests and analysis",
                        "code": "import numpy as np\nfrom scipy import stats\n\n# Sample data\ngroup1 = np.random.normal(100, 15, 50)\ngroup2 = np.random.normal(110, 15, 50)\n\n# Descriptive statistics\nprint(f'Mean: {np.mean(group1):.2f}')\nprint(f'Median: {np.median(group1):.2f}')\nprint(f'Std Dev: {np.std(group1):.2f}')\n\n# T-test\nt_stat, p_value = stats.ttest_ind(group1, group2)\nprint(f'T-statistic: {t_stat:.2f}')\nprint(f'P-value: {p_value:.4f}')\n\n# Correlation\ncorrelation = np.corrcoef(group1, group2)[0, 1]\nprint(f'Correlation: {correlation:.2f}')",
                        "difficulty": "advanced"
                    }
                ],
                "practice_problems": [
                    {
                        "question": "What is the difference between mean, median, and mode?",
                        "answer": "Mean is the average of all values (sum/count). Median is the middle value when data is sorted (50th percentile). Mode is the most frequently occurring value. Mean is sensitive to outliers, while median is more robust.",
                        "difficulty": "easy"
                    },
                    {
                        "question": "How do you handle missing data in a dataset?",
                        "answer": "Common approaches: 1) Remove rows with missing values (if small percentage), 2) Fill with mean/median/mode, 3) Forward/backward fill for time series, 4) Use interpolation, 5) Predict missing values using ML models. Choice depends on data type and amount of missing data.",
                        "difficulty": "medium"
                    },
                    {
                        "question": "Explain the concept of data normalization and why it's important",
                        "answer": "Data normalization scales features to a common range (e.g., 0-1 or standardized). It's important because: 1) Algorithms converge faster, 2) Prevents features with large ranges from dominating, 3) Improves model performance, 4) Required for distance-based algorithms like KNN.",
                        "difficulty": "medium"
                    }
                ],
                "resources": [
                    "https://www.kaggle.com/learn/intro-to-data-science",
                    "https://pandas.pydata.org/docs/getting_started/tutorials.html",
                    "https://www.datacamp.com/courses/intro-to-python-for-data-science",
                    "https://towardsdatascience.com/"
                ],
                "prerequisites": ["Python Programming", "Basic Statistics", "Mathematics"],
                "related_topics": ["machine_learning", "statistics", "data_visualization"],
                "last_updated": datetime.now().isoformat()
            },
            
            "natural_language_processing": {
                "title": "Natural Language Processing",
                "category": "AI/ML",
                "definition": "Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on the interaction between computers and humans through natural language. It enables computers to understand, interpret, and generate human language in a valuable way.",
                "key_concepts": [
                    "Tokenization",
                    "Text Preprocessing",
                    "Word Embeddings",
                    "Named Entity Recognition (NER)",
                    "Sentiment Analysis",
                    "Language Models",
                    "Transformers and BERT"
                ],
                "examples": [
                    {
                        "name": "Text Preprocessing",
                        "description": "Basic text cleaning and preprocessing",
                        "code": "import re\nimport nltk\nfrom nltk.corpus import stopwords\nfrom nltk.tokenize import word_tokenize\n\n# Download required data\nnltk.download('punkt')\nnltk.download('stopwords')\n\ntext = \"This is an EXAMPLE sentence, showing text preprocessing!\"\n\n# Convert to lowercase\ntext = text.lower()\n\n# Remove punctuation\ntext = re.sub(r'[^\\w\\s]', '', text)\n\n# Tokenize\ntokens = word_tokenize(text)\n\n# Remove stopwords\nstop_words = set(stopwords.words('english'))\nfiltered_tokens = [w for w in tokens if w not in stop_words]\n\nprint(filtered_tokens)  # ['example', 'sentence', 'showing', 'text', 'preprocessing']",
                        "difficulty": "beginner"
                    },
                    {
                        "name": "Sentiment Analysis",
                        "description": "Analyzing sentiment of text using transformers",
                        "code": "from transformers import pipeline\n\n# Load sentiment analysis pipeline\nsentiment_analyzer = pipeline('sentiment-analysis')\n\n# Analyze sentiment\ntexts = [\n    \"I love this product! It's amazing!\",\n    \"This is terrible, worst purchase ever.\",\n    \"It's okay, nothing special.\"\n]\n\nfor text in texts:\n    result = sentiment_analyzer(text)[0]\n    print(f\"Text: {text}\")\n    print(f\"Sentiment: {result['label']}, Score: {result['score']:.2f}\\n\")",
                        "difficulty": "intermediate"
                    },
                    {
                        "name": "Named Entity Recognition",
                        "description": "Extracting entities from text",
                        "code": "import spacy\n\n# Load spacy model\nnlp = spacy.load('en_core_web_sm')\n\ntext = \"Apple Inc. was founded by Steve Jobs in Cupertino, California.\"\n\n# Process text\ndoc = nlp(text)\n\n# Extract entities\nfor ent in doc.ents:\n    print(f\"{ent.text}: {ent.label_}\")\n\n# Output:\n# Apple Inc.: ORG\n# Steve Jobs: PERSON\n# Cupertino: GPE\n# California: GPE",
                        "difficulty": "advanced"
                    }
                ],
                "practice_problems": [
                    {
                        "question": "What is tokenization and why is it important in NLP?",
                        "answer": "Tokenization is the process of breaking text into smaller units (tokens) like words or sentences. It's important because it's the first step in text processing, converting unstructured text into structured data that algorithms can process.",
                        "difficulty": "easy"
                    },
                    {
                        "question": "Explain the difference between word embeddings and one-hot encoding",
                        "answer": "One-hot encoding represents words as sparse vectors with only one '1' and rest '0's, losing semantic relationships. Word embeddings (like Word2Vec, GloVe) represent words as dense vectors that capture semantic meaning, allowing similar words to have similar representations.",
                        "difficulty": "medium"
                    }
                ],
                "resources": [
                    "https://huggingface.co/course/chapter1",
                    "https://www.nltk.org/book/",
                    "https://spacy.io/usage/spacy-101",
                    "https://www.tensorflow.org/tutorials/text/word_embeddings"
                ],
                "prerequisites": ["Python Programming", "Machine Learning", "Basic Linguistics"],
                "related_topics": ["machine_learning", "deep_learning", "transformers"],
                "last_updated": datetime.now().isoformat()
            },
            
            "algorithms": {
                "title": "Algorithms and Data Structures",
                "category": "Computer Science",
                "definition": "Algorithms are step-by-step procedures for solving problems, while data structures are ways of organizing and storing data efficiently. Together, they form the foundation of computer science and software engineering.",
                "key_concepts": [
                    "Time and Space Complexity",
                    "Sorting Algorithms",
                    "Searching Algorithms",
                    "Graph Algorithms",
                    "Dynamic Programming",
                    "Arrays and Linked Lists",
                    "Trees and Graphs"
                ],
                "examples": [
                    {
                        "name": "Binary Search",
                        "description": "Efficient searching in sorted arrays",
                        "code": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    \n    while left <= right:\n        mid = (left + right) // 2\n        \n        if arr[mid] == target:\n            return mid  # Found\n        elif arr[mid] < target:\n            left = mid + 1  # Search right half\n        else:\n            right = mid - 1  # Search left half\n    \n    return -1  # Not found\n\n# Usage\narr = [1, 3, 5, 7, 9, 11, 13]\nresult = binary_search(arr, 7)\nprint(f'Found at index: {result}')  # Found at index: 3",
                        "difficulty": "intermediate"
                    },
                    {
                        "name": "Quick Sort",
                        "description": "Efficient divide-and-conquer sorting algorithm",
                        "code": "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    \n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    \n    return quick_sort(left) + middle + quick_sort(right)\n\n# Usage\narr = [3, 6, 8, 10, 1, 2, 1]\nsorted_arr = quick_sort(arr)\nprint(sorted_arr)  # [1, 1, 2, 3, 6, 8, 10]",
                        "difficulty": "intermediate"
                    },
                    {
                        "name": "Depth-First Search (DFS)",
                        "description": "Graph traversal algorithm",
                        "code": "def dfs(graph, start, visited=None):\n    if visited is None:\n        visited = set()\n    \n    visited.add(start)\n    print(start, end=' ')\n    \n    for neighbor in graph[start]:\n        if neighbor not in visited:\n            dfs(graph, neighbor, visited)\n    \n    return visited\n\n# Usage\ngraph = {\n    'A': ['B', 'C'],\n    'B': ['D', 'E'],\n    'C': ['F'],\n    'D': [],\n    'E': ['F'],\n    'F': []\n}\n\ndfs(graph, 'A')  # Output: A B D E F C",
                        "difficulty": "advanced"
                    }
                ],
                "practice_problems": [
                    {
                        "question": "What is Big O notation and why is it important?",
                        "answer": "Big O notation describes the upper bound of an algorithm's time or space complexity as input size grows. It's important for comparing algorithm efficiency and predicting performance. Common complexities: O(1) constant, O(log n) logarithmic, O(n) linear, O(n²) quadratic.",
                        "difficulty": "medium"
                    },
                    {
                        "question": "Implement a function to reverse a linked list",
                        "answer": "def reverse_linked_list(head):\n    prev = None\n    current = head\n    while current:\n        next_node = current.next\n        current.next = prev\n        prev = current\n        current = next_node\n    return prev",
                        "difficulty": "medium"
                    }
                ],
                "resources": [
                    "https://www.geeksforgeeks.org/fundamentals-of-algorithms/",
                    "https://leetcode.com/",
                    "https://www.hackerrank.com/domains/algorithms",
                    "https://visualgo.net/en"
                ],
                "prerequisites": ["Python Programming", "Basic Mathematics"],
                "related_topics": ["programming", "problem_solving", "computer_science"],
                "last_updated": datetime.now().isoformat()
            },
        
            "deep_learning": {
                "title": "Deep Learning",
                "category": "AI/ML",
                "definition": "Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers (deep neural networks) to model and understand complex patterns in data. It's particularly effective for tasks like image recognition, natural language processing, and speech recognition.",
            "key_concepts": [
                "Neural Networks",
                "Deep Neural Networks",
                "Convolutional Neural Networks (CNN)",
                "Recurrent Neural Networks (RNN)",
                "Backpropagation",
                "Activation Functions",
                "Gradient Descent"
            ],
            "examples": [
                {
                    "name": "Simple Neural Network with Keras",
                    "description": "Building a basic neural network for classification",
                    "code": "import tensorflow as tf\nfrom tensorflow import keras\n\n# Create a sequential model\nmodel = keras.Sequential([\n    keras.layers.Dense(128, activation='relu', input_shape=(784,)),\n    keras.layers.Dense(64, activation='relu'),\n    keras.layers.Dense(10, activation='softmax')\n])\n\n# Compile the model\nmodel.compile(optimizer='adam',\n              loss='sparse_categorical_crossentropy',\n              metrics=['accuracy'])\n\n# Train the model\n# model.fit(X_train, y_train, epochs=10)",
                    "difficulty": "intermediate"
                }
            ],
            "practice_problems": [
                {
                    "question": "What is the difference between machine learning and deep learning?",
                    "answer": "Machine learning uses algorithms to parse data, learn from it, and make decisions. Deep learning structures algorithms in layers to create an 'artificial neural network' that can learn and make intelligent decisions on its own.",
                    "difficulty": "easy"
                }
            ],
            "resources": [
                "https://www.tensorflow.org/tutorials",
                "https://pytorch.org/tutorials/",
                "https://www.deeplearning.ai/"
            ],
            "prerequisites": ["Machine Learning", "Python Programming", "Linear Algebra"],
            "related_topics": ["machine_learning", "neural_networks", "computer_vision"],
            "last_updated": datetime.now().isoformat()
            }
        }
    
    def save_knowledge(self):
        """Save knowledge base to file"""
        try:
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge, f, indent=2, ensure_ascii=False)
            logger.info(f"Knowledge base saved to {self.data_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
            return False
    
    def search_topic(self, keywords):
        """Enhanced topic search with better matching"""
        if not keywords:
            return None
    
        # Clean and normalize keywords
        cleaned_keywords = []
        for kw in keywords:
            if isinstance(kw, str):
            # Remove common question words and clean
                kw_clean = kw.lower().strip()
                if len(kw_clean) > 2 and kw_clean not in ['what', 'how', 'why', 'when', 'where', 'who']:
                    cleaned_keywords.append(kw_clean)
    
        if not cleaned_keywords:
            return None
    
        # Create cache key
        cache_key = '-'.join(sorted(cleaned_keywords))
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
    
        # First, try exact topic matches
        best_match = self._search_exact_match(cleaned_keywords)
    
        # If no exact match, try broader search
        if not best_match:
            best_match = self._search_broad_match(cleaned_keywords)
    
        # If still no match and research enabled, try web research             
        # Cache result
        if best_match:
            self.search_cache[cache_key] = best_match
            self.access_stats[best_match['title']] += 1
        
        return best_match
    
    def _search_exact_match(self, keywords):
        """Search for exact topic matches"""
        # Try direct topic key match
        for kw in keywords:
            topic_key = '_'.join(keywords)
            if topic_key in self.knowledge:
                return self.knowledge[topic_key]
        
        # Try title exact match
        for key, data in self.knowledge.items():
            title_lower = data['title'].lower()
            for kw in keywords:
                if kw in title_lower:
                    return data
        
        return None
    
    def _search_broad_match(self, keywords):
        """Search with broader matching"""
        best_match = None
        best_score = 0
        
        for topic_key, topic_data in self.knowledge.items():
            score = 0
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                
                # High score for title matches
                if keyword_lower in topic_data['title'].lower():
                    score += 5
                
                # Medium score for topic key matches
                if keyword_lower in topic_key:
                    score += 3
                
                # Lower score for definition matches
                if keyword_lower in topic_data['definition'].lower():
                    score += 1
                
                # Score for key concepts
                for concept in topic_data.get('key_concepts', []):
                    if keyword_lower in concept.lower():
                        score += 2
                        break
                    
            if score > best_score:
                best_score = score
                best_match = topic_data
        
        return best_match if best_score > 0 else None

    def _search_existing_knowledge(self, keywords):
    
        best_match = None
        best_score = 0
    
        for topic_key, topic_data in self.knowledge.items():
            score = 0
        
            for keyword in keywords:
                keyword_lower = keyword.lower()
            
                if keyword_lower in topic_key:
                    score += 3
            
                if keyword_lower in topic_data['title'].lower():
                    score += 3
            
                if keyword_lower in topic_data['definition'].lower():
                    score += 1
            
                for concept in topic_data.get('key_concepts', []):
                    if keyword_lower in concept.lower():
                        score += 2
                        break
        
            if score > best_score:
                best_score = score
                best_match = topic_data
    
        return best_match
    
    def get_topic_by_name(self, topic_name):
        """
        Get topic by exact name or key
        
        Args:
            topic_name (str): Topic name or key
            
        Returns:
            dict: Topic data or None
        """
        # Try direct key lookup
        topic_key = topic_name.lower().replace(' ', '_')
        if topic_key in self.knowledge:
            return self.knowledge[topic_key]
        
        # Try title match
        for key, data in self.knowledge.items():
            if data['title'].lower() == topic_name.lower():
                return data
        
        return None
    
    def get_all_topics(self):
        """
        Get list of all available topics with metadata
        
        Returns:
            list: List of topic summaries
        """
        topics = []
        for key, data in self.knowledge.items():
            topics.append({
                'key': key,
                'title': data['title'],
                'category': data.get('category', 'General'),
                'description': data['definition'][:150] + '...' if len(data['definition']) > 150 else data['definition'],
                'examples_count': len(data.get('examples', [])),
                'practice_count': len(data.get('practice_problems', [])),
                'difficulty': self._estimate_difficulty(data)
            })
        
        return topics
    
    def _estimate_difficulty(self, topic_data):
        """Estimate overall difficulty of a topic"""
        prereqs = len(topic_data.get('prerequisites', []))
        if prereqs == 0:
            return 'beginner'
        elif prereqs <= 2:
            return 'intermediate'
        else:
            return 'advanced'
    
    def get_practice_problems(self, topic, difficulty=None):
        """
        Get practice problems for a topic
        
        Args:
            topic (str): Topic name
            difficulty (str): Optional difficulty filter
            
        Returns:
            list: List of practice problems
        """
        topic_data = self.get_topic_by_name(topic)
        
        if topic_data and 'practice_problems' in topic_data:
            problems = topic_data['practice_problems']
            
            # Filter by difficulty if specified
            if difficulty:
                problems = [p for p in problems if p.get('difficulty', '').lower() == difficulty.lower()]
            
            return problems
        
        # Return default problems if topic not found
        return [
            {
                "question": f"What is {topic}?",
                "answer": f"{topic} is an important concept that requires understanding of fundamental principles.",
                "difficulty": "medium"
            }
        ]
    
    def add_topic(self, topic_key, topic_data):
        """
        Add new topic to knowledge base
        
        Args:
            topic_key (str): Unique key for the topic
            topic_data (dict): Topic information
            
        Returns:
            bool: Success status
        """
        try:
            # Validate required fields
            required_fields = ['title', 'definition']
            for field in required_fields:
                if field not in topic_data:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Add timestamp
            topic_data['last_updated'] = datetime.now().isoformat()
            
            # Add to knowledge base
            self.knowledge[topic_key] = topic_data
            
            # Clear cache
            self.search_cache.clear()
            
            # Save to file
            self.save_knowledge()
            
            logger.info(f"Added new topic: {topic_key}")
            return True
        except Exception as e:
            logger.error(f"Error adding topic: {e}")
            return False
    
    def update_topic(self, topic_key, updates):
        """
        Update existing topic
        
        Args:
            topic_key (str): Topic key
            updates (dict): Fields to update
            
        Returns:
            bool: Success status
        """
        if topic_key not in self.knowledge:
            logger.warning(f"Topic not found: {topic_key}")
            return False
        
        try:
            # Update fields
            self.knowledge[topic_key].update(updates)
            self.knowledge[topic_key]['last_updated'] = datetime.now().isoformat()
            
            # Clear cache
            self.search_cache.clear()
            
            # Save to file
            self.save_knowledge()
            
            logger.info(f"Updated topic: {topic_key}")
            return True
        except Exception as e:
            logger.error(f"Error updating topic: {e}")
            return False
    
    def delete_topic(self, topic_key):
        """
        Delete a topic from knowledge base
        
        Args:
            topic_key (str): Topic key
            
        Returns:
            bool: Success status
        """
        if topic_key not in self.knowledge:
            logger.warning(f"Topic not found: {topic_key}")
            return False
        
        try:
            del self.knowledge[topic_key]
            self.search_cache.clear()
            self.save_knowledge()
            logger.info(f"Deleted topic: {topic_key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting topic: {e}")
            return False
    
    def learn_from_conversation(self, query, response, intent, confidence=0.0):
        """
    Automatically learn from successful conversations
    Only learns from high-confidence, high-quality interactions
    """
        try:
        # Only learn from high-confidence explanations
            if intent != 'explanation' or confidence < 0.8:
                return False
        
        # Extract potential topic from query
            topic_key = self._extract_topic_from_query(query)
            if not topic_key:
                return False
        
        # Check if topic already exists
            if topic_key in self.knowledge:
                return self._enhance_existing_topic(topic_key, query, response)
            else:
                return self._create_new_topic(topic_key, query, response)
            
        except Exception as e:
            logger.error(f"Auto-learning failed: {e}")
            return False

    
    def get_popular_topics(self, limit=5):
        """
        Get most accessed topics based on analytics
        
        Args:
            limit (int): Number of topics to return
            
        Returns:
            list: Popular topics with access counts
        """
        sorted_topics = sorted(
            self.access_stats.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            {
                'topic': topic,
                'access_count': count
            }
            for topic, count in sorted_topics
        ]
    
    def get_related_topics(self, topic_name):
        """
        Get topics related to a given topic
        
        Args:
            topic_name (str): Topic name
            
        Returns:
            list: Related topic titles
        """
        topic_data = self.get_topic_by_name(topic_name)
        
        if topic_data and 'related_topics' in topic_data:
            related = []
            for related_key in topic_data['related_topics']:
                if related_key in self.knowledge:
                    related.append(self.knowledge[related_key]['title'])
            return related
        
        return []
    
    def get_learning_path(self, target_topic):
        """
        Generate a learning path to reach a target topic
        
        Args:
            target_topic (str): Target topic name
            
        Returns:
            list: Ordered list of topics to study
        """
        topic_data = self.get_topic_by_name(target_topic)
        
        if not topic_data:
            return []
        
        # Build learning path from prerequisites
        path = []
        prerequisites = topic_data.get('prerequisites', [])
        
        # Add prerequisites
        for prereq in prerequisites:
            prereq_data = self.get_topic_by_name(prereq)
            if prereq_data:
                # Recursively get prerequisites of prerequisites
                sub_path = self.get_learning_path(prereq)
                path.extend(sub_path)
                if prereq not in path:
                    path.append(prereq)
        
        # Add target topic
        if target_topic not in path:
            path.append(target_topic)
        
        return path
    
    def search_by_category(self, category):
        """
        Get all topics in a specific category
        
        Args:
            category (str): Category name
            
        Returns:
            list: Topics in the category
        """
        topics = []
        for key, data in self.knowledge.items():
            if data.get('category', '').lower() == category.lower():
                topics.append({
                    'key': key,
                    'title': data['title'],
                    'description': data['definition'][:100] + '...'
                })
        
        return topics
    
    def get_statistics(self):
        """
        Get knowledge base statistics
        
        Returns:
            dict: Statistics about the knowledge base
        """
        total_topics = len(self.knowledge)
        total_examples = sum(len(data.get('examples', [])) for data in self.knowledge.values())
        total_problems = sum(len(data.get('practice_problems', [])) for data in self.knowledge.values())
        
        # Count by category
        categories = {}
        for data in self.knowledge.values():
            cat = data.get('category', 'Other')
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'total_topics': total_topics,
            'total_examples': total_examples,
            'total_practice_problems': total_problems,
            'categories': categories,
            'most_accessed': self.get_popular_topics(3)
        }


# Test the knowledge base
if __name__ == "__main__":
    print("="*70)
    print("TESTING KNOWLEDGE BASE")
    print("="*70)
    
    kb = KnowledgeBase()
    
    # Test 1: Search
    print("\n1. Testing search...")
    result = kb.search_topic(['machine', 'learning'])
    if result:
        print(f"   Found: {result['title']}")
        print(f"   Category: {result.get('category')}")
        print(f"   Definition: {result['definition'][:100]}...")
        print(f"   Examples: {len(result.get('examples', []))}")
    
    # Test 2: All topics
    print("\n2. Testing get all topics...")
    topics = kb.get_all_topics()
    print(f"   Total topics: {len(topics)}")
    for topic in topics:
        print(f"   - {topic['title']} ({topic['category']}) - {topic['difficulty']}")
    
    # Test 3: Practice problems
    print("\n3. Testing practice problems...")
    problems = kb.get_practice_problems('Machine Learning')
    print(f"   Found {len(problems)} problems")
    if problems:
        print(f"   First problem: {problems[0]['question']}")
    
    # Test 4: Related topics
    print("\n4. Testing related topics...")
    related = kb.get_related_topics('Machine Learning')
    print(f"   Related topics: {', '.join(related)}")
    
    # Test 5: Learning path
    print("\n5. Testing learning path...")
    path = kb.get_learning_path('Deep Learning')
    print(f"   Learning path to Deep Learning:")
    for i, topic in enumerate(path, 1):
        print(f"   {i}. {topic}")
    
    # Test 6: Popular topics tracking
    print("\n6. Testing popular topics tracking...")
    kb.search_topic(['python'])
    kb.search_topic(['python'])
    kb.search_topic(['machine', 'learning'])
    kb.search_topic(['data', 'science'])
    popular = kb.get_popular_topics(3)
    print("   Popular topics:")
    for item in popular:
        print(f"   - {item['topic']}: {item['access_count']} accesses")
    
    # Test 7: Category search
    print("\n7. Testing category search...")
    ai_topics = kb.search_by_category('AI/ML')
    print(f"   Topics in AI/ML category: {len(ai_topics)}")
    for topic in ai_topics:
        print(f"   - {topic['title']}")
    
    # Test 8: Statistics
    print("\n8. Testing statistics...")
    stats = kb.get_statistics()
    print(f"   Total Topics: {stats['total_topics']}")
    print(f"   Total Examples: {stats['total_examples']}")
    print(f"   Total Practice Problems: {stats['total_practice_problems']}")
    print(f"   Categories: {stats['categories']}")
    
    # Test 9: Save
    print("\n9. Testing save...")
    if kb.save_knowledge():
        print("   ✓ Knowledge base saved successfully")
        print(f"   ✓ Saved to: {kb.data_path}")
    
    # Test 10: Add new topic
    print("\n10. Testing add new topic...")
    new_topic = {
        "title": "Computer Vision",
        "category": "AI/ML",
        "definition": "Computer Vision is a field of AI that enables computers to interpret and understand visual information from the world.",
        "key_concepts": ["Image Processing", "Object Detection", "Image Classification"],
        "examples": [],
        "practice_problems": [],
        "resources": ["https://opencv.org/"],
        "prerequisites": ["Python Programming", "Deep Learning"],
        "related_topics": ["deep_learning", "machine_learning"]
    }
    
    if kb.add_topic("computer_vision", new_topic):
        print("   ✓ New topic 'Computer Vision' added successfully")
        print(f"   ✓ Total topics now: {len(kb.knowledge)}")
    
    print("\n" + "="*70)
    print("✅ KNOWLEDGE BASE TESTS COMPLETE!")
    print("="*70)
    print("\nKnowledge Base Features:")
    print("  ✓ Comprehensive educational content")
    print("  ✓ Intelligent search with scoring")
    print("  ✓ Practice problems and examples")
    print("  ✓ Learning path generation")
    print("  ✓ Analytics and tracking")
    print("  ✓ Category-based organization")
    print("  ✓ Auto-save functionality")
    print("  ✓ Dynamic topic management")
    print("\nReady for production use! 🚀")
    print("="*70)
    # print("structured data.":{
    #             "key_concepts": [
    #                 "Data Collection",
    #                 "Data Cleaning",
    #                 "Exploratory Data Analysis",
    #                 "Statistical Analysis",
    #                 "Data Visualization",
    #                 "Machine Learning",
    #                 "Big Data"
    #             ],
    #             "examples": [
    #                 {
    #                     "name": "Data Analysis with Pandas",
    #                     "description": "Basic data manipulation using pandas",
    #                     "code": "import pandas as pd\nimport numpy as np\n\n# Create DataFrame\ndata = {\n    'name': ['Alice', 'Bob', 'Charlie'],\n    'age': [25, 30, 35],\n    'score': [85, 90, 95]\n}\ndf = pd.DataFrame(data)\n\n# Basic operations\nprint(df.describe())\nprint(df[df['age'] > 25])\nprint(df.groupby('age')['score'].mean())",
    #                     "difficulty": "beginner"
    #                 }
    #             ],
    #             "practice_problems": [
    #                 {
    #                     "question": "What is the difference between mean, median, and mode?",
    #                     "answer": "Mean is the average of all values. Median is the middle value when data is sorted. Mode is the most frequently occurring value.",
    #                     "difficulty": "easy"
    #                 }
    #             ],
    #             "resources": [
    #                 "https://www.kaggle.com/learn/intro-to-data-science",
    #                 "https://pandas.pydata.org/docs/getting_started/tutorials.html"
    #             ],
    #             "prerequisites": ["Python Programming", "Basic Statistics"],
    #             "related_topics": ["machine_learning", "statistics"],
    #             "last_updated": datetime.now().isoformat()
    #             }, )
        
    
    def save_knowledge(self):
        """Save knowledge base to file"""
        try:
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge, f, indent=2, ensure_ascii=False)
            logger.info(f"Knowledge base saved to {self.data_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
            return False
    
    def search_topic(self, keywords):
        """
        Search for topic by keywords
        
        Args:
            keywords (list): List of keywords to search
            
        Returns:
            dict: Topic data or None
        """
        if not keywords:
            return None
        
        # Create cache key
        cache_key = '-'.join(sorted(keywords))
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        # Search through topics
        best_match = None
        best_score = 0
        
        for topic_key, topic_data in self.knowledge.items():
            score = 0
            
            # Check in topic key
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in topic_key:
                    score += 3
                
                # Check in title
                if keyword_lower in topic_data['title'].lower():
                    score += 3
                
                # Check in definition
                if keyword_lower in topic_data['definition'].lower():
                    score += 1
                
                # Check in key concepts
                for concept in topic_data.get('key_concepts', []):
                    if keyword_lower in concept.lower():
                        score += 2
            
            if score > best_score:
                best_score = score
                best_match = topic_data
        
        # Cache result
        if best_match:
            self.search_cache[cache_key] = best_match
            # Track access
            self.access_stats[best_match['title']] += 1
        
        return best_match
    
    def get_topic_by_name(self, topic_name):
        """Get topic by exact name"""
        topic_key = topic_name.lower().replace(' ', '_')
        return self.knowledge.get(topic_key)
    
    def get_all_topics(self):
        """Get list of all available topics"""
        return [
            {
                'key': key,
                'title': data['title'],
                'category': data.get('category', 'General'),
                'description': data['definition'][:150] + '...'
            }
            for key, data in self.knowledge.items()
        ]
    
    def get_practice_problems(self, topic):
        """
        Get practice problems for a topic
        
        Args:
            topic (str): Topic name
            
        Returns:
            list: List of practice problems
        """
        topic_data = self.get_topic_by_name(topic)
        
        if topic_data and 'practice_problems' in topic_data:
            return topic_data['practice_problems']
        
        # Return default problems if topic not found
        return [
            {
                "question": f"What is {topic}?",
                "answer": f"{topic} is an important concept that requires understanding of fundamental principles.",
                "difficulty": "medium"
            }
        ]
    
    def add_topic(self, topic_key, topic_data):
        """
        Add new topic to knowledge base
        
        Args:
            topic_key (str): Unique key for the topic
            topic_data (dict): Topic information
            
        Returns:
            bool: Success status
        """
        try:
            topic_data['last_updated'] = datetime.now().isoformat()
            self.knowledge[topic_key] = topic_data
            self.save_knowledge()
            logger.info(f"Added new topic: {topic_key}")
            return True
        except Exception as e:
            logger.error(f"Error adding topic: {e}")
            return False
    
    def update_topic(self, topic_key, updates):
        """
        Update existing topic
        
        Args:
            topic_key (str): Topic key
            updates (dict): Fields to update
            
        Returns:
            bool: Success status
        """
        if topic_key not in self.knowledge:
            logger.warning(f"Topic not found: {topic_key}")
            return False
        
        try:
            self.knowledge[topic_key].update(updates)
            self.knowledge[topic_key]['last_updated'] = datetime.now().isoformat()
            self.save_knowledge()
            logger.info(f"Updated topic: {topic_key}")
            return True
        except Exception as e:
            logger.error(f"Error updating topic: {e}")
            return False
    
    def learn_from_conversation(self, query, response, intent, feedback=None):
        """
        Auto-learn from user conversations (ML-powered feature)
        This can be used to improve the knowledge base over time
        
        Args:
            query (str): User query
            response (str): Generated response
            intent (str): Detected intent
            feedback (int): User feedback rating (1-5)
        """
        # Only learn from highly-rated responses
        if feedback and feedback >= 4:
            logger.info(f"Learning from positive feedback: {query[:50]}...")
            # TODO: Implement ML-based content extraction and addition
            # This could use NLP to extract key concepts and add them to knowledge base
    
    def get_popular_topics(self, limit=5):
        """
        Get most accessed topics
        
        Args:
            limit (int): Number of topics to return
            
        Returns:
            list: Popular topics
        """
        sorted_topics = sorted(
            self.access_stats.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            {
                'topic': topic,
                'access_count': count
            }
            for topic, count in sorted_topics
        ]
    
    def get_related_topics(self, topic_name):
        """
        Get topics related to a given topic
        
        Args:
            topic_name (str): Topic name
            
        Returns:
            list: Related topic names
        """
        topic_data = self.get_topic_by_name(topic_name)
        
        if topic_data and 'related_topics' in topic_data:
            related = []
            for related_key in topic_data['related_topics']:
                if related_key in self.knowledge:
                    related.append(self.knowledge[related_key]['title'])
            return related
        
        return []


# Test the knowledge base
if __name__ == "__main__":
    print("="*70)
    print("TESTING KNOWLEDGE BASE")
    print("="*70)
    
    kb = KnowledgeBase()
    
    # Test search
    print("\n1. Testing search...")
    result = kb.search_topic(['machine', 'learning'])
    if result:
        print(f"   Found: {result['title']}")
        print(f"   Definition: {result['definition'][:100]}...")
    
    # Test all topics
    print("\n2. Testing get all topics...")
    topics = kb.get_all_topics()
    print(f"   Total topics: {len(topics)}")
    for topic in topics:
        print(f"   - {topic['title']} ({topic['category']})")
    
    # Test practice problems
    print("\n3. Testing practice problems...")
    problems = kb.get_practice_problems('Machine Learning')
    print(f"   Found {len(problems)} problems")
    
    # Test popular topics
    print("\n4. Testing popular topics tracking...")
    kb.search_topic(['python'])
    kb.search_topic(['python'])
    kb.search_topic(['machine', 'learning'])
    popular = kb.get_popular_topics()
    print("   Popular topics:", [t['topic'] for t in popular])
    
    # Test save
    print("\n5. Testing save...")
    if kb.save_knowledge():
        print("   ✓ Knowledge base saved successfully")
    
    print("\n" + "="*70)
    print("KNOWLEDGE BASE TESTS COMPLETE!")
    print("="*70)