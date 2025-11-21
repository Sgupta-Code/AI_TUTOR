"""
COMPLETE FREE AI Tutor - Uses Serper API (Free Tier) + LLM fallback (Groq example)

How it works:
1) Try Serper web search
2) If direct answer/snippet -> return formatted web response
3) Else try Wikipedia fallback
4) Else call the LLM fallback (_llm_answer)
5) If LLM returns -> return LLM answer
6) Else use predefined responses / fallback
"""

import logging
import requests
import json
import os
from typing import Optional, Dict
from dotenv import load_dotenv
load_dotenv()


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ResponseGenerator:
    """AI Tutor using FREE Serper API for Google-quality search with LLM fallback"""
    
    def __init__(self):
        # Serper free key (you currently had a public key in your file)
        # Recommended: move these to environment variables for security.
        self.serper_api_key = os.getenv("SERPER_API_KEY", "75858b0c1f20ad6d5c9055355da2f1d2550b00f0")
        
        # Groq / LLM fallback API key (replace or set env var GROQ_API_KEY)
        # You may use another LLM provider that exposes an OpenAI-compatible API.
        self.groq_api_key = os.getenv("GROQ_API_KEY", None)
        
        logger.info("AI Tutor with FREE Web Search Initialized!")
        if self.groq_api_key:
            logger.info("LLM fallback enabled (Groq key found).")
        else:
            logger.info("LLM fallback disabled (no GROQ_API_KEY found). Will rely on predefined fallbacks.")
    
    def generate_response(self, query: str, intent: str, preprocessed: Optional[dict] = None) -> dict:
        """Generate response using web search and fallback to LLM if needed"""
        try:
            # Try web search first
            search_data = self._search_web(query)
            
            if search_data and search_data.get('answer'):
                return self._format_web_response(query, search_data, intent)
            
            # If web search gave nothing, try Wikipedia fallback (already integrated in _search_web)
            if search_data is None:
                # _search_web already attempts Wikipedia fallback; re-call explicitly for clarity
                wiki_data = self._search_wikipedia_fallback(query)
                if wiki_data and wiki_data.get('answer'):
                    return self._format_web_response(query, wiki_data, intent)
            
            # If still no web answer, try the LLM fallback (if key available)
            llm_reply = self._llm_answer(query)
            if llm_reply:
                # format into the same response shape as other branches
                response_text = f"**Answer (LLM)**\n\n{llm_reply}\n\n"
                response_text += self._add_educational_context(query)
                return {
                    'response': response_text,
                    'suggestions': self._get_suggestions(query, intent),
                    'resources': self._get_resources(self._extract_topic(query)),
                    'topic': self._extract_topic(query),
                    'source': 'llm'
                }
            
            # As a final fallback, use the smart/predefined responses
            return self._generate_smart_response(query, intent)
                
        except Exception as e:
            logger.error(f"Error in generate_response: {e}", exc_info=True)
            return self._generate_fallback_response(query)
    
    # --------------------------
    # Web search (Serper) + parsing
    # --------------------------
    def _search_web(self, query: str) -> Optional[dict]:
        """Search web using FREE Serper API (Google-quality results)"""
        try:
            url = "https://google.serper.dev/search"
            headers = {
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }
            payload = {
                'q': query,
                'gl': 'us',
                'hl': 'en',
                'num': 5
            }
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_search_results(data, query)
            else:
                logger.warning(f"Serper returned status {response.status_code}. Trying Wikipedia fallback.")
                return self._search_wikipedia_fallback(query)
                
        except Exception as e:
            logger.warning(f"Web search failed: {e}. Trying Wikipedia fallback.")
            return self._search_wikipedia_fallback(query)
    
    def _parse_search_results(self, data: dict, query: str) -> Optional[dict]:
        """Parse search results to get the best answer"""
        try:
            # Try to get answer box first (direct answer)
            if data.get('answerBox'):
                answer_box = data['answerBox']
                if answer_box.get('answer'):
                    return {
                        'answer': answer_box['answer'],
                        'title': answer_box.get('title', query),
                        'source': 'Google Search',
                        'type': 'direct_answer'
                    }
            
            # Get organic results
            organic_results = data.get('organic', [])
            if organic_results:
                first_result = organic_results[0]
                snippet = first_result.get('snippet', '') or ''
                title = first_result.get('title', '') or ''
                
                if snippet and len(snippet) > 50:
                    return {
                        'answer': snippet,
                        'title': title if title else query,
                        'source': first_result.get('link', ''),
                        'type': 'web_search'
                    }
                    
        except Exception as e:
            logger.warning(f"Parsing search results failed: {e}")
        
        return None
    
    # --------------------------
    # Wikipedia fallback
    # --------------------------
    def _search_wikipedia_fallback(self, query: str) -> Optional[dict]:
        """Fallback to Wikipedia API"""
        try:
            clean_query = self._clean_query(query)
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean_query}"
            response = requests.get(url, timeout=8)
            if response.status_code == 200:
                data = response.json()
                extract = data.get('extract', '')
                if extract and len(extract) > 100:
                    return {
                        'answer': extract,
                        'title': data.get('title', clean_query.replace('_', ' ')),
                        'source': 'Wikipedia',
                        'type': 'wikipedia'
                    }
        except Exception as e:
            logger.debug(f"Wikipedia fallback failed: {e}")
        return None
    
    # --------------------------
    # LLM fallback (Groq example / OpenAI-compatible)
    # --------------------------
    def _llm_answer(self, query: str) -> Optional[str]:
        """
        Call a generative LLM as a fallback. This example uses a Groq-style endpoint
        that accepts OpenAI-compatible chat completion payloads.
        
        If you use a different provider, adapt headers/URL/payload accordingly.
        """
        if not self.groq_api_key:
            logger.info("No GROQ_API_KEY set; skipping LLM fallback.")
            return None
        
        try:
            # Example using an OpenAI-compatible "chat completions" style payload.
            # Replace url with your provider's URL if needed.
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.3-70b-versatile",   # adjust to available model
                "messages": [
                    {"role": "system", "content": "You are an educational AI tutor. Explain clearly and concisely."},
                    {"role": "user", "content": query}
                ],
                "max_tokens": 512,
                "temperature": 0.2
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                # Compatible parsing for OpenAI-style response
                choice = data.get("choices", [{}])[0]
                msg = choice.get("message", {}).get("content")
                if not msg:
                    # Some providers return 'text' or other structure
                    msg = choice.get("text") or data.get("text")
                return msg
            else:
                logger.warning(f"LLM request failed: {resp.status_code} {resp.text}")
                return None
        except Exception as e:
            logger.warning(f"LLM call exception: {e}")
            return None
    
    # --------------------------
    # Helpers: clean topic, format output, suggestions, resources
    # --------------------------
    def _clean_query(self, query: str) -> str:
        """Clean query for search"""
        question_words = ['what', 'is', 'are', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 
                         'who', 'which', 'can', 'you', 'tell', 'me', 'about', 'explain', 'show']
        
        words = query.lower().split()
        clean_words = [word for word in words if word not in question_words and len(word) > 2]
        
        if not clean_words:
            return query.replace(' ', '_')
        
        full_query = query.lower()
        common_topics = {
            'machine learning': 'Machine_learning',
            'data science': 'Data_science', 
            'artificial intelligence': 'Artificial_intelligence',
            'neural network': 'Artificial_neural_network',
            'python programming': 'Python_(programming_language)',
            'quantum computing': 'Quantum_computing',
            'indian independence day': 'Independence_Day_(India)',
            'elon musk': 'Elon_Musk',
            'photosynthesis': 'Photosynthesis',
            'global warming': 'Global_warming',
            'blockchain': 'Blockchain',
            'virtual reality': 'Virtual_reality',
            'augmented reality': 'Augmented_reality',
            'climate change': 'Climate_change',
            'renewable energy': 'Renewable_energy'
        }
        
        for topic, wiki_name in common_topics.items():
            if topic in full_query:
                return wiki_name
        
        meaningful_words = [word for word in clean_words if len(word) > 2]
        if meaningful_words:
            topic = max(meaningful_words, key=len)
            return topic.title().replace(' ', '_')
        
        return query.replace(' ', '_')
    
    def _format_web_response(self, query: str, search_data: dict, intent: str) -> dict:
        """Format web search response"""
        answer = search_data['answer']
        title = search_data.get('title', self._extract_topic(query))
        
        response = f"**{title}**\n\n"
        response += f"{answer}\n\n"
        
        # Add educational context
        response += self._add_educational_context(query)
        
        # Add source
        source_text = search_data.get('source', 'Web Search')
        if not source_text.startswith('http'):
            response += f"*Source: {source_text}*"
        else:
            response += f"*Source: Web Search*"
        
        return {
            'response': response,
            'suggestions': self._get_suggestions(query, intent),
            'resources': self._get_resources(title, search_data.get('source')),
            'topic': title,
            'source': 'web_search'
        }
    
    def _add_educational_context(self, query: str) -> str:
        """Add educational context"""
        context = "\n**🎓 Learning Guide:**\n"
        
        query_lower = query.lower()
        
        if any(phrase in query_lower for phrase in ['what is', 'explain']):
            context += "• Start with the basic definition\n"
            context += "• Understand key components and principles\n" 
            context += "• Explore real-world applications\n"
            context += "• Practice with examples and exercises\n"
        
        elif any(phrase in query_lower for phrase in ['how to', 'how does']):
            context += "• Break down the process into steps\n"
            context += "• Understand each step thoroughly\n"
            context += "• Practice with simple examples first\n"
            context += "• Apply knowledge to practical scenarios\n"
        
        elif any(phrase in query_lower for phrase in ['history', 'about']):
            context += "• Learn about historical background\n"
            context += "• Identify key events and figures\n"
            context += "• Understand evolution over time\n"
            context += "• Connect past to present applications\n"
        
        else:
            context += "• Take notes on key information\n"
            context += "• Relate to existing knowledge\n"
            context += "• Ask follow-up questions\n"
            context += "• Practice recall and application\n"
        
        return context
    
    def _generate_smart_response(self, query: str, intent: str) -> dict:
        """Generate smart response when search and LLM both fail"""
        topic = self._extract_topic(query)
        
        # Predefined responses for common topics
        predefined_responses = {
            'machine learning': "Machine Learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze patterns in data and make predictions or decisions.",
            
            'python': "Python is a high-level programming language known for its clear syntax and readability. It's widely used in web development, data science, artificial intelligence, scientific computing, and automation.",
            
            'data science': "Data Science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.",
            
            'neural networks': "Neural Networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that can learn to perform tasks by considering examples without being programmed with task-specific rules.",
            
            'elon musk': "Elon Musk is a business magnate and investor known for founding several technology companies including Tesla (electric vehicles), SpaceX (aerospace), Neuralink (neurotechnology), and The Boring Company (infrastructure).",
            
            'indian independence': "Indian Independence Day is celebrated on August 15th each year, marking India's independence from British rule in 1947. It's a national holiday celebrated with flag-hoisting ceremonies, parades, and cultural events across the country.",
            
            'photosynthesis': "Photosynthesis is the process used by plants, algae and certain bacteria to convert light energy into chemical energy that can be released to fuel the organisms' activities.",
            
            'blockchain': "Blockchain is a distributed ledger technology that maintains a secure and decentralized record of transactions. It's the foundation for cryptocurrencies like Bitcoin and has applications in supply chain, voting systems, and more.",
            
            'artificial intelligence': "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions, including learning and problem-solving.",
            
            'quantum computing': "Quantum computing uses quantum-mechanical phenomena to perform computation. Unlike classical computers that use bits, quantum computers use quantum bits (qubits) that can exist in multiple states simultaneously."
        }
        
        topic_lower = topic.lower()
        if topic_lower in predefined_responses:
            response = f"**{topic}**\n\n{predefined_responses[topic_lower]}"
        else:
            response = f"**{topic}**\n\n{topic} is an important concept worth exploring. It involves understanding fundamental principles and their practical applications across various fields and industries."
        
        response += "\n\n**💡 Learning Tip**: For the most current information, I recommend checking reliable educational resources and recent publications on this topic."
        
        return {
            'response': response,
            'suggestions': self._get_suggestions(query, intent),
            'resources': self._get_resources(topic),
            'topic': topic
        }
    
    def _extract_topic(self, query: str) -> str:
        """Extract topic from query"""
        clean_query = self._clean_query(query)
        return clean_query.replace('_', ' ').title()
    
    def _get_suggestions(self, query: str, intent: str) -> list:
        """Get contextual suggestions"""
        base_suggestions = [
            "Explain this in simpler terms",
            "Show me practical examples", 
            "Provide learning resources",
            "Break this down step by step"
        ]
        return base_suggestions
    
    def _get_resources(self, topic: str, source_url: str = None) -> list:
        """Get learning resources"""
        resources = [
            f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
            f"https://www.khanacademy.org/search?page_search_query={topic}",
            f"https://www.coursera.org/courses?query={topic}",
            f"https://www.youtube.com/results?search_query={topic}+explained"
        ]
        
        if source_url and source_url.startswith('http') and source_url not in resources:
            resources.insert(0, source_url)
            
        return resources
    
    def _generate_fallback_response(self, query: str) -> dict:
        """Generate final fallback response"""
        topic = self._extract_topic(query)
        
        return {
            'response': f"**I'm here to help you learn about {topic}!** 🎓\n\nWhile I work on getting the most current information for you, here are some excellent learning resources where you can find comprehensive information about {topic}.\n\nWhat specific aspect of {topic} would you like me to help you understand?",
            'suggestions': [
                "Explain the basic concepts",
                "Show me learning resources", 
                "Break it down step by step",
                "Give me practical examples"
            ],
            'resources': [
                f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
                "https://www.khanacademy.org/",
                "https://www.coursera.org/",
                "https://ocw.mit.edu/"
            ],
            'topic': topic
        }
