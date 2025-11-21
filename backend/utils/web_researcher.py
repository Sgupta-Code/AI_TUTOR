"""
Web Research Module for Dynamic Knowledge Expansion
Fetches real-time information from reliable sources
"""

import requests
import logging
import re
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class WebResearcher:
    """Automatically research topics from multiple reliable sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AITutor/1.0 Educational Research Bot'
        })
        
        # Import ddgs with error handling
        try:
            from ddgs import DDGS
            self.ddgs = DDGS()
            self.ddgs_available = True
        except ImportError:
            logger.warning("DDGS not available, using fallback research")
            self.ddgs_available = False
        
        # Import wikipedia with error handling
        try:
            import wikipediaapi
            self.wikipedia = wikipediaapi.Wikipedia(
                user_agent='AITutor/1.0',
                language='en',
                extract_format=wikipediaapi.ExtractFormat.WIKI
            )
            self.wikipedia_available = True
        except ImportError:
            logger.warning("Wikipedia API not available")
            self.wikipedia_available = False
    
    def research_topic(self, topic):
        """
        Research a topic from multiple web sources
        Returns structured educational content
        """
        logger.info(f"Researching topic: {topic}")
        
        try:
            # Try multiple sources in order of reliability
            sources = []
            
            if self.wikipedia_available:
                wiki_data = self._fetch_wikipedia(topic)
                if wiki_data:
                    sources.append(wiki_data)
            
            edu_data = self._fetch_educational_sites(topic)
            if edu_data:
                sources.append(edu_data)
            
            if self.ddgs_available:
                ddg_data = self._fetch_duckduckgo(topic)
                if ddg_data:
                    sources.append(ddg_data)
            
            # Combine results from all sources
            if sources:
                combined_data = self._combine_sources(sources, topic)
                logger.info(f"Successfully researched: {topic}")
                return combined_data
            else:
                return self._create_fallback_content(topic)
                
        except Exception as e:
            logger.error(f"Research failed for {topic}: {e}")
            return self._create_fallback_content(topic)
    
    def _fetch_wikipedia(self, topic):
        """Fetch comprehensive data from Wikipedia"""
        try:
            page = self.wikipedia.page(topic)
            
            if page.exists():
                return {
                    "source": "wikipedia",
                    "title": page.title,
                    "summary": page.summary[:1000],
                    "full_content": page.text[:2000],
                    "categories": list(page.categories.keys())[:5],
                    "url": page.fullurl,
                    "confidence": 0.9
                }
        except Exception as e:
            logger.warning(f"Wikipedia fetch failed: {e}")
        
        return None
    
    def _fetch_educational_sites(self, topic):
        """Fetch from reliable educational websites"""
        educational_sites = [
            f"https://www.geeksforgeeks.org/{topic.replace(' ', '-')}/",
            f"https://www.tutorialspoint.com/{topic.replace(' ', '_')}/index.htm",
            f"https://www.javatpoint.com/{topic.replace(' ', '-')}"
        ]
        
        for site in educational_sites:
            try:
                response = self.session.get(site, timeout=5)
                if response.status_code == 200:
                    return {
                        "source": "educational_site",
                        "url": site,
                        "content": self._extract_educational_content(response.text),
                        "confidence": 0.8
                    }
            except:
                continue
        
        return None
    
    def _fetch_duckduckgo(self, topic):
        """Fetch instant answers from DuckDuckGo"""
        try:
            results = list(self.ddgs.text(f"what is {topic}", max_results=3))
            
            if results:
                return {
                    "source": "duckduckgo",
                    "results": results,
                    "confidence": 0.7
                }
        except Exception as e:
            logger.warning(f"DuckDuckGo fetch failed: {e}")
        
        return None
    
    def _extract_educational_content(self, html_content):
        """Extract educational content from HTML"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean it
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:1500]
        except:
            return ""
    
    def _combine_sources(self, sources, topic):
        """Combine data from multiple sources"""
        valid_sources = [s for s in sources if s is not None]
        
        if not valid_sources:
            return None
        
        # Use the highest confidence source as primary
        primary_source = max(valid_sources, key=lambda x: x.get('confidence', 0))
        
        # Create structured knowledge base entry
        knowledge_entry = {
            "title": primary_source.get('title', topic.title()),
            "definition": self._create_definition(primary_source, valid_sources),
            "key_concepts": self._extract_key_concepts(valid_sources),
            "examples": self._generate_examples(topic),
            "resources": self._extract_resources(valid_sources),
            "category": self._determine_category(valid_sources),
            "prerequisites": [],
            "related_topics": self._extract_related_topics(valid_sources),
            "last_updated": datetime.now().isoformat(),
            "source": "web_research",
            "auto_learned": True
        }
        
        return knowledge_entry
    
    def _create_definition(self, primary_source, all_sources):
        """Create a comprehensive definition from multiple sources"""
        definitions = []
        
        for source in all_sources:
            if source.get('summary'):
                definitions.append(source['summary'])
            elif source.get('content'):
                definitions.append(source['content'][:500])
        
        if definitions:
            # Use the longest definition (usually most comprehensive)
            return max(definitions, key=len)
        else:
            return f"{primary_source.get('title', 'This topic')} is an important concept that can be explored through various educational resources."
    
    def _extract_key_concepts(self, sources):
        """Extract key concepts from source content"""
        concepts = set()
        
        for source in sources:
            content = ""
            if source.get('summary'):
                content += source['summary']
            if source.get('full_content'):
                content += source['full_content']
            if source.get('content'):
                content += source['content']
            
            # Extract capitalized phrases (potential concepts)
            found_concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
            concepts.update(found_concepts)
        
        return list(concepts)[:8]
    
    def _generate_examples(self, topic):
        """Generate basic example structure"""
        return [
            {
                "name": f"Basic {topic.title()} Example",
                "description": f"This demonstrates fundamental concepts of {topic}",
                "code": f"# Example code for {topic}\n# This would typically show implementation details",
                "difficulty": "beginner"
            }
        ]
    
    def _extract_resources(self, sources):
        """Extract resource URLs from sources"""
        resources = []
        
        for source in sources:
            if source.get('url'):
                resources.append(source['url'])
            if source.get('results'):
                for result in source['results']:
                    if 'href' in result:
                        resources.append(result['href'])
        
        return list(set(resources))[:5]
    
    def _determine_category(self, sources):
        """Determine the category based on source content"""
        category = "Computer Science"
        
        for source in sources:
            if source.get('categories'):
                categories = source['categories']
                for cat in categories:
                    cat_lower = cat.lower()
                    if 'programming' in cat_lower or 'computer' in cat_lower:
                        return "Programming"
                    elif 'math' in cat_lower:
                        return "Mathematics"
                    elif 'science' in cat_lower:
                        return "Science"
                    elif 'physics' in cat_lower:
                        return "Physics"
                    elif 'biology' in cat_lower:
                        return "Biology"
        
        return category
    
    def _extract_related_topics(self, sources):
        """Extract related topics from source content"""
        related = set()
        
        for source in sources:
            if source.get('categories'):
                categories = source['categories']
                for cat in categories[:3]:
                    clean_cat = cat.replace('Category:', '').replace('_', ' ').title()
                    related.add(clean_cat)
        
        return list(related)
    
    def _create_fallback_content(self, topic):
        """Create fallback content when research fails"""
        return {
            "title": topic.title(),
            "definition": f"I've researched {topic} and found it to be an important concept. While I'm gathering more detailed information, you might want to check reliable educational resources for comprehensive details.",
            "key_concepts": [topic.title(), "Fundamentals", "Applications"],
            "examples": [],
            "resources": [
                f"https://www.wikipedia.org/wiki/{topic.replace(' ', '_')}",
                f"https://www.geeksforgeeks.org/{topic.replace(' ', '-')}/"
            ],
            "category": "General",
            "prerequisites": [],
            "related_topics": [],
            "last_updated": datetime.now().isoformat(),
            "source": "fallback",
            "auto_learned": True
        }

    # Add this alias method for convenience
    def research(self, topic):
        """Alias for research_topic for easier usage"""
        return self.research_topic(topic)