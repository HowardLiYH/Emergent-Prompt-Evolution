"""
L4 Web Search Tool - Uses Tavily API for real-time web search.
"""
import os
from typing import Optional, List, Dict, Any
import asyncio


class WebSearchTool:
    """
    L4 Tool: Real-time web search using Tavily API.
    
    This provides REAL web search capabilities, not simulated.
    """
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = 60
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize web search tool.
        
        Args:
            api_key: Tavily API key
        """
        self.api_key = api_key or os.getenv('TAVILY_API_KEY')
        self._client = None
        self._request_count = 0
        self._last_reset = None
    
    def _get_client(self):
        """Lazily initialize the Tavily client."""
        if self._client is None:
            from tavily import TavilyClient
            self._client = TavilyClient(api_key=self.api_key)
        return self._client
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        import time
        
        current_time = time.time()
        
        if self._last_reset is None:
            self._last_reset = current_time
            self._request_count = 0
            return True
        
        # Reset counter every minute
        if current_time - self._last_reset > 60:
            self._last_reset = current_time
            self._request_count = 0
        
        if self._request_count >= self.MAX_REQUESTS_PER_MINUTE:
            return False
        
        self._request_count += 1
        return True
    
    async def execute(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic"
    ) -> str:
        """
        Search the web and return relevant results.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            search_depth: "basic" or "advanced"
            
        Returns:
            Search results with answer
        """
        if not self._check_rate_limit():
            return "Rate limit exceeded. Please try again later."
        
        try:
            client = self._get_client()
            
            # Run search in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.search(
                    query=query,
                    max_results=max_results,
                    search_depth=search_depth,
                    include_answer=True
                )
            )
            
            # Format response
            answer = response.get('answer', '')
            results = response.get('results', [])
            
            output_parts = []
            
            if answer:
                output_parts.append(f"Answer: {answer}\n")
            
            if results:
                output_parts.append("Sources:")
                for i, r in enumerate(results[:max_results], 1):
                    title = r.get('title', 'No title')
                    content = r.get('content', '')[:200]
                    url = r.get('url', '')
                    output_parts.append(f"\n{i}. {title}")
                    output_parts.append(f"   {content}")
                    if url:
                        output_parts.append(f"   URL: {url}")
            
            return "\n".join(output_parts)
            
        except Exception as e:
            return f"Web search error: {e}"
    
    async def search(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search the web and return structured results.
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            List of result dicts
        """
        if not self._check_rate_limit():
            return []
        
        try:
            client = self._get_client()
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.search(
                    query=query,
                    max_results=max_results,
                    search_depth="basic"
                )
            )
            
            return response.get('results', [])
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    async def get_answer(self, query: str) -> str:
        """
        Get a direct answer to a question using web search.
        
        Args:
            query: Question to answer
            
        Returns:
            Direct answer
        """
        if not self._check_rate_limit():
            return "Rate limit exceeded."
        
        try:
            client = self._get_client()
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.search(
                    query=query,
                    max_results=3,
                    search_depth="basic",
                    include_answer=True
                )
            )
            
            return response.get('answer', 'No answer found')
            
        except Exception as e:
            return f"Error: {e}"
    
    async def get_current_info(
        self,
        topic: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get current information about a topic.
        
        Args:
            topic: Topic to research
            context: Additional context
            
        Returns:
            Dict with 'answer', 'sources', 'timestamp'
        """
        import datetime
        
        query = topic
        if context:
            query = f"{topic} {context}"
        
        result = await self.execute(query, max_results=3)
        
        return {
            'answer': result,
            'query': query,
            'timestamp': datetime.datetime.now().isoformat()
        }
