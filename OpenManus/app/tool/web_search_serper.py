import asyncio
import json
from typing import Dict, List, Optional, Union
import aiohttp
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, before_log, after_log

from app.config import config
from app.tool.base import BaseTool

# Set up logger
logger = logging.getLogger(__name__)

class WebSearch(BaseTool):
    name: str = "web_search"
    description: str = """Search for information using keywords.
    This tool helps find relevant travel articles, guides, and information from the web."""
    parameters: dict = {
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "(required) The search keyword or phrase for travel information.",
            },
            "page_size": {
                "type": "integer",
                "description": "(optional) Number of results per page. Default is 10.",
                "default": 3,
            },
            "page_num": {
                "type": "integer",
                "description": "(optional) Page number to retrieve. Default is 0.",
                "default": 0,
            }
        },
        "required": ["keyword"],
    }

    _API_URL = "https://google.serper.dev/search"  # API endpoint
    _API_KEY = config.get("serper_api_key", "your_api_key_here")  # API key from config

    async def execute(
        self,
        keyword: str,
        page_size: int = 5,
        page_num: int = 0
    ) -> Dict:
        if not keyword.strip():
            raise ValueError("Search keyword cannot be empty")

        payload = {
            "q": keyword,
            "num": page_size,
        }

        logger.info(f"Searching keyword via serper.dev: '{keyword}'")
        try:
            result = await self._send_request(self._API_URL, payload)
            return self._format_search_results(result)
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to search: {str(e)}",
                "data": None
            }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before=before_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO)
    )
    async def _send_request(self, url: str, payload: Dict) -> Dict:
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": self._API_KEY
        }
        timeout = aiohttp.ClientTimeout(total=10)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"Request failed: {response.status}, {text}")

                try:
                    return await response.json()
                except Exception as e:
                    text = await response.text()
                    raise Exception(f"Failed to parse JSON: {str(e)}\n{text[:300]}")

    def _format_search_results(self, result: Dict) -> Dict:
        # serper.dev response structure example, see official docs: https://serper.dev/docs
        try:
            if "organic" not in result:
                return {"status": "success", "message": "No results found", "results": []}

            formatted_results = []
            for item in result["organic"]:
                formatted_results.append({
                    "title": item.get("title", "No title"),
                    "content": item.get("snippet", "No content"),
                    "url": item.get("link", ""),
                    "source": "google.serper.dev",
                    "time": item.get("date", "")
                })

            json.dump(formatted_results, open("web_search_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

            return {
                "status": "success",
                "total_results": len(formatted_results),
                "results": formatted_results
            }
            
        

        
        except Exception as e:
            logger.error(f"Error formatting serper result: {str(e)}")
            return {"status": "error", "message": f"Formatting error: {str(e)}"}

    def _validate_response(self, response_data: Dict) -> Dict:
        """
        Validate the response data structure.

        Args:
            response_data (Dict): The API response data.

        Returns:
            Dict: The validated response data.

        Raises:
            Exception: If the response data is invalid.
        """
        if not isinstance(response_data, dict):
            raise Exception("Invalid response: not a dictionary")

        if "errmsg" in response_data and response_data["errmsg"] != "SUCCESS":
            error_msg = response_data.get("errmsg", "Unknown error")
            raise Exception(f"API returned error: {error_msg}")

        return response_data