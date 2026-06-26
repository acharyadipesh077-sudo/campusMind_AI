"""Base scraper class with retry logic and caching."""
import logging
import random
import time
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from mcp_server.cache import TTLCache

logger = logging.getLogger(__name__)


class BaseScraper:
    """Base class for all IOE scrapers with built-in retry and caching."""

    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/109.0",
    ]

    def __init__(self, cache_ttl_minutes: int = 30) -> None:
        """Initialize scraper.

        Args:
            cache_ttl_minutes: Cache TTL in minutes
        """
        self.session = self._create_session()
        self.cache = TTLCache(ttl_minutes=cache_ttl_minutes)
        self.base_url = "https://ioe.tu.edu.np"

    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy.

        Returns:
            Configured requests session
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with random user agent.

        Returns:
            Headers dictionary
        """
        return {
            "User-Agent": random.choice(self.USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    def fetch(self, url: str, use_cache: bool = True, timeout: int = 10) -> Optional[str]:
        """Fetch URL content with retry logic and caching.

        Args:
            url: URL to fetch
            use_cache: Whether to use cache
            timeout: Request timeout in seconds

        Returns:
            Page content or None on failure
        """
        cache_key = f"fetch_{url}"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info(f"Cache hit for {url}")
                return cached

        try:
            response = self.session.get(
                url, headers=self._get_headers(), timeout=timeout
            )
            response.raise_for_status()
            content = response.text
            if use_cache:
                self.cache.set(cache_key, content)
            logger.info(f"Successfully fetched {url}")
            return content
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching {url}")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error fetching {url}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching {url}: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {str(e)}")
            return None

    def parse_json(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch and parse JSON from URL.

        Args:
            url: URL to fetch

        Returns:
            Parsed JSON or None on failure
        """
        try:
            cache_key = f"json_{url}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

            response = self.session.get(
                url, headers=self._get_headers(), timeout=10
            )
            response.raise_for_status()
            data = response.json()
            self.cache.set(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error parsing JSON from {url}: {str(e)}")
            return None

    def close(self) -> None:
        """Close session."""
        self.session.close()
