"""Thread-safe cache system with TTL and disk persistence."""
import json
import time
from datetime import timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Optional


class CacheStats:
    """Statistics for cache operations."""

    def __init__(self) -> None:
        self.hits: int = 0
        self.misses: int = 0
        self.size_bytes: int = 0
        self.entries: int = 0

    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


class TTLCache:
    """Thread-safe cache with TTL (Time To Live) and disk persistence."""

    def __init__(
        self,
        ttl_minutes: int = 30,
        cache_dir: str = "cache",
    ) -> None:
        """Initialize cache.

        Args:
            ttl_minutes: Time to live in minutes
            cache_dir: Directory for cache files
        """
        self.ttl = timedelta(minutes=ttl_minutes)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._lock = Lock()
        self.stats = CacheStats()
        self._load_from_disk()

    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key."""
        safe_key = key.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_key}.json"

    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        key = data.get("key")
                        value = data.get("value")
                        timestamp = data.get("timestamp")
                        if key and value is not None and timestamp:
                            self._cache[key] = (value, float(timestamp))
                except (json.JSONDecodeError, KeyError):
                    cache_file.unlink(missing_ok=True)
        except Exception as e:
            print(f"Error loading cache from disk: {e}")

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired_keys = [
            key
            for key, (_, timestamp) in self._cache.items()
            if now - timestamp > self.ttl.total_seconds()
        ]
        for key in expired_keys:
            del self._cache[key]
            cache_file = self._get_cache_file(key)
            cache_file.unlink(missing_ok=True)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        with self._lock:
            self._cleanup_expired()
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp <= self.ttl.total_seconds():
                    self.stats.hits += 1
                    return value
                else:
                    del self._cache[key]
                    self._get_cache_file(key).unlink(missing_ok=True)
            self.stats.misses += 1
            return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            timestamp = time.time()
            self._cache[key] = (value, timestamp)
            self._save_to_disk(key, value, timestamp)
            self.stats.entries = len(self._cache)
            self.stats.size_bytes = sum(
                len(json.dumps(v).encode()) for _, (v, _) in self._cache.items()
            )

    def _save_to_disk(self, key: str, value: Any, timestamp: float) -> None:
        """Save cache entry to disk."""
        try:
            cache_file = self._get_cache_file(key)
            data = {"key": key, "value": value, "timestamp": timestamp}
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving cache to disk: {e}")

    def clear(self) -> None:
        """Clear all cache."""
        with self._lock:
            self._cache.clear()
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink(missing_ok=True)
            self.stats = CacheStats()

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats


def cached(ttl_minutes: int = 30):
    """Decorator for caching function results.

    Args:
        ttl_minutes: Time to live in minutes
    """
    cache = TTLCache(ttl_minutes=ttl_minutes)

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_key = f"{func.__name__}_{args}_{kwargs}"
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result

        return wrapper

    return decorator
