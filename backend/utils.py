import asyncio
import functools
import time
import logging
from typing import Any, Callable, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

# Thread pool for CPU-intensive tasks
thread_pool = ThreadPoolExecutor(max_workers=4)


def async_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying async functions with exponential backoff"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")

                    if attempt < max_attempts - 1:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")

            raise last_exception

        return wrapper

    return decorator


def log_performance(func: Callable) -> Callable:
    """Decorator to log function performance metrics"""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()

        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
            raise

    return wrapper


async def run_in_thread(func: Callable, *args, **kwargs) -> Any:
    """Run a blocking function in a thread pool"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(thread_pool, func, *args, **kwargs)


class AsyncCache:
    """Simple async-safe cache implementation"""

    def __init__(self, ttl: int = 300):  # 5 minutes default TTL
        self._cache: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self.ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() - entry['timestamp'] < self.ttl:
                    return entry['value']
                else:
                    del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        with self._lock:
            self._cache[key] = {
                'value': value,
                'timestamp': time.time()
            }

    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()


# Global cache instance
cache = AsyncCache(ttl=600)  # 10 minutes


class RateLimiter:
    """Simple rate limiter for API calls"""

    def __init__(self, max_calls: int = 100, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: Dict[str, list] = {}
        self._lock = threading.Lock()

    def is_allowed(self, identifier: str) -> bool:
        """Check if a request is allowed"""
        with self._lock:
            now = time.time()

            if identifier not in self.calls:
                self.calls[identifier] = []

            # Remove old calls outside the time window
            self.calls[identifier] = [
                call_time for call_time in self.calls[identifier]
                if now - call_time < self.time_window
            ]

            # Check if under limit
            if len(self.calls[identifier]) < self.max_calls:
                self.calls[identifier].append(now)
                return True

            return False


# Global rate limiter
rate_limiter = RateLimiter(max_calls=50, time_window=60)


async def batch_process(items: list, func: Callable, batch_size: int = 10) -> list:
    """Process items in batches to avoid overwhelming APIs"""
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_tasks = [func(item) for item in batch]

        try:
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)

            # Small delay between batches
            if i + batch_size < len(items):
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            results.extend([None] * len(batch))

    return results


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely parse JSON string with fallback"""
    try:
        import json
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"JSON parsing failed: {e}")
        return default


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


async def timeout_wrapper(coro, timeout_seconds: float = 30.0):
    """Wrap coroutine with timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.error(f"Operation timed out after {timeout_seconds} seconds")
        raise


class AsyncLock:
    """Context manager for async locks"""

    def __init__(self):
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await self._lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()


# Utility functions for text processing
def clean_text(text: str) -> str:
    """Clean and normalize text"""
    import re

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\-\.,!?;:]', '', text)

    return text


def extract_keywords(text: str, max_keywords: int = 10) -> list:
    """Extract keywords from text (simple implementation)"""
    import re
    from collections import Counter

    # Simple keyword extraction (in production, use proper NLP)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    # Filter out common stop words
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'can', 'must'
    }

    keywords = [word for word in words if word not in stop_words]

    # Return most common keywords
    return [word for word, count in Counter(keywords).most_common(max_keywords)]
