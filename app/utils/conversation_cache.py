"""
Conversation Cache with LRU and TTL
Prevents memory leaks from unlimited conversation storage
"""
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio


class ConversationCache:
    """
    Thread-safe conversation cache with LRU eviction and TTL

    Features:
    - LRU eviction when max size reached
    - TTL-based cleanup (default 24 hours)
    - Thread-safe operations with async locks
    """

    def __init__(self, max_size: int = 1000, max_age_hours: int = 24):
        self.cache: OrderedDict[str, List[Dict]] = OrderedDict()
        self.timestamps: Dict[str, datetime] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        self.max_size = max_size
        self.max_age = timedelta(hours=max_age_hours)
        self._global_lock = asyncio.Lock()

    async def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create lock for conversation"""
        async with self._global_lock:
            if key not in self.locks:
                self.locks[key] = asyncio.Lock()
            return self.locks[key]

    def _cleanup_old(self):
        """Remove expired conversations"""
        now = datetime.utcnow()
        to_remove = [
            key for key, timestamp in self.timestamps.items()
            if now - timestamp > self.max_age
        ]
        for key in to_remove:
            del self.cache[key]
            del self.timestamps[key]
            if key in self.locks:
                del self.locks[key]

    def _evict_lru(self):
        """Evict least recently used conversation"""
        if len(self.cache) >= self.max_size:
            # Remove oldest (first) item in OrderedDict
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
            if oldest_key in self.locks:
                del self.locks[oldest_key]

    async def get(self, conversation_id: str) -> List[Dict]:
        """Get conversation history"""
        self._cleanup_old()

        lock = await self._get_lock(conversation_id)
        async with lock:
            if conversation_id in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(conversation_id)
                return self.cache[conversation_id].copy()
            return []

    async def append(self, conversation_id: str, messages: List[Dict]):
        """Append messages to conversation"""
        self._cleanup_old()
        self._evict_lru()

        lock = await self._get_lock(conversation_id)
        async with lock:
            if conversation_id not in self.cache:
                self.cache[conversation_id] = []

            self.cache[conversation_id].extend(messages)
            self.timestamps[conversation_id] = datetime.utcnow()

            # Move to end (most recently used)
            self.cache.move_to_end(conversation_id)

    async def set(self, conversation_id: str, messages: List[Dict]):
        """Set entire conversation history"""
        self._cleanup_old()
        self._evict_lru()

        lock = await self._get_lock(conversation_id)
        async with lock:
            self.cache[conversation_id] = messages
            self.timestamps[conversation_id] = datetime.utcnow()
            self.cache.move_to_end(conversation_id)

    async def delete(self, conversation_id: str) -> bool:
        """Delete conversation"""
        lock = await self._get_lock(conversation_id)
        async with lock:
            if conversation_id in self.cache:
                del self.cache[conversation_id]
                del self.timestamps[conversation_id]
                del self.locks[conversation_id]
                return True
            return False

    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)

    def stats(self) -> Dict:
        """Get cache statistics"""
        self._cleanup_old()
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'max_age_hours': self.max_age.total_seconds() / 3600,
            'active_conversations': len(self.cache),
            'active_locks': len(self.locks)
        }
