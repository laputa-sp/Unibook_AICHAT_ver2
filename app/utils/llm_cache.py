"""
LLM Response Cache
LLM 응답을 캐싱하여 반복 쿼리 성능 향상
"""
import hashlib
import time
import logging
from typing import Optional, Dict, Any
from collections import OrderedDict


class LLMCache:
    """
    LLM 응답 캐시

    Features:
    - LRU eviction policy
    - TTL (Time To Live)
    - Query + Context 기반 캐싱
    """

    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        """
        Args:
            max_size: 최대 캐시 항목 수
            ttl_hours: 캐시 유효 시간 (시간)
        """
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        self.logger = logging.getLogger(__name__)

        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expirations': 0
        }

    def _get_cache_key(self, query: str, context: str = "") -> str:
        """
        캐시 키 생성 (쿼리 + 컨텍스트 해시)

        Args:
            query: 사용자 쿼리
            context: 컨텍스트 (ISBN + 검색 유형, 실제 내용 제외)

        Returns:
            MD5 해시 키
        """
        # Normalize query (소문자, 공백 정리, 구두점 제거)
        normalized_query = query.lower().strip()

        # 질문 정규화: 구두점, 조사 제거, 공백 정리
        import re
        # 1. 구두점 제거
        normalized_query = re.sub(r'[?!.,\s]+', ' ', normalized_query).strip()

        # 2. 한글 조사 제거 (은/는/이/가/을/를/와/과/의/에/에서/으로/로)
        # 단어에 붙어있는 조사도 제거 (예: "내향형과" -> "내향형")
        particles = [
            '은는', '이가', '을를', '와과',  # 복합 조사
            '에서', '으로',  # 2글자 조사
            '는', '은', '이', '가', '을', '를', '와', '과', '의', '에', '로'  # 1글자 조사
        ]
        for particle in particles:
            # 한글 뒤에 붙은 조사 제거 (예: "내향형과" -> "내향형 ")
            normalized_query = re.sub(rf'([가-힣]){particle}(\s|$)', r'\1 ', normalized_query)
            # 공백 사이의 조사 제거
            normalized_query = re.sub(rf'\s+{particle}\s+', ' ', normalized_query)

        # 3. 공백 정리
        normalized_query = re.sub(r'\s+', ' ', normalized_query).strip()

        # 4. 동의어 통일
        synonyms = {
            '무엇': '뭐',
            '차이점': '차이',
            '차이는': '차이',
            '다른점': '차이',
            '구별': '차이'
        }
        for original, replacement in synonyms.items():
            normalized_query = normalized_query.replace(original, replacement)

        # Context는 ISBN + search_type만 포함 (실제 검색 내용 제외)
        # 이렇게 하면 같은 질문에 대해서는 항상 같은 캐시 키가 생성됨
        normalized_context = context.lower().strip()

        # Create hash key
        text = f"{normalized_query}|||{normalized_context}"
        hash_key = hashlib.md5(text.encode('utf-8')).hexdigest()

        # Debug logging
        self.logger.debug(f"🔑 Cache key generation:")
        self.logger.debug(f"   Original: '{query}'")
        self.logger.debug(f"   Normalized: '{normalized_query}'")
        self.logger.debug(f"   Context: '{context}' -> '{normalized_context}'")
        self.logger.debug(f"   Hash: {hash_key}")

        return hash_key

    def get(self, query: str, context: str = "") -> Optional[Dict[str, Any]]:
        """
        캐시에서 응답 조회

        Args:
            query: 사용자 쿼리
            context: 컨텍스트

        Returns:
            캐시된 응답 또는 None
        """
        key = self._get_cache_key(query, context)

        if key not in self.cache:
            self.stats['misses'] += 1
            return None

        # Get entry
        entry = self.cache[key]

        # Check TTL
        age = time.time() - entry['timestamp']
        if age > self.ttl_seconds:
            # Expired
            del self.cache[key]
            self.stats['expirations'] += 1
            self.stats['misses'] += 1
            self.logger.info(f"Cache expired (age: {age:.1f}s): {query[:50]}")
            return None

        # Move to end (LRU)
        self.cache.move_to_end(key)

        self.stats['hits'] += 1
        self.logger.info(f"✅ Cache hit (age: {age:.1f}s): {query[:50]}")

        return entry['response']

    def set(self, query: str, context: str, response: Dict[str, Any]) -> None:
        """
        캐시에 응답 저장

        Args:
            query: 사용자 쿼리
            context: 컨텍스트
            response: LLM 응답
        """
        key = self._get_cache_key(query, context)

        # Check if already exists (update)
        if key in self.cache:
            del self.cache[key]

        # Check size limit
        if len(self.cache) >= self.max_size:
            # Remove oldest (first item)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.stats['evictions'] += 1
            self.logger.info(f"Cache eviction (size: {self.max_size})")

        # Add new entry
        self.cache[key] = {
            'response': response,
            'timestamp': time.time(),
            'query': query[:100],  # For debugging
            'context': context[:100]
        }

        self.logger.info(f"💾 Cache set: {query[:50]} (total: {len(self.cache)})")

    def clear(self) -> None:
        """캐시 전체 삭제"""
        count = len(self.cache)
        self.cache.clear()
        self.logger.info(f"Cache cleared ({count} entries)")

    def get_stats(self) -> Dict[str, Any]:
        """
        캐시 통계 조회

        Returns:
            통계 정보
        """
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0

        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': f"{hit_rate:.1f}%",
            'evictions': self.stats['evictions'],
            'expirations': self.stats['expirations'],
            'ttl_hours': self.ttl_seconds / 3600
        }

    def cleanup_expired(self) -> int:
        """
        만료된 항목 정리

        Returns:
            삭제된 항목 수
        """
        now = time.time()
        expired_keys = []

        for key, entry in self.cache.items():
            age = now - entry['timestamp']
            if age > self.ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired entries")

        return len(expired_keys)


# Global instance
_llm_cache = None


def get_llm_cache() -> LLMCache:
    """
    글로벌 LLM 캐시 인스턴스 조회

    Returns:
        LLMCache 인스턴스
    """
    global _llm_cache
    if _llm_cache is None:
        # 더 공격적인 캐싱: 크기 3배, TTL 3일
        _llm_cache = LLMCache(max_size=3000, ttl_hours=72)
    return _llm_cache


# Singleton instance for direct import
llm_cache = get_llm_cache()
