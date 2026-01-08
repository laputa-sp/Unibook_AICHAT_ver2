"""
Query Logger - 사용자 쿼리부터 응답까지 전체 흐름을 추적하는 로깅 시스템

DB 저장용 구조화된 로그를 생성하여 역추적 및 원인 분석을 지원합니다.
"""

import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
import aiosqlite
from pathlib import Path


class QueryLogger:
    """쿼리 처리 과정을 단계별로 기록하는 로거"""

    def __init__(self, db_path: str = "uploads/app.db"):
        self.db_path = db_path
        self.current_log: Optional[Dict[str, Any]] = None

    async def init_database(self):
        """로그 테이블 초기화"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    -- 요청 정보
                    userid TEXT,
                    sessionid TEXT,
                    isbn TEXT,
                    query TEXT,
                    select_text TEXT,

                    -- Step 1: TOC 로드
                    toc_type TEXT,  -- 'simple' or 'core_summary'
                    toc_length INTEGER,
                    toc_load_time_ms REAL,

                    -- Step 2: 대화 컨텍스트
                    has_conversation_history BOOLEAN,
                    conversation_turns INTEGER,
                    context_dependent BOOLEAN,

                    -- Step 3: 캐시 체크
                    cache_check_type TEXT,  -- 'full' or 'llm' or 'none'
                    cache_hit BOOLEAN,
                    cache_hit_time_ms REAL,

                    -- Step 4: LLM Search Type 결정
                    llm_search_type_request_time_ms REAL,
                    search_type TEXT,  -- keyword, semantic, toc, summary, etc.
                    search_reason TEXT,
                    core_keywords TEXT,  -- JSON array
                    sub_keywords TEXT,   -- JSON array
                    page_range TEXT,
                    use_previous_context BOOLEAN,

                    -- Step 5: DB 검색
                    db_search_method TEXT,  -- 'fts5', 'qdrant', 'toc_lookup', 'none'
                    db_search_query TEXT,  -- 실제 실행한 쿼리
                    db_search_time_ms REAL,
                    db_results_count INTEGER,
                    db_pages_found TEXT,  -- JSON array of page numbers

                    -- Step 5-1: FTS5 상세 (keyword 타입인 경우)
                    fts5_keywords_used TEXT,  -- JSON: {core: [], sub: []}
                    fts5_page_restriction TEXT,  -- 페이지 제한 여부
                    fts5_fallback_triggered BOOLEAN,  -- 전체 검색 fallback 여부

                    -- Step 5-2: Qdrant 상세 (semantic 타입인 경우)
                    qdrant_query_embedding_time_ms REAL,
                    qdrant_search_time_ms REAL,
                    qdrant_top_k INTEGER,
                    qdrant_fallback_to_keyword BOOLEAN,

                    -- Step 6: LLM 답변 생성
                    llm_answer_request_time_ms REAL,
                    llm_prompt_length INTEGER,
                    llm_context_chunks_count INTEGER,
                    answer_length INTEGER,

                    -- Step 7: 결과 및 성능
                    total_time_ms REAL,
                    answer_preview TEXT,  -- 처음 200자
                    chunks_returned INTEGER,
                    visual_elements_count INTEGER,

                    -- 오류 추적
                    has_error BOOLEAN DEFAULT 0,
                    error_step TEXT,
                    error_message TEXT,

                    -- 메타데이터
                    server_version TEXT,
                    llm_engine TEXT  -- 'vllm' or 'ollama'
                )
            """)

            # 인덱스 생성 (빠른 조회를 위해)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_logs_timestamp
                ON query_logs(timestamp)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_logs_sessionid
                ON query_logs(sessionid)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_logs_search_type
                ON query_logs(search_type)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_logs_isbn
                ON query_logs(isbn)
            """)

            await db.commit()

    def start_log(self, userid: str, sessionid: str, isbn: str, query: str,
                  select_text: Optional[str] = None) -> str:
        """새 쿼리 로그 시작"""
        log_id = str(uuid.uuid4())
        self.current_log = {
            'id': log_id,
            'timestamp': datetime.now().isoformat(),
            'userid': userid,
            'sessionid': sessionid,
            'isbn': isbn,
            'query': query,
            'select_text': select_text,
            'start_time': time.time(),
            'steps': []  # 각 단계별 상세 로그
        }
        return log_id

    def log_step(self, step_name: str, data: Dict[str, Any]):
        """처리 단계 기록"""
        if not self.current_log:
            return

        self.current_log['steps'].append({
            'name': step_name,
            'timestamp': time.time(),
            'data': data
        })

    def log_toc_load(self, toc_type: str, toc_length: int, load_time_ms: float):
        """TOC 로드 단계 기록"""
        self.log_step('toc_load', {
            'toc_type': toc_type,
            'toc_length': toc_length,
            'load_time_ms': load_time_ms
        })

    def log_conversation_context(self, has_history: bool, turns: int,
                                  context_dependent: bool):
        """대화 컨텍스트 단계 기록"""
        self.log_step('conversation_context', {
            'has_history': has_history,
            'turns': turns,
            'context_dependent': context_dependent
        })

    def log_cache_check(self, cache_type: str, hit: bool, hit_time_ms: float):
        """캐시 체크 단계 기록"""
        self.log_step('cache_check', {
            'cache_type': cache_type,
            'hit': hit,
            'hit_time_ms': hit_time_ms
        })

    def log_llm_search_type(self, request_time_ms: float, search_type: str,
                           reason: str, core_keywords: List[str],
                           sub_keywords: List[str], page_range: Optional[str],
                           use_previous_context: bool):
        """LLM 검색 타입 결정 단계 기록"""
        self.log_step('llm_search_type', {
            'request_time_ms': request_time_ms,
            'search_type': search_type,
            'reason': reason,
            'core_keywords': core_keywords,
            'sub_keywords': sub_keywords,
            'page_range': page_range,
            'use_previous_context': use_previous_context
        })

    def log_db_search(self, method: str, query: str, time_ms: float,
                     results_count: int, pages_found: List[int],
                     extra_data: Optional[Dict] = None):
        """DB 검색 단계 기록"""
        data = {
            'method': method,
            'query': query,
            'time_ms': time_ms,
            'results_count': results_count,
            'pages_found': pages_found
        }
        if extra_data:
            data.update(extra_data)

        self.log_step('db_search', data)

    def log_llm_answer(self, request_time_ms: float, prompt_length: int,
                      chunks_count: int, answer_length: int):
        """LLM 답변 생성 단계 기록"""
        self.log_step('llm_answer', {
            'request_time_ms': request_time_ms,
            'prompt_length': prompt_length,
            'chunks_count': chunks_count,
            'answer_length': answer_length
        })

    def log_error(self, step: str, error_message: str):
        """오류 기록"""
        if not self.current_log:
            return

        self.current_log['has_error'] = True
        self.current_log['error_step'] = step
        self.current_log['error_message'] = error_message
        self.log_step('error', {
            'step': step,
            'message': error_message
        })

    async def save_log(self, answer: str, chunks_count: int,
                      visual_elements_count: int,
                      server_version: str = "1.0.0",
                      llm_engine: str = "vllm"):
        """로그를 DB에 저장"""
        if not self.current_log:
            return

        total_time = (time.time() - self.current_log['start_time']) * 1000

        # 각 단계별 데이터 추출
        steps_dict = {step['name']: step['data'] for step in self.current_log['steps']}

        toc_load = steps_dict.get('toc_load', {})
        conv_context = steps_dict.get('conversation_context', {})
        cache_check = steps_dict.get('cache_check', {})
        search_type_data = steps_dict.get('llm_search_type', {})
        db_search = steps_dict.get('db_search', {})
        llm_answer = steps_dict.get('llm_answer', {})

        # FTS5/Qdrant 상세 데이터
        fts5_data = db_search.get('fts5', {}) if db_search.get('method') == 'fts5' else {}
        qdrant_data = db_search.get('qdrant', {}) if db_search.get('method') == 'qdrant' else {}

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO query_logs (
                    id, timestamp, userid, sessionid, isbn, query, select_text,
                    toc_type, toc_length, toc_load_time_ms,
                    has_conversation_history, conversation_turns, context_dependent,
                    cache_check_type, cache_hit, cache_hit_time_ms,
                    llm_search_type_request_time_ms, search_type, search_reason,
                    core_keywords, sub_keywords, page_range, use_previous_context,
                    db_search_method, db_search_query, db_search_time_ms,
                    db_results_count, db_pages_found,
                    fts5_keywords_used, fts5_page_restriction, fts5_fallback_triggered,
                    qdrant_query_embedding_time_ms, qdrant_search_time_ms,
                    qdrant_top_k, qdrant_fallback_to_keyword,
                    llm_answer_request_time_ms, llm_prompt_length,
                    llm_context_chunks_count, answer_length,
                    total_time_ms, answer_preview, chunks_returned,
                    visual_elements_count, has_error, error_step, error_message,
                    server_version, llm_engine
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?
                )
            """, (
                self.current_log['id'],
                self.current_log['timestamp'],
                self.current_log['userid'],
                self.current_log['sessionid'],
                self.current_log['isbn'],
                self.current_log['query'],
                self.current_log.get('select_text'),

                toc_load.get('toc_type'),
                toc_load.get('toc_length'),
                toc_load.get('load_time_ms'),

                conv_context.get('has_history'),
                conv_context.get('turns'),
                conv_context.get('context_dependent'),

                cache_check.get('cache_type'),
                cache_check.get('hit'),
                cache_check.get('hit_time_ms'),

                search_type_data.get('request_time_ms'),
                search_type_data.get('search_type'),
                search_type_data.get('reason'),
                json.dumps(search_type_data.get('core_keywords', []), ensure_ascii=False),
                json.dumps(search_type_data.get('sub_keywords', []), ensure_ascii=False),
                search_type_data.get('page_range'),
                search_type_data.get('use_previous_context'),

                db_search.get('method'),
                db_search.get('query'),
                db_search.get('time_ms'),
                db_search.get('results_count'),
                json.dumps(db_search.get('pages_found', []), ensure_ascii=False),

                json.dumps(fts5_data.get('keywords_used', {}), ensure_ascii=False),
                fts5_data.get('page_restriction'),
                fts5_data.get('fallback_triggered'),

                qdrant_data.get('embedding_time_ms'),
                qdrant_data.get('search_time_ms'),
                qdrant_data.get('top_k'),
                qdrant_data.get('fallback_to_keyword'),

                llm_answer.get('request_time_ms'),
                llm_answer.get('prompt_length'),
                llm_answer.get('chunks_count'),
                llm_answer.get('answer_length'),

                total_time,
                answer[:200] if answer else None,
                chunks_count,
                visual_elements_count,

                self.current_log.get('has_error', False),
                self.current_log.get('error_step'),
                self.current_log.get('error_message'),

                server_version,
                llm_engine
            ))

            await db.commit()

        # 로그 초기화
        self.current_log = None

    async def get_log_by_id(self, log_id: str) -> Optional[Dict[str, Any]]:
        """ID로 로그 조회"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM query_logs WHERE id = ?",
                (log_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return dict(row)
        return None

    async def get_logs_by_session(self, sessionid: str, limit: int = 50) -> List[Dict[str, Any]]:
        """세션별 로그 조회"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM query_logs
                   WHERE sessionid = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (sessionid, limit)
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """최근 로그 조회"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM query_logs
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (limit,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def get_stats_by_search_type(self) -> Dict[str, Any]:
        """검색 타입별 통계"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT
                    search_type,
                    COUNT(*) as count,
                    AVG(total_time_ms) as avg_time_ms,
                    AVG(db_search_time_ms) as avg_db_time_ms,
                    AVG(llm_answer_request_time_ms) as avg_llm_time_ms,
                    SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cache_hits
                FROM query_logs
                WHERE search_type IS NOT NULL
                GROUP BY search_type
            """) as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        'search_type': row[0],
                        'count': row[1],
                        'avg_time_ms': row[2],
                        'avg_db_time_ms': row[3],
                        'avg_llm_time_ms': row[4],
                        'cache_hits': row[5]
                    }
                    for row in rows
                ]


# 전역 인스턴스
_query_logger: Optional[QueryLogger] = None


def get_query_logger() -> QueryLogger:
    """Query Logger 싱글톤 인스턴스 반환"""
    global _query_logger
    if _query_logger is None:
        _query_logger = QueryLogger()
    return _query_logger
