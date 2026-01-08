"""
Query Logs API - 쿼리 추적 로그 조회 API
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from app.utils.query_logger import get_query_logger
import json

router = APIRouter()


@router.get("/api/query-logs/recent")
async def get_recent_logs(limit: int = 100):
    """최근 쿼리 로그 조회"""
    try:
        query_logger = get_query_logger()
        await query_logger.init_database()  # Ensure table exists

        logs = await query_logger.get_recent_logs(limit=limit)

        # Parse JSON fields
        for log in logs:
            if log.get('core_keywords'):
                try:
                    log['core_keywords'] = json.loads(log['core_keywords'])
                except:
                    pass
            if log.get('sub_keywords'):
                try:
                    log['sub_keywords'] = json.loads(log['sub_keywords'])
                except:
                    pass
            if log.get('db_pages_found'):
                try:
                    log['db_pages_found'] = json.loads(log['db_pages_found'])
                except:
                    pass

        return {
            "status": "success",
            "count": len(logs),
            "logs": logs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/query-logs/{log_id}")
async def get_log_by_id(log_id: str):
    """특정 로그 상세 조회"""
    try:
        query_logger = get_query_logger()
        await query_logger.init_database()

        log = await query_logger.get_log_by_id(log_id)

        if not log:
            raise HTTPException(status_code=404, detail=f"Log {log_id} not found")

        # Parse JSON fields
        if log.get('core_keywords'):
            try:
                log['core_keywords'] = json.loads(log['core_keywords'])
            except:
                pass
        if log.get('sub_keywords'):
            try:
                log['sub_keywords'] = json.loads(log['sub_keywords'])
            except:
                pass
        if log.get('db_pages_found'):
            try:
                log['db_pages_found'] = json.loads(log['db_pages_found'])
            except:
                pass

        return {
            "status": "success",
            "log": log
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/query-logs/session/{sessionid}")
async def get_logs_by_session(sessionid: str, limit: int = 50):
    """세션별 로그 조회"""
    try:
        query_logger = get_query_logger()
        await query_logger.init_database()

        logs = await query_logger.get_logs_by_session(sessionid=sessionid, limit=limit)

        # Parse JSON fields
        for log in logs:
            if log.get('core_keywords'):
                try:
                    log['core_keywords'] = json.loads(log['core_keywords'])
                except:
                    pass
            if log.get('sub_keywords'):
                try:
                    log['sub_keywords'] = json.loads(log['sub_keywords'])
                except:
                    pass
            if log.get('db_pages_found'):
                try:
                    log['db_pages_found'] = json.loads(log['db_pages_found'])
                except:
                    pass

        return {
            "status": "success",
            "sessionid": sessionid,
            "count": len(logs),
            "logs": logs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/query-logs/stats/search-types")
async def get_search_type_stats():
    """검색 타입별 통계"""
    try:
        query_logger = get_query_logger()
        await query_logger.init_database()

        stats = await query_logger.get_stats_by_search_type()

        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/query-logs/trace/{log_id}")
async def get_query_trace(log_id: str):
    """
    쿼리 추적 정보를 단계별로 포맷팅하여 반환

    역추적 및 디버깅에 최적화된 포맷
    """
    try:
        query_logger = get_query_logger()
        await query_logger.init_database()

        log = await query_logger.get_log_by_id(log_id)

        if not log:
            raise HTTPException(status_code=404, detail=f"Log {log_id} not found")

        # 단계별 트레이스 생성
        trace = {
            "log_id": log['id'],
            "timestamp": log['timestamp'],
            "query_info": {
                "userid": log['userid'],
                "sessionid": log['sessionid'],
                "isbn": log['isbn'],
                "query": log['query'],
                "select_text": log['select_text']
            },
            "steps": []
        }

        # Step 1: TOC 로드
        if log['toc_type']:
            trace['steps'].append({
                "step": 1,
                "name": "TOC Load",
                "data": {
                    "toc_type": log['toc_type'],
                    "toc_length": log['toc_length'],
                    "time_ms": log['toc_load_time_ms']
                }
            })

        # Step 2: 대화 컨텍스트
        trace['steps'].append({
            "step": 2,
            "name": "Conversation Context",
            "data": {
                "has_history": log['has_conversation_history'],
                "conversation_turns": log['conversation_turns'],
                "context_dependent": log['context_dependent']
            }
        })

        # Step 3: 캐시 체크
        if log['cache_check_type']:
            trace['steps'].append({
                "step": 3,
                "name": "Cache Check",
                "data": {
                    "cache_type": log['cache_check_type'],
                    "cache_hit": log['cache_hit'],
                    "time_ms": log['cache_hit_time_ms']
                },
                "result": "HIT" if log['cache_hit'] else "MISS"
            })

        # Step 4: LLM Search Type 결정
        if log['search_type']:
            keywords_data = {}
            if log['core_keywords']:
                try:
                    keywords_data['core'] = json.loads(log['core_keywords'])
                except:
                    keywords_data['core'] = log['core_keywords']
            if log['sub_keywords']:
                try:
                    keywords_data['sub'] = json.loads(log['sub_keywords'])
                except:
                    keywords_data['sub'] = log['sub_keywords']

            trace['steps'].append({
                "step": 4,
                "name": "LLM Search Type Decision",
                "data": {
                    "search_type": log['search_type'],
                    "reason": log['search_reason'],
                    "keywords": keywords_data,
                    "page_range": log['page_range'],
                    "use_previous_context": log['use_previous_context'],
                    "time_ms": log['llm_search_type_request_time_ms']
                },
                "decision": log['search_type'].upper()
            })

        # Step 5: DB 검색
        if log['db_search_method']:
            pages_found = []
            if log['db_pages_found']:
                try:
                    pages_found = json.loads(log['db_pages_found'])
                except:
                    pages_found = log['db_pages_found']

            db_step = {
                "step": 5,
                "name": f"DB Search ({log['db_search_method'].upper()})",
                "data": {
                    "method": log['db_search_method'],
                    "query": log['db_search_query'],
                    "results_count": log['db_results_count'],
                    "pages_found": pages_found,
                    "time_ms": log['db_search_time_ms']
                }
            }

            # FTS5 상세
            if log['db_search_method'] == 'fts5' and log['fts5_keywords_used']:
                try:
                    db_step['data']['fts5_details'] = {
                        "keywords_used": json.loads(log['fts5_keywords_used']),
                        "page_restriction": log['fts5_page_restriction'],
                        "fallback_triggered": log['fts5_fallback_triggered']
                    }
                except:
                    pass

            # Qdrant 상세
            if log['db_search_method'] == 'qdrant':
                db_step['data']['qdrant_details'] = {
                    "embedding_time_ms": log['qdrant_query_embedding_time_ms'],
                    "search_time_ms": log['qdrant_search_time_ms'],
                    "top_k": log['qdrant_top_k'],
                    "fallback_to_keyword": log['qdrant_fallback_to_keyword']
                }

            trace['steps'].append(db_step)

        # Step 6: LLM 답변 생성
        if log['llm_answer_request_time_ms']:
            trace['steps'].append({
                "step": 6,
                "name": "LLM Answer Generation",
                "data": {
                    "prompt_length": log['llm_prompt_length'],
                    "context_chunks_count": log['llm_context_chunks_count'],
                    "answer_length": log['answer_length'],
                    "time_ms": log['llm_answer_request_time_ms']
                }
            })

        # 최종 결과
        trace['result'] = {
            "total_time_ms": log['total_time_ms'],
            "answer_preview": log['answer_preview'],
            "chunks_returned": log['chunks_returned'],
            "visual_elements_count": log['visual_elements_count'],
            "has_error": log['has_error'],
            "error_step": log['error_step'],
            "error_message": log['error_message']
        }

        # 성능 분석
        trace['performance_breakdown'] = {
            "toc_load_ms": log['toc_load_time_ms'] or 0,
            "cache_check_ms": log['cache_hit_time_ms'] or 0,
            "llm_search_type_ms": log['llm_search_type_request_time_ms'] or 0,
            "db_search_ms": log['db_search_time_ms'] or 0,
            "llm_answer_ms": log['llm_answer_request_time_ms'] or 0,
            "total_ms": log['total_time_ms'] or 0
        }

        return {
            "status": "success",
            "trace": trace
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
