"""
LLM Monitoring and Comparison Dashboard API
"""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from datetime import datetime

from app.utils.llm_comparison import llm_comparison
from app.utils.llm_cache import llm_cache

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


@router.get("/llm/stats")
async def get_llm_stats(
    engine: Optional[str] = Query(None, description="Filter by engine: 'vllm' or 'ollama'"),
    time_window: Optional[int] = Query(None, description="Time window in minutes, e.g. 60 for last hour")
):
    """
    LLM 엔진 통계 조회

    Args:
        engine: 특정 엔진 필터 (vllm/ollama), None이면 전체
        time_window: 최근 N분 데이터만 조회

    Returns:
        통계 데이터
    """
    try:
        stats = llm_comparison.get_stats(engine=engine, time_window_minutes=time_window)
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "time_window_minutes": time_window,
            "filter_engine": engine,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm/comparison")
async def get_comparison_summary(
    time_window: Optional[int] = Query(60, description="Time window in minutes")
):
    """
    vLLM vs Ollama 비교 요약

    최근 N분간의 데이터를 기반으로 두 엔진 비교

    Returns:
        비교 통계
    """
    try:
        stats = llm_comparison.get_stats(time_window_minutes=time_window)

        # 비교 요약 생성
        summary = {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "time_window_minutes": time_window,
            "engines": {}
        }

        for engine in ["vllm", "ollama"]:
            if engine in stats and stats[engine].get("total_requests", 0) > 0:
                summary["engines"][engine] = {
                    "total_requests": stats[engine]["total_requests"],
                    "success_rate": stats[engine]["success_rate"],
                    "avg_latency_ms": stats[engine]["latency"]["mean_ms"],
                    "p95_latency_ms": stats[engine]["latency"]["p95_ms"],
                    "avg_tokens_per_second": stats[engine]["throughput"]["avg_tokens_per_second"]
                }

        # 엔진 간 비교
        if "comparison" in stats:
            summary["comparison"] = stats["comparison"]

        # 추천 엔진
        if "vllm" in summary["engines"] and "ollama" in summary["engines"]:
            vllm_latency = summary["engines"]["vllm"]["avg_latency_ms"]
            ollama_latency = summary["engines"]["ollama"]["avg_latency_ms"]

            if vllm_latency < ollama_latency:
                summary["recommendation"] = {
                    "engine": "vllm",
                    "reason": f"vLLM is {((ollama_latency - vllm_latency) / ollama_latency * 100):.1f}% faster on average"
                }
            else:
                summary["recommendation"] = {
                    "engine": "ollama",
                    "reason": f"Ollama is {((vllm_latency - ollama_latency) / vllm_latency * 100):.1f}% faster on average"
                }

        return summary

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm/metrics/realtime")
async def get_realtime_metrics():
    """
    실시간 LLM 메트릭 조회

    최근 요청들의 메트릭 데이터 (캐시 히트율, 평균 응답 시간 등)

    Returns:
        실시간 메트릭
    """
    try:
        cache_stats = llm_cache.get_stats()

        # LLM 비교 통계에서 전체 요청 수 가져오기
        llm_stats = llm_comparison.get_stats(time_window_minutes=60)

        total_requests = 0
        successful_requests = 0
        failed_requests = 0

        for engine in ["vllm", "ollama"]:
            if engine in llm_stats and llm_stats[engine].get("total_requests", 0) > 0:
                total_requests += llm_stats[engine]["total_requests"]
                successful_requests += llm_stats[engine]["successful_requests"]
                failed_requests += llm_stats[engine]["failed_requests"]

        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "cache": {
                "size": cache_stats["size"],
                "max_size": cache_stats["max_size"],
                "hit_rate": cache_stats["hit_rate"],
                "total_hits": cache_stats["hits"],
                "total_misses": cache_stats["misses"],
                "total_requests": cache_stats["hits"] + cache_stats["misses"]
            },
            "llm_requests": {
                "total": total_requests,
                "successful": successful_requests,
                "failed": failed_requests,
                "success_rate": (
                    successful_requests / total_requests
                    if total_requests > 0 else 0
                )
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/llm/export-report")
async def export_comparison_report():
    """
    비교 리포트를 JSON 파일로 내보내기

    Returns:
        파일 경로
    """
    try:
        file_path = llm_comparison.export_comparison_report()
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "report_file": file_path,
            "message": "Comparison report exported successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_dashboard_data(
    time_window: Optional[int] = Query(60, description="Time window in minutes")
):
    """
    대시보드용 통합 데이터

    모니터링 대시보드에 필요한 모든 데이터를 한 번에 반환

    Returns:
        대시보드 데이터
    """
    try:
        # LLM 비교 통계
        llm_stats = llm_comparison.get_stats(time_window_minutes=time_window)

        # 실시간 메트릭
        cache_stats = llm_cache.get_stats()

        # 전체 요청 수 계산
        total_requests = 0
        successful_requests = 0
        failed_requests = 0

        for engine in ["vllm", "ollama"]:
            if engine in llm_stats and llm_stats[engine].get("total_requests", 0) > 0:
                total_requests += llm_stats[engine]["total_requests"]
                successful_requests += llm_stats[engine]["successful_requests"]
                failed_requests += llm_stats[engine]["failed_requests"]

        dashboard_data = {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "time_window_minutes": time_window,

            # LLM 엔진 통계
            "llm_engines": {},

            # 캐시 통계
            "cache": {
                "hit_rate": cache_stats["hit_rate"],
                "total_hits": cache_stats["hits"],
                "total_misses": cache_stats["misses"],
                "size": cache_stats["size"],
                "max_size": cache_stats["max_size"]
            },

            # 전체 요청 통계
            "requests": {
                "total": total_requests,
                "successful": successful_requests,
                "failed": failed_requests,
                "success_rate": (
                    successful_requests / total_requests
                    if total_requests > 0 else 0
                )
            }
        }

        # 각 엔진 통계 추가
        for engine in ["vllm", "ollama"]:
            if engine in llm_stats and llm_stats[engine].get("total_requests", 0) > 0:
                dashboard_data["llm_engines"][engine] = {
                    "total_requests": llm_stats[engine]["total_requests"],
                    "success_rate": llm_stats[engine]["success_rate"],
                    "latency": {
                        "mean": llm_stats[engine]["latency"]["mean_ms"],
                        "median": llm_stats[engine]["latency"]["median_ms"],
                        "p95": llm_stats[engine]["latency"]["p95_ms"],
                        "p99": llm_stats[engine]["latency"]["p99_ms"],
                        "min": llm_stats[engine]["latency"]["min_ms"],
                        "max": llm_stats[engine]["latency"]["max_ms"]
                    },
                    "throughput": {
                        "tokens_per_second": llm_stats[engine]["throughput"]["avg_tokens_per_second"],
                        "requests_per_minute": llm_stats[engine]["throughput"]["requests_per_minute"]
                    },
                    "tokens": {
                        "total_prompt_tokens": llm_stats[engine]["tokens"]["total_prompt_tokens"],
                        "total_completion_tokens": llm_stats[engine]["tokens"]["total_completion_tokens"],
                        "avg_tokens_per_request": llm_stats[engine]["tokens"]["avg_total_tokens"]
                    }
                }

        # 엔진 간 비교
        if "comparison" in llm_stats:
            dashboard_data["comparison"] = llm_stats["comparison"]

        return dashboard_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
