"""
LLM Engine Comparison Metrics
vLLM vs Ollama 성능 및 품질 비교용 메트릭 수집 시스템
"""
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import defaultdict
import statistics


class LLMComparisonMetrics:
    """LLM 엔진 비교 메트릭 수집 및 분석"""

    def __init__(self, metrics_dir: str = "logs/llm_comparison"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # 실시간 메트릭 저장
        self.metrics: Dict[str, List[Dict]] = defaultdict(list)

    def record_request(
        self,
        engine: str,
        model: str,
        query: str,
        response: str,
        latency_ms: float,
        prompt_length: int,
        response_length: int,
        token_usage: Dict[str, int],
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        LLM 요청 메트릭 기록

        Args:
            engine: 'vllm' or 'ollama'
            model: 모델명
            query: 사용자 질문
            response: LLM 응답
            latency_ms: 응답 시간 (밀리초)
            prompt_length: 프롬프트 길이 (문자 수)
            response_length: 응답 길이 (문자 수)
            token_usage: 토큰 사용량 dict
            success: 성공 여부
            error: 에러 메시지 (실패 시)
            metadata: 추가 메타데이터
        """
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "engine": engine,
            "model": model,
            "query": query[:100],  # 처음 100자만 저장
            "response_preview": response[:200] if response else "",  # 처음 200자만 저장
            "latency_ms": latency_ms,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "token_usage": token_usage,
            "success": success,
            "error": error,
            "metadata": metadata or {}
        }

        # 메모리에 저장
        self.metrics[engine].append(metric)

        # 파일에 append
        today = datetime.utcnow().strftime("%Y-%m-%d")
        log_file = self.metrics_dir / f"comparison_{today}.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metric, ensure_ascii=False) + "\n")

    def get_stats(
        self,
        engine: Optional[str] = None,
        time_window_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        통계 생성

        Args:
            engine: 특정 엔진만 ('vllm' or 'ollama'), None이면 전체
            time_window_minutes: 최근 N분 데이터만, None이면 전체

        Returns:
            통계 딕셔너리
        """
        # 필터링할 엔진 결정
        engines_to_analyze = [engine] if engine else ["vllm", "ollama"]

        # 시간 필터링
        if time_window_minutes:
            cutoff_time = time.time() - (time_window_minutes * 60)
        else:
            cutoff_time = None

        stats = {}

        for eng in engines_to_analyze:
            metrics = self.metrics.get(eng, [])

            # 시간 필터
            if cutoff_time:
                metrics = [
                    m for m in metrics
                    if datetime.fromisoformat(m["timestamp"]).timestamp() >= cutoff_time
                ]

            if not metrics:
                stats[eng] = {
                    "total_requests": 0,
                    "message": "No data available"
                }
                continue

            # 성공/실패 분류
            successful = [m for m in metrics if m["success"]]
            failed = [m for m in metrics if not m["success"]]

            # 레이턴시 통계
            latencies = [m["latency_ms"] for m in successful]

            # 토큰 통계
            prompt_tokens = [m["token_usage"].get("prompt_tokens", 0) for m in successful]
            completion_tokens = [m["token_usage"].get("completion_tokens", 0) for m in successful]
            total_tokens = [m["token_usage"].get("total_tokens", 0) for m in successful]

            stats[eng] = {
                "total_requests": len(metrics),
                "successful_requests": len(successful),
                "failed_requests": len(failed),
                "success_rate": len(successful) / len(metrics) if metrics else 0,

                "latency": {
                    "min_ms": min(latencies) if latencies else 0,
                    "max_ms": max(latencies) if latencies else 0,
                    "mean_ms": statistics.mean(latencies) if latencies else 0,
                    "median_ms": statistics.median(latencies) if latencies else 0,
                    "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                    "p95_ms": self._percentile(latencies, 95) if latencies else 0,
                    "p99_ms": self._percentile(latencies, 99) if latencies else 0,
                },

                "tokens": {
                    "avg_prompt_tokens": statistics.mean(prompt_tokens) if prompt_tokens else 0,
                    "avg_completion_tokens": statistics.mean(completion_tokens) if completion_tokens else 0,
                    "avg_total_tokens": statistics.mean(total_tokens) if total_tokens else 0,
                    "total_prompt_tokens": sum(prompt_tokens),
                    "total_completion_tokens": sum(completion_tokens),
                    "total_tokens": sum(total_tokens),
                },

                "throughput": {
                    "avg_tokens_per_second": self._calculate_throughput(successful),
                    "requests_per_minute": self._calculate_rpm(metrics, time_window_minutes),
                },

                "errors": self._get_error_summary(failed)
            }

        # 엔진 간 비교 (둘 다 있을 때)
        if "vllm" in stats and "ollama" in stats and stats["vllm"]["total_requests"] > 0 and stats["ollama"]["total_requests"] > 0:
            vllm_mean = stats["vllm"]["latency"]["mean_ms"]
            ollama_mean = stats["ollama"]["latency"]["mean_ms"]

            stats["comparison"] = {
                "vllm_faster_by_percent": ((ollama_mean - vllm_mean) / ollama_mean * 100) if ollama_mean > 0 else 0,
                "latency_ratio_vllm_to_ollama": (vllm_mean / ollama_mean) if ollama_mean > 0 else 0,
                "vllm_throughput_advantage": (
                    stats["vllm"]["throughput"]["avg_tokens_per_second"] -
                    stats["ollama"]["throughput"]["avg_tokens_per_second"]
                )
            }

        return stats

    def compare_engines(
        self,
        query: str,
        responses: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        동일한 질문에 대한 두 엔진의 응답 비교

        Args:
            query: 사용자 질문
            responses: {
                "vllm": {"response": str, "latency_ms": float, ...},
                "ollama": {"response": str, "latency_ms": float, ...}
            }

        Returns:
            비교 결과
        """
        comparison = {
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "engines": {}
        }

        for engine, data in responses.items():
            comparison["engines"][engine] = {
                "response_length": len(data.get("response", "")),
                "latency_ms": data.get("latency_ms", 0),
                "token_usage": data.get("token_usage", {}),
                "success": data.get("success", False)
            }

        # 속도 비교
        if "vllm" in responses and "ollama" in responses:
            vllm_latency = responses["vllm"].get("latency_ms", float('inf'))
            ollama_latency = responses["ollama"].get("latency_ms", float('inf'))

            if vllm_latency < ollama_latency:
                faster_engine = "vllm"
                speedup = ((ollama_latency - vllm_latency) / ollama_latency * 100)
            else:
                faster_engine = "ollama"
                speedup = ((vllm_latency - ollama_latency) / vllm_latency * 100)

            comparison["speed_comparison"] = {
                "faster_engine": faster_engine,
                "speedup_percent": speedup,
                "vllm_latency_ms": vllm_latency,
                "ollama_latency_ms": ollama_latency
            }

        return comparison

    def _percentile(self, data: List[float], percentile: int) -> float:
        """계산 percentile"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def _calculate_throughput(self, metrics: List[Dict]) -> float:
        """평균 tokens/second 계산"""
        if not metrics:
            return 0

        total_tokens = sum(m["token_usage"].get("completion_tokens", 0) for m in metrics)
        total_time_s = sum(m["latency_ms"] / 1000 for m in metrics)

        return total_tokens / total_time_s if total_time_s > 0 else 0

    def _calculate_rpm(self, metrics: List[Dict], time_window_minutes: Optional[int]) -> float:
        """Requests per minute 계산"""
        if not metrics:
            return 0

        if time_window_minutes:
            return len(metrics) / time_window_minutes
        else:
            # 전체 시간 범위에서 RPM 계산
            if len(metrics) < 2:
                return 0

            first_time = datetime.fromisoformat(metrics[0]["timestamp"])
            last_time = datetime.fromisoformat(metrics[-1]["timestamp"])
            duration_minutes = (last_time - first_time).total_seconds() / 60

            return len(metrics) / duration_minutes if duration_minutes > 0 else 0

    def _get_error_summary(self, failed_metrics: List[Dict]) -> List[Dict]:
        """에러 요약"""
        error_counts = defaultdict(int)

        for m in failed_metrics:
            error_msg = m.get("error", "Unknown error")
            error_counts[error_msg] += 1

        return [
            {"error": error, "count": count}
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        ]

    def export_comparison_report(
        self,
        output_file: Optional[str] = None
    ) -> str:
        """
        비교 리포트를 JSON 파일로 내보내기

        Returns:
            출력 파일 경로
        """
        if not output_file:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_file = self.metrics_dir / f"comparison_report_{timestamp}.json"

        stats = self.get_stats()

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        return str(output_file)


# 싱글톤 인스턴스
llm_comparison = LLMComparisonMetrics()
