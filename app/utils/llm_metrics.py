"""
LLM Metrics Collection Module

Provides structured logging and Prometheus metrics for LLM service calls.
Enables performance comparison between Ollama and vLLM engines.
"""
import logging
import json
import time
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)


# Prometheus Metrics
llm_request_latency = Histogram(
    'llm_request_latency_seconds',
    'LLM request latency in seconds',
    ['engine', 'model', 'mode', 'has_schema'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0]
)

llm_request_total = Counter(
    'llm_request_total',
    'Total LLM requests',
    ['engine', 'model', 'mode', 'status']
)

llm_prompt_length_chars = Histogram(
    'llm_prompt_length_chars',
    'LLM prompt length in characters',
    ['engine', 'model'],
    buckets=[100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
)

llm_completion_length_chars = Histogram(
    'llm_completion_length_chars',
    'LLM completion length in characters',
    ['engine', 'model'],
    buckets=[50, 100, 500, 1000, 2000, 5000, 10000]
)

llm_token_usage = Histogram(
    'llm_token_usage_total',
    'Total token usage per request',
    ['engine', 'model', 'type'],
    buckets=[10, 50, 100, 500, 1000, 2000, 5000, 10000, 20000]
)

llm_active_requests = Gauge(
    'llm_active_requests',
    'Number of active LLM requests',
    ['engine', 'model']
)

llm_fallback_total = Counter(
    'llm_fallback_total',
    'Total number of engine fallbacks',
    ['from_engine', 'to_engine', 'reason']
)

llm_engine_health = Gauge(
    'llm_engine_health',
    'Engine health status (1=healthy, 0=unhealthy)',
    ['engine', 'model']
)


class LLMMetricsCollector:
    """
    Metrics collector for LLM service calls

    Provides both structured logging and Prometheus metrics
    for performance monitoring and engine comparison.
    """

    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)

    def log_request_start(
        self,
        engine: str,
        model: str,
        prompt_length: int,
        mode: str = "chat",
        has_schema: bool = False
    ) -> Dict[str, Any]:
        """
        Log request start and return context for completion logging

        Args:
            engine: "ollama" or "vllm"
            model: Model name
            prompt_length: Prompt length in characters
            mode: "chat", "vision", etc.
            has_schema: Whether response schema is specified

        Returns:
            Context dict for passing to log_request_end()
        """
        context = {
            'engine': engine,
            'model': model,
            'prompt_length': prompt_length,
            'mode': mode,
            'has_schema': has_schema,
            'start_time': time.time()
        }

        # Structured log
        log_data = {
            'event': 'llm_request_start',
            'engine': engine,
            'model': model,
            'prompt_length_chars': prompt_length,
            'mode': mode,
            'has_schema': has_schema
        }
        self.logger.info(f"LLM_REQUEST_START {json.dumps(log_data, ensure_ascii=False)}")

        # Update Prometheus metrics
        llm_prompt_length_chars.labels(
            engine=engine,
            model=model
        ).observe(prompt_length)

        llm_active_requests.labels(
            engine=engine,
            model=model
        ).inc()

        return context

    def log_request_end(
        self,
        context: Dict[str, Any],
        completion_length: int,
        token_usage: Optional[Dict[str, int]] = None,
        error: Optional[str] = None
    ):
        """
        Log request completion with metrics

        Args:
            context: Context from log_request_start()
            completion_length: Completion text length in characters
            token_usage: Token usage dict (prompt_tokens, completion_tokens, total_tokens)
            error: Error message if request failed
        """
        end_time = time.time()
        latency_ms = int((end_time - context['start_time']) * 1000)
        latency_s = end_time - context['start_time']

        engine = context['engine']
        model = context['model']
        mode = context['mode']
        has_schema = context['has_schema']
        status = 'error' if error else 'success'

        # Structured log
        log_data = {
            'event': 'llm_request_end',
            'engine': engine,
            'model': model,
            'mode': mode,
            'has_schema': has_schema,
            'status': status,
            'latency_ms': latency_ms,
            'prompt_length_chars': context['prompt_length'],
            'completion_length_chars': completion_length,
            'token_usage': token_usage or {},
        }

        if error:
            log_data['error'] = error
            self.logger.error(f"LLM_REQUEST_ERROR {json.dumps(log_data, ensure_ascii=False)}")
        else:
            self.logger.info(f"LLM_REQUEST_SUCCESS {json.dumps(log_data, ensure_ascii=False)}")

        # Update Prometheus metrics
        llm_request_latency.labels(
            engine=engine,
            model=model,
            mode=mode,
            has_schema=str(has_schema)
        ).observe(latency_s)

        llm_request_total.labels(
            engine=engine,
            model=model,
            mode=mode,
            status=status
        ).inc()

        llm_completion_length_chars.labels(
            engine=engine,
            model=model
        ).observe(completion_length)

        if token_usage:
            for token_type in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
                if token_type in token_usage:
                    llm_token_usage.labels(
                        engine=engine,
                        model=model,
                        type=token_type
                    ).observe(token_usage[token_type])

        llm_active_requests.labels(
            engine=engine,
            model=model
        ).dec()

    def log_fallback(
        self,
        from_engine: str,
        to_engine: str,
        reason: str,
        original_error: Optional[str] = None
    ):
        """
        Log engine fallback event

        Args:
            from_engine: Original engine that failed
            to_engine: Fallback engine
            reason: Reason for fallback (e.g., "connection_error", "timeout")
            original_error: Original error message
        """
        log_data = {
            'event': 'llm_fallback',
            'from_engine': from_engine,
            'to_engine': to_engine,
            'reason': reason,
            'original_error': original_error
        }

        self.logger.warning(f"LLM_FALLBACK {json.dumps(log_data, ensure_ascii=False)}")

        # Update Prometheus metrics
        llm_fallback_total.labels(
            from_engine=from_engine,
            to_engine=to_engine,
            reason=reason
        ).inc()

    def update_engine_health(self, engine: str, model: str, is_healthy: bool):
        """
        Update engine health status

        Args:
            engine: Engine name
            model: Model name
            is_healthy: True if healthy, False otherwise
        """
        llm_engine_health.labels(
            engine=engine,
            model=model
        ).set(1 if is_healthy else 0)

    @asynccontextmanager
    async def track_request(
        self,
        engine: str,
        model: str,
        prompt: str,
        mode: str = "chat",
        has_schema: bool = False
    ):
        """
        Context manager for automatic request tracking

        Usage:
            async with metrics.track_request("ollama", "gpt-oss:20b", prompt) as ctx:
                result = await some_llm_call()
                ctx['completion_length'] = len(result)
                ctx['token_usage'] = result['token']
        """
        context = self.log_request_start(
            engine=engine,
            model=model,
            prompt_length=len(prompt),
            mode=mode,
            has_schema=has_schema
        )

        # Yield context for caller to update
        tracking_ctx = {
            'completion_length': 0,
            'token_usage': None,
            'error': None
        }

        try:
            yield tracking_ctx
        except Exception as e:
            tracking_ctx['error'] = str(e)
            raise
        finally:
            self.log_request_end(
                context=context,
                completion_length=tracking_ctx.get('completion_length', 0),
                token_usage=tracking_ctx.get('token_usage'),
                error=tracking_ctx.get('error')
            )


# Global metrics collector instance
llm_metrics = LLMMetricsCollector()
