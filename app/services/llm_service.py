"""
LLM Service - Multi-Engine Support (Ollama + vLLM)
Replaces Google Gemini API calls with local LLM services

This service supports:
- Ollama: gpt-oss:20b via AsyncClient
- vLLM: OpenAI-compatible API endpoint

Engine can be switched via LLM_ENGINE config or per-call engine parameter.

Includes comprehensive metrics collection for performance comparison.
"""
import logging
import asyncio
from typing import List, Dict, Optional, Any
from ollama import AsyncClient
import json
import httpx

from app.config import settings
from app.utils.llm_metrics import llm_metrics

logger = logging.getLogger(__name__)


class LLMService:
    """
    LLM Service with Multi-Engine Support

    Supports both Ollama and vLLM engines.
    Engine selection via settings.LLM_ENGINE or per-call engine parameter.

    Replaces all Gemini API calls from the original Node.js implementation:
    - gemini.js -> generateText()
    - getAnswer.js -> get_answer()
    - getSummary.js -> get_summary()
    - getImageToText.js -> get_image_to_text()
    - getSearchType.js -> get_search_type()
    - getFormatToc.js -> get_format_toc()
    """

    def __init__(self):
        # Ollama client
        self.ollama_client = AsyncClient(host=settings.OLLAMA_BASE_URL)
        self.ollama_model = settings.OLLAMA_MODEL or "gpt-oss:20b"

        # vLLM client (OpenAI-compatible)
        self.vllm_base_url = settings.VLLM_BASE_URL
        self.vllm_model = settings.VLLM_MODEL_NAME
        self.vllm_api_key = settings.VLLM_API_KEY

        # Default engine from settings
        self.default_engine = settings.LLM_ENGINE.lower()

        # Fallback configuration
        self.enable_fallback = settings.ENABLE_LLM_FALLBACK
        self.fallback_engine = settings.FALLBACK_ENGINE.lower()

        # Legacy compatibility
        self.client = self.ollama_client
        self.model = self.ollama_model

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🚀 LLM Service initialized with default engine: {self.default_engine}")
        self.logger.info(f"   Ollama: {settings.OLLAMA_BASE_URL} ({self.ollama_model})")
        self.logger.info(f"   vLLM: {self.vllm_base_url} ({self.vllm_model})")
        self.logger.info(f"   Fallback: {'enabled' if self.enable_fallback else 'disabled'} (→ {self.fallback_engine})")

    def _clean_llm_response(self, text: str) -> str:
        """
        Clean LLM response by removing prompt artifacts and instructions

        Removes common issues where LLM echoes back prompt instructions
        """
        if not text:
            return text

        # Remove leading meta-descriptions (model explaining what it's doing)
        # Common pattern: "We need answer short, 200-400 chars. Provide..."
        import re

        # Pattern 1: English meta instructions at the start
        # Matches things like "We need answer short" or "Provide concept description"
        text = re.sub(
            r'^(We need .{0,100}?\.|Provide .{0,100}?\.|Answer should .{0,100}?\.)+\s*',
            '',
            text,
            flags=re.IGNORECASE | re.MULTILINE
        )

        # Pattern 2: Inline prompt leakage (English instructions before Korean content)
        # Matches: "We need answer short, 200-400 chars. Provide...자기효능감은"
        text = re.sub(
            r'^.*?([가-힣])',
            r'\1',
            text,
            count=1
        )

        # Remove lines that look like prompt instructions
        lines = text.split('\n')
        cleaned_lines = []

        skip_patterns = [
            '출력 형식:',
            '출력형식:',
            '다음 JSON 스키마',
            '응답 예시:',
            '중요: 반드시',
            'JSON만 출력',
            '스키마:',
            '출력 예시',
            'We need',
            'Provide concept',
            'Answer should',
        ]

        for line in lines:
            # Skip lines that match instruction patterns
            if any(pattern in line for pattern in skip_patterns):
                continue
            cleaned_lines.append(line)

        result = '\n'.join(cleaned_lines).strip()

        # Remove leading/trailing artifacts
        if result.startswith('출력:') or result.startswith('응답:'):
            result = result.split(':', 1)[1].strip()

        return result

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Enhanced JSON extraction with multiple strategies

        Tries multiple approaches to extract valid JSON from LLM response:
        1. Direct parsing
        2. Remove markdown code blocks
        3. Find outermost braces with nesting support
        4. Multiple regex patterns
        """
        import re

        if not text:
            self.logger.error("Empty text for JSON extraction")
            raise ValueError("Empty response from LLM")

        # Strategy 1: Direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Remove markdown code blocks
        cleaned = text
        if '```' in text:
            # Remove ```json or ``` markers
            cleaned = re.sub(r'```(?:json)?\s*', '', text)
            cleaned = cleaned.strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find outermost braces with proper nesting
        first_brace = text.find('{')
        if first_brace != -1:
            # Find matching closing brace
            depth = 0
            for i in range(first_brace, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        json_str = text[first_brace:i+1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            break

        # Strategy 4: Multiple regex patterns
        patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested braces
            r'\{.*?\}',  # Non-greedy
            r'\{.+\}',  # Greedy
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    continue

        # All strategies failed
        self.logger.error(f"JSON extraction failed after all strategies")
        self.logger.error(f"Response text (first 500 chars): {text[:500]}")
        raise ValueError(f"Could not extract valid JSON from response")

    async def generate_text(
        self,
        prompt: str,
        response_schema: Optional[Dict] = None,
        mode: str = "chat",
        image_data: Optional[Dict] = None,
        max_retries: int = 2,
        engine: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Core LLM generation function with multi-engine support and automatic fallback

        Args:
            prompt: The prompt to send to LLM
            response_schema: Optional JSON schema for structured output
            mode: "chat", "vision", or "chat2" (mode handling varies by engine)
            image_data: Optional image data for vision tasks
            max_retries: Maximum retry attempts for empty responses (default: 2)
            engine: Engine to use ("ollama" or "vllm"). If None, uses default from settings.

        Returns:
            Dict with 'result', 'token' (usage metadata), 'duration' (ms)

        Raises:
            ValueError: If engine is unknown or both primary and fallback engines fail
        """
        # Determine which engine to use
        selected_engine = (engine or self.default_engine).lower()

        # Validate engine
        if selected_engine not in ["ollama", "vllm"]:
            raise ValueError(f"Unknown engine: {selected_engine}. Use 'ollama' or 'vllm'")

        # Try primary engine
        try:
            return await self._execute_engine_call(
                engine=selected_engine,
                prompt=prompt,
                response_schema=response_schema,
                mode=mode,
                image_data=image_data,
                max_retries=max_retries
            )

        except Exception as primary_error:
            # Check if fallback is enabled and appropriate
            should_fallback = (
                self.enable_fallback and
                self.fallback_engine != selected_engine and
                self.fallback_engine in ["ollama", "vllm"]
            )

            if not should_fallback:
                # No fallback configured or same engine, re-raise original error
                raise

            # Log and attempt fallback
            error_type = type(primary_error).__name__
            error_msg = str(primary_error)

            self.logger.warning(
                f"⚠️ [{selected_engine.upper()}] Engine failed: {error_type}: {error_msg}"
            )
            self.logger.warning(
                f"🔄 Attempting fallback to {self.fallback_engine.upper()} engine..."
            )

            # Record fallback metrics
            llm_metrics.log_fallback(
                from_engine=selected_engine,
                to_engine=self.fallback_engine,
                reason=error_type,
                original_error=error_msg
            )

            try:
                # Attempt fallback
                result = await self._execute_engine_call(
                    engine=self.fallback_engine,
                    prompt=prompt,
                    response_schema=response_schema,
                    mode=mode,
                    image_data=image_data,
                    max_retries=max_retries
                )

                self.logger.info(
                    f"✅ Fallback successful: {self.fallback_engine.upper()} completed request"
                )

                # Mark fallback engine as healthy
                llm_metrics.update_engine_health(
                    engine=self.fallback_engine,
                    model=self.vllm_model if self.fallback_engine == "vllm" else self.ollama_model,
                    is_healthy=True
                )

                return result

            except Exception as fallback_error:
                # Both engines failed
                self.logger.error(
                    f"❌ Fallback to {self.fallback_engine.upper()} also failed: {fallback_error}"
                )

                # Mark both engines as unhealthy
                llm_metrics.update_engine_health(
                    engine=selected_engine,
                    model=self.vllm_model if selected_engine == "vllm" else self.ollama_model,
                    is_healthy=False
                )
                llm_metrics.update_engine_health(
                    engine=self.fallback_engine,
                    model=self.vllm_model if self.fallback_engine == "vllm" else self.ollama_model,
                    is_healthy=False
                )

                # Re-raise the original error (more relevant)
                raise primary_error from fallback_error

    async def _execute_engine_call(
        self,
        engine: str,
        prompt: str,
        response_schema: Optional[Dict] = None,
        mode: str = "chat",
        image_data: Optional[Dict] = None,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Execute LLM call on specified engine

        Internal method used by generate_text() to route to specific engine.

        Args:
            engine: "ollama" or "vllm"
            prompt: The prompt to send
            response_schema: Optional JSON schema
            mode: Operation mode
            image_data: Optional image data
            max_retries: Retry attempts

        Returns:
            Dict with 'result', 'token', 'duration'
        """
        if engine == "vllm":
            return await self._generate_text_vllm(
                prompt=prompt,
                response_schema=response_schema,
                mode=mode,
                max_retries=max_retries
            )
        elif engine == "ollama":
            return await self._generate_text_ollama(
                prompt=prompt,
                response_schema=response_schema,
                mode=mode,
                image_data=image_data,
                max_retries=max_retries
            )
        else:
            raise ValueError(f"Unknown engine: {engine}")

    async def _generate_text_ollama(
        self,
        prompt: str,
        response_schema: Optional[Dict] = None,
        mode: str = "chat",
        image_data: Optional[Dict] = None,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Ollama-specific text generation

        Args:
            prompt: The prompt to send to LLM
            response_schema: Optional JSON schema for structured output
            mode: "chat", "vision", or "chat2"
            image_data: Optional image data for vision tasks
            max_retries: Maximum retry attempts

        Returns:
            Dict with 'result', 'token', 'duration'
        """
        import time

        # For vision mode with image, use vision-capable model
        model_name = self.ollama_model
        if mode == "vision" and image_data:
            model_name = "llava:latest"

        # Start metrics collection
        metrics_ctx = llm_metrics.log_request_start(
            engine="ollama",
            model=model_name,
            prompt_length=len(prompt),
            mode=mode,
            has_schema=response_schema is not None
        )

        # Retry loop for empty responses
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()

                # Build request
                if response_schema:
                    # Request JSON output with detailed instructions
                    enhanced_prompt = f"""{prompt}

출력 형식:
다음 JSON 스키마를 정확히 따라 응답하세요. 설명이나 다른 텍스트 없이 JSON만 출력하세요.

스키마:
{json.dumps(response_schema, ensure_ascii=False, indent=2)}

응답 예시:
{{
  "field1": "value1",
  "field2": "value2"
}}

중요: 반드시 유효한 JSON만 출력하고, 앞뒤에 어떤 텍스트도 포함하지 마세요."""

                    if attempt > 0:
                        self.logger.warning(f"🔄 [Ollama] Retry attempt {attempt}/{max_retries} for JSON response")

                    response = await self.ollama_client.generate(
                        model=model_name,
                        prompt=enhanced_prompt,
                        keep_alive=-1
                    )

                    response_text = response['response'].strip()
                    self.logger.debug(f"📝 [Ollama] Response length: {len(response_text)} chars")

                    result = self._extract_json_from_text(response_text)

                else:
                    # Plain text response
                    if attempt > 0:
                        self.logger.warning(f"🔄 [Ollama] Retry attempt {attempt}/{max_retries} for text response")

                    response = await self.ollama_client.generate(
                        model=model_name,
                        prompt=prompt,
                        keep_alive=-1
                    )

                    response_text = response['response'].strip()
                    self.logger.debug(f"📝 [Ollama] Response length: {len(response_text)} chars")

                    if not response_text:
                        raise ValueError("Empty response from LLM")

                    result = self._clean_llm_response(response_text)

                duration_ms = int((time.time() - start_time) * 1000)

                if attempt > 0:
                    self.logger.info(f"✅ [Ollama] Retry successful on attempt {attempt}")
                break

            except ValueError as e:
                last_error = e
                if attempt < max_retries:
                    self.logger.warning(f"⚠️ [Ollama] Attempt {attempt + 1} failed: {str(e)}, retrying...")
                    self.logger.warning(f"📊 Prompt length: {len(prompt)} chars, Model: {model_name}")
                    await asyncio.sleep(0.5)
                    continue
                else:
                    # Log failed request metrics
                    llm_metrics.log_request_end(
                        context=metrics_ctx,
                        completion_length=0,
                        token_usage=None,
                        error=str(e)
                    )
                    self.logger.error(f"❌ [Ollama] All {max_retries + 1} attempts failed")
                    self.logger.error(f"📊 Final attempt - Prompt length: {len(prompt)} chars")
                    raise

        # Prepare token info
        token_info = {
            'prompt_tokens': response.get('prompt_eval_count', 0),
            'completion_tokens': response.get('eval_count', 0),
            'total_tokens': response.get('prompt_eval_count', 0) + response.get('eval_count', 0)
        }

        # Calculate completion length
        completion_length = len(str(result)) if isinstance(result, str) else len(json.dumps(result, ensure_ascii=False))

        # Log successful request metrics
        llm_metrics.log_request_end(
            context=metrics_ctx,
            completion_length=completion_length,
            token_usage=token_info,
            error=None
        )

        return {
            'result': result,
            'token': token_info,
            'duration': duration_ms
        }

    async def _generate_text_vllm(
        self,
        prompt: str,
        response_schema: Optional[Dict] = None,
        mode: str = "chat",
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        vLLM-specific text generation via OpenAI-compatible API

        Args:
            prompt: The prompt to send to LLM
            response_schema: Optional JSON schema (currently text mode only)
            mode: "chat" mode (vision not yet supported)
            max_retries: Maximum retry attempts

        Returns:
            Dict with 'result', 'token', 'duration'
        """
        import time

        if mode == "vision":
            self.logger.warning("⚠️ [vLLM] Vision mode not yet supported, falling back to text mode")

        # Start metrics collection
        metrics_ctx = llm_metrics.log_request_start(
            engine="vllm",
            model=self.vllm_model,
            prompt_length=len(prompt),
            mode=mode,
            has_schema=response_schema is not None
        )

        # Build OpenAI-compatible request
        url = f"{self.vllm_base_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }

        # Add API key if not "EMPTY"
        if self.vllm_api_key and self.vllm_api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.vllm_api_key}"

        # Retry loop
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()

                # Build messages
                user_content = prompt
                if response_schema:
                    # Add JSON schema instructions to prompt
                    user_content = f"""{prompt}

출력 형식:
다음 JSON 스키마를 정확히 따라 응답하세요. 설명이나 다른 텍스트 없이 JSON만 출력하세요.

스키마:
{json.dumps(response_schema, ensure_ascii=False, indent=2)}

중요: 반드시 유효한 JSON만 출력하고, 앞뒤에 어떤 텍스트도 포함하지 마세요."""

                    if attempt > 0:
                        self.logger.warning(f"🔄 [vLLM] Retry attempt {attempt}/{max_retries} for JSON response")

                # CRITICAL: Dynamic max_tokens based on prompt length (prevent overflow)
                # Korean text: ~1 char = 0.7 tokens (conservative estimate)
                estimated_prompt_tokens = int(len(user_content) * 0.7)
                model_context_limit = 8192
                available_tokens = model_context_limit - estimated_prompt_tokens

                if available_tokens < 200:
                    max_tokens = 100
                    self.logger.warning(f"⚠️ [vLLM] Very long prompt ({estimated_prompt_tokens} tokens), limiting max_tokens to {max_tokens}")
                else:
                    # 유동적 max_tokens: JSON은 중간, 텍스트는 길게 (문제/설명 대응)
                    desired_max = 600 if response_schema else 1200
                    max_tokens = min(available_tokens - 200, desired_max)

                if max_tokens < 50:
                    self.logger.error(f"❌ [vLLM] Prompt too long! Estimated: {estimated_prompt_tokens} tokens")
                    max_tokens = 50

                self.logger.info(f"🎯 [vLLM] Prompt: {len(user_content)} chars (~{estimated_prompt_tokens} tokens), max_tokens: {max_tokens} ({'JSON' if response_schema else 'text'})")

                payload = {
                    "model": self.vllm_model,
                    "messages": [
                        {"role": "user", "content": user_content}
                    ],
                    "temperature": 0.5,
                    "max_tokens": max_tokens,
                    "top_p": 0.85,
                }

                # Make HTTP request
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()

                # Extract response
                if "choices" not in data or len(data["choices"]) == 0:
                    raise ValueError("No choices in vLLM response")

                message = data["choices"][0]["message"]

                # 디버깅: 메시지 전체 구조 확인
                if attempt == 0:
                    self.logger.info(f"🔍 [vLLM] Message keys: {list(message.keys())}")

                response_text = message.get("content") or message.get("reasoning_content") or ""

                if not response_text:
                    self.logger.error(f"❌ [vLLM] Empty response, full message: {message}")
                    raise ValueError("Empty response from vLLM")

                response_text = response_text.strip()
                self.logger.debug(f"📝 [vLLM] Response length: {len(response_text)} chars")

                # 디버깅: JSON 응답인 경우 첫 100자 로그
                if response_schema and attempt == 0:
                    self.logger.info(f"📄 [vLLM] JSON response preview: {response_text[:200]}...")

                # Parse response
                if response_schema:
                    result = self._extract_json_from_text(response_text)
                else:
                    if not response_text:
                        raise ValueError("Empty response from vLLM")
                    result = self._clean_llm_response(response_text)

                duration_ms = int((time.time() - start_time) * 1000)

                # Extract token usage
                usage = data.get("usage", {})
                token_info = {
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0)
                }

                if attempt > 0:
                    self.logger.info(f"✅ [vLLM] Retry successful on attempt {attempt}")
                break

            except (httpx.HTTPError, ValueError) as e:
                last_error = e
                if attempt < max_retries:
                    self.logger.warning(f"⚠️ [vLLM] Attempt {attempt + 1} failed: {str(e)}, retrying...")
                    self.logger.warning(f"📊 Prompt length: {len(prompt)} chars, Model: {self.vllm_model}")
                    await asyncio.sleep(0.5)
                    continue
                else:
                    # Log failed request metrics
                    llm_metrics.log_request_end(
                        context=metrics_ctx,
                        completion_length=0,
                        token_usage=None,
                        error=str(e)
                    )
                    self.logger.error(f"❌ [vLLM] All {max_retries + 1} attempts failed")
                    self.logger.error(f"📊 Final attempt - Prompt length: {len(prompt)} chars")
                    raise

        # Calculate completion length
        completion_length = len(str(result)) if isinstance(result, str) else len(json.dumps(result, ensure_ascii=False))

        # Log successful request metrics
        llm_metrics.log_request_end(
            context=metrics_ctx,
            completion_length=completion_length,
            token_usage=token_info,
            error=None
        )

        return {
            'result': result,
            'token': token_info,
            'duration': duration_ms
        }

    async def get_answer_stream(
        self,
        title: str,
        texts: str,
        history_user_query: str,
        question: str,
        toc: str,
        engine: Optional[str] = None
    ):
        """
        Stream answer generation for real-time user feedback

        Supports both vLLM and Ollama engines with automatic fallback.

        Args:
            title: Book title
            texts: Related book content
            history_user_query: Previous conversation history
            question: User's question
            toc: Table of contents
            engine: Engine to use ('vllm' or 'ollama'). If None, uses default_engine.

        Yields:
            Chunks of answer text as they're generated
        """
        prompt = self._build_answer_prompt(title, texts, history_user_query, question, toc)
        selected_engine = (engine or self.default_engine).lower()

        # Try primary engine first
        try:
            if selected_engine == "vllm":
                self.logger.debug(f"🎯 [vLLM] Starting streaming answer for: {question[:50]}")
                async for chunk in self._stream_answer_vllm(prompt):
                    yield chunk
                return
            elif selected_engine == "ollama":
                self.logger.debug(f"🎯 [Ollama] Starting streaming answer for: {question[:50]}")
                async for chunk in self._stream_answer_ollama(prompt, question):
                    yield chunk
                return
            else:
                raise ValueError(f"Unknown engine: {selected_engine}")

        except Exception as e:
            self.logger.warning(f"⚠️ [{selected_engine.upper()}] Streaming failed: {e}")

            # Try fallback if enabled
            if self.enable_fallback and self.fallback_engine != selected_engine:
                self.logger.warning(f"🔄 Attempting fallback to {self.fallback_engine.upper()} engine...")
                llm_metrics.log_fallback(
                    from_engine=selected_engine,
                    to_engine=self.fallback_engine,
                    reason=type(e).__name__,
                    error_message=str(e)
                )

                try:
                    if self.fallback_engine == "ollama":
                        async for chunk in self._stream_answer_ollama(prompt, question):
                            yield chunk
                        return
                    elif self.fallback_engine == "vllm":
                        async for chunk in self._stream_answer_vllm(prompt):
                            yield chunk
                        return
                except Exception as fallback_error:
                    self.logger.error(f"❌ Fallback to {self.fallback_engine.upper()} also failed: {fallback_error}")
                    raise fallback_error
            else:
                raise

    async def _stream_answer_vllm(self, prompt: str):
        """
        Stream answer using vLLM engine

        Args:
            prompt: The formatted prompt

        Yields:
            Text chunks as they're generated
        """
        url = f"{self.vllm_base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        if self.vllm_api_key and self.vllm_api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.vllm_api_key}"

        # CRITICAL: Conservative token estimation for Korean text
        # Korean text: ~1 char = 1 token (Unicode, CJK characters)
        # English text: ~1 char = 0.25 tokens (4 chars per token)
        # Mixed Korean/English with formatting: Use 1 char = 0.7 tokens to be VERY safe
        estimated_prompt_tokens = int(len(prompt) * 0.7)

        # Model context: 8192 tokens for gpt-oss:20b
        model_context_limit = 8192

        # Calculate available space for completion
        available_tokens = model_context_limit - estimated_prompt_tokens

        # Hard cap max_tokens to prevent overflow
        # vLLM will error if prompt + max_tokens > context_limit
        if available_tokens < 200:
            # Prompt is very long, use minimum viable max_tokens
            max_tokens = 100
            self.logger.warning(f"⚠️ [vLLM SSE] Very long prompt ({estimated_prompt_tokens} tokens), limiting max_tokens to {max_tokens}")
        else:
            # 유동적 max_tokens: 간단한 답변부터 긴 설명/문제까지 대응
            max_tokens = min(available_tokens - 200, 1200)

        # Absolute minimum check
        if max_tokens < 50:
            self.logger.error(f"❌ [vLLM SSE] Prompt too long! Estimated: {estimated_prompt_tokens} tokens, limit: {model_context_limit}")
            max_tokens = 50  # Try anyway with minimum

        self.logger.info(f"🎯 [vLLM SSE] Prompt: {len(prompt)} chars (~{estimated_prompt_tokens} tokens), max_tokens: {max_tokens}, available: {available_tokens}")

        payload = {
            "model": self.vllm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "max_tokens": max_tokens,
            "top_p": 0.85,
            "stream": True
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                # Check for errors before streaming
                if response.status_code != 200:
                    error_body = await response.aread()
                    self.logger.error(f"❌ [vLLM SSE] HTTP {response.status_code}: {error_body.decode('utf-8')}")
                    self.logger.error(f"❌ [vLLM SSE] Request payload: {json.dumps(payload, ensure_ascii=False)[:500]}")
                    response.raise_for_status()

                chunk_count = 0
                line_count = 0
                async for line in response.aiter_lines():
                    line_count += 1

                    if not line.strip() or line.startswith(":"):
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix

                        if data_str == "[DONE]":
                            self.logger.debug("🏁 [vLLM SSE] Received [DONE] signal")
                            break

                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})

                                # 디버깅: 처음 5개 delta 출력
                                if line_count <= 5:
                                    self.logger.info(f"🔍 [vLLM SSE] Delta #{line_count}: {delta}")

                                # Reasoning 모델: reasoning_content (사고) + content (답변)
                                # ✅ reasoning_content는 SKIP (영어 사고 과정)
                                # ✅ content만 출력 (실제 한국어 답변)
                                content = delta.get("content", "")

                                if content:
                                    chunk_count += 1
                                    if chunk_count <= 2:
                                        self.logger.info(f"✅ [vLLM SSE] Content chunk #{chunk_count}: {content[:50]}")
                                    yield content
                                # reasoning_content는 로그만 (출력 안 함)
                                elif delta.get("reasoning_content"):
                                    if line_count <= 3:
                                        self.logger.debug(f"🧠 [vLLM SSE] Skipping reasoning #{line_count}")
                        except json.JSONDecodeError as e:
                            if line_count <= 5:
                                self.logger.warning(f"⚠️ [vLLM SSE] JSON decode error on line {line_count}: {e}")
                            continue

                self.logger.debug(f"📤 [vLLM] Streamed {chunk_count} chunks (total lines: {line_count})")

    async def _stream_answer_ollama(self, prompt: str, question: str):
        """
        Stream answer using Ollama engine

        Args:
            prompt: The formatted prompt
            question: User's question (for logging)

        Yields:
            Text chunks as they're generated
        """
        messages = [{'role': 'user', 'content': prompt}]

        chunk_count = 0
        total_chunks = 0
        async for chunk in await self.ollama_client.chat(
            model=self.ollama_model,
            messages=messages,
            stream=True
        ):
            total_chunks += 1
            # Ollama returns ChatResponse object (not dict!)
            # For reasoning models, answer appears in 'content' after 'thinking' phase
            if hasattr(chunk, 'message') and chunk.message:
                # Stream only the final answer content, skip thinking tokens
                if chunk.message.content:
                    content = chunk.message.content
                    chunk_count += 1
                    yield content
                elif total_chunks <= 3:  # Log first 3 empty chunks
                    thinking_preview = chunk.message.thinking[:30] if chunk.message.thinking else ''
                    self.logger.debug(f"⚠️ Empty content chunk #{total_chunks}: thinking={thinking_preview}")

        self.logger.debug(f"📤 [Ollama] Streamed {chunk_count}/{total_chunks} chunks for question: {question[:50]}")

    def _build_answer_prompt(
        self,
        title: str,
        texts: str,
        history_user_query: str,
        question: str,
        toc: str
    ) -> str:
        """
        Build answer prompt (extracted for reuse in streaming and non-streaming)

        Returns:
            Formatted prompt string
        """
        return f"""**교재 AI 어시스턴트**

**규칙:**
1. 🇰🇷 한국어만 (영어 절대 금지)
2. 책 제목 금지 → "이 책", "교재" 사용
3. **답변 길이:** 질문에 맞게 유동적

**🔥 문제 번호 참조 (CRITICAL):**
사용자가 "문제 1번", "문제 2번", "1번 문제" 등을 언급하면:
→ 아래 **대화 기록**에서 해당 번호 찾기 (예: "**문제 1번**", "**문제 2번**")
→ 해당 문제 내용 복사
→ 그 문제에 대한 정답/해설 제공
→ 다른 문제와 혼동하지 말 것!

**인사:** "안녕" → "안녕하세요! 무엇을 도와드릴까요?"

---
**대화 기록:**
{history_user_query[:1500] if history_user_query else "없음"}

**책 내용:**
{texts[:3000] if texts else "없음"}

**질문:** {question}

**답변:**
"""

    async def get_answer(
        self,
        title: str,
        texts: str,
        history_user_query: str,
        question: str,
        toc: str
    ) -> Dict[str, Any]:
        """
        Answer questions about book content (non-streaming version)

        Args:
            title: Book title
            texts: Related book content
            history_user_query: Previous conversation history
            question: User's question
            toc: Table of contents

        Returns:
            Response dict with answer text
        """
        prompt = self._build_answer_prompt(title, texts, history_user_query, question, toc)

        try:
            response = await self.generate_text(prompt, mode="chat")
            return response
        except Exception as e:
            self.logger.error(f"Error in get_answer: {e}")
            raise

    async def get_toc_answer(
        self,
        title: str,
        toc: str,
        question: str,
        show_full_details: bool = False,
        history_user_query: str = "[]"
    ) -> Dict[str, Any]:
        """
        Answer TOC-specific questions with simplified prompt (faster processing)

        This method is optimized for table of contents queries like:
        - "목차 보여줘" (show table of contents)
        - "2장 제목이 뭐야?" (what's the title of chapter 2?)
        - "몇 장까지 있어?" (how many chapters are there?)

        Args:
            title: Book title
            toc: Table of contents (simple or full version)
            question: User's question about TOC
            show_full_details: If False, show only chapter-level; if True, show all details
            history_user_query: Previous conversation history (optional)

        Returns:
            Response dict with answer text
        """
        # Determine output format based on show_full_details flag
        if show_full_details:
            detail_instruction = "목차의 모든 장과 절을 포함하여 상세하게 보여주세요."
        else:
            detail_instruction = "목차는 **장(Chapter) 수준만** 간략하게 보여주세요. 세부 절(Section)은 생략하고 대제목만 포함하세요."

        prompt = f"""당신은 목차 정보를 제공하는 AI 어시스턴트입니다.

## 중요 규칙
**책 제목을 응답에 포함하지 마세요.**
- "이 책", "본서", "해당 도서" 같은 표현을 사용하세요
- 예시: ❌ "성격심리학의 목차" → ✅ "목차" 또는 "이 책의 목차"

## 목차 정보 (전체)
{toc}

## 이전 대화
{history_user_query}

**중요**: 현재 질문이 이전 대화의 후속 질문인 경우, 이전 대화 내용을 참고하여 답변하세요.
특히 "객관식으로", "서술형으로" 같은 형식 변경 요청은 이전 내용을 해당 형식으로 다시 생성하는 것입니다.

## 사용자 질문
{question}

## 답변 가이드
1. **책 제목을 직접 쓰지 마세요** (할루시네이션 방지)
2. 목차 정보를 기반으로 정확하게 답변하세요
3. 목차 전체를 보여달라는 요청이면:
   - {detail_instruction}
   - 보기 좋게 마크다운 형식으로 정리해서 제공하세요
   - 페이지 범위도 함께 표시하세요
4. 특정 정보를 묻는다면 (예: "2장 제목", "몇 장까지"), 해당 정보만 간결하게 답변하세요
5. 목차에 없는 내용은 "목차에서 해당 정보를 찾을 수 없습니다"라고 답변하세요
6. 친절하고 자연스러운 톤을 유지하세요

답변 (최대 1500자):"""

        try:
            response = await self.generate_text(prompt, mode="chat")
            return response
        except Exception as e:
            self.logger.error(f"Error in get_toc_answer: {e}")
            raise

    async def get_summary(
        self,
        origin_text: str,
        book_title: str,
        section_title: str
    ) -> Dict[str, Any]:
        """
        Generate summary and extract keywords (replaces getSummary.js)

        Args:
            origin_text: Original text to summarize
            book_title: Book title
            section_title: Section title/TOC entry

        Returns:
            Dict with 'coreSummary' and 'sectionDetails'
        """
        prompt = f"""필요한 하위 목차 작성 및 요약하기.
주어진 작업 순서대로 작업을 진행해야 합니다.

# 0. 현재 처리할 부분에 대해서 확인하세요.
[책 제목]: {book_title}
[현재 목차]: {section_title}

# 1. 현재 목차({section_title})는 어떤 범주인가? (부,장,절,항)


# 2. coreSummary
이 책에서만 얻을 수 있는 핵심 개념을 파악할 수 있는 키워드들을 , 로 구분해서 작성한다.


# 3. 현재 목차의 범주가 절이 아닐때
다음처럼 목차의 범주가 부,장,기타 일때
예) 1부 이책을 시작하면서
예) 들어가면서
예) 1장 개론학
sectionDetails = [
      {{
        "title": "없음",
        "pageStart": 0,
        "pageEnd": 0,
        "summary": "없음"
      }},
]
**절대** 하위 목차들을 작성하지 않는다. 단지 1개만 위의 값대로 작성한다.



# 4. 현재 목차가 절인 경우에
- 1. [처리할 원문]은 현재 목차에 대한 내용입니다.
- 2. [처리할 원문]에서 현재 목차의 하위 목차들을 찾아서 정리해야 합니다.

"sectionDetails": [
      {{
        "title": "세부절 제목",
        "pageStart": 시작페이지(정수),
        "pageEnd": 끝페이지(정수),
        "summary": "100-200 단어 핵심문장으로 정리한다."
      }},
      ...
    ]

# 5. 참고문헌은 요약하지 않는다. 키워드 작성을 하지 않는다.

# 6. 목차는 요약하지 않는다. 키워드 작성을 하지 않는다.

# 7. 용어 설명은 요약하지 않는다. 키워드 작성을 하지 않는다.

---



### [처리할 원문]
{origin_text}
"""

        response_schema = {
            "type": "object",
            "properties": {
                "coreSummary": {
                    "type": "string",
                    "description": "이 책에서만 얻을 수 있는 핵심 개념을 파악할 수 있는 키워드들을 , 로 구분해서 작성한다."
                },
                "sectionDetails": {
                    "type": "array",
                    "description": "목차 정보 목록",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "pageStart": {"type": "integer"},
                            "pageEnd": {"type": "integer"},
                            "summary": {"type": "string"}
                        },
                        "required": ["title", "pageStart", "pageEnd", "summary"]
                    }
                }
            },
            "required": ["coreSummary", "sectionDetails"]
        }

        self.logger.info(f"📄 요약 생성 중... 원문 길이: {len(origin_text)}자")

        try:
            response = await self.generate_text(prompt, response_schema=response_schema, mode="chat")
            return response['result']
        except Exception as e:
            self.logger.error(f"Error in get_summary: {e}")
            raise

    async def get_image_to_text(
        self,
        image_data: Dict[str, str],
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Extract text from PDF page image (replaces getImageToText.js)

        NOTE: This uses vision model for OCR. If vision model not available,
        this will need to use traditional OCR (tesseract, etc.)

        Args:
            image_data: Dict with 'data' (base64) and 'mimeType'
            max_retries: Number of retry attempts

        Returns:
            Dict with 'result' containing 'pageNumber' and 'extractedText'
        """
        prompt = """너는 문서 구조를 분석하고 텍스트를 추출하는 최고 수준의 전문가야.
지금부터 이미지를 분석하여 텍스트를 추출해 줘. 가장 중요한 원칙은 **'사람이 읽는 자연스러운 순서'**를 그대로 따르는 거야.
**이미지를 별도 분리하지 않고**, 본문에서 이미지가 나타나는 위치에
이미지가 보이는 위치에는 ![…](…) 대신,
[이미지: 이미지 설명] 식으로 Alt-text만 텍스트 안에 삽입해 줘. 형식으로 넣어야 해.

이를 위해 다음 지침을 반드시 지켜줘:

### 1. 시각적 분석 우선: 텍스트를 읽기 전에, 먼저 문서 전체의 레이아웃(단, 문단, 그림 등의 배치)을 시각적으로 파악해.


### 2. 인간의 독서 흐름 모방:
- 만약 1단 구조라면, 위에서 아래로 순서대로 추출해.
- 만약 신문이나 논문처럼 여러 단 구조라면, 반드시 왼쪽 첫 번째 단의 내용을 위에서 아래로 모두 추출한 뒤, 다음 단으로 넘어가서 작업을 반복해.
- 제목이나 초록처럼 페이지 전체에 걸쳐 있는 텍스트는 가장 먼저 처리해.

### 3. **본문 추출**
- 본문 텍스트와 표의 데이터, 이미지 설명을 모두 포함. 이미지에 정보가 있다면 요약하지 말고 전체를 작성한다.
- 이미지가 나오면 [이미지: 이미지를 텍스트로 설명] 형태로 **본문 안에** 삽입.
- 이미지에 텍스트가 있다면 해당 텍스트를 모두 작성해야 한다. 관계표시등도 모두 마찬가지이다.


### 4. 내용의 정확성: 텍스트가 뒤섞이거나 누락되지 않도록, 보이는 그대로 정확한 순서와 내용으로 변환해 줘.

### 5. 출력내용
- 추출된 텍스트 : 일반 텍스트 및 표와 이미지등 모든 내용을 텍스트화 합니다. markdown 문법을 사용합니다.
- 추출한 pdf 의 페이지 번호

### 6. 텍스트가 없다면 "내용없음" 으로 출력할 것!

### 7. 이미지만 있는 페이지면 이미지에 대한 설명을 출력하면 됨

### 8. 저작권에 관련된 정보를 제외하고 출력해야 한다.

출력은 json 입니다.
"""

        response_schema = {
            "type": "object",
            "description": "PDF의 한 페이지 이미지에서 추출한 정보",
            "properties": {
                "pageNumber": {
                    "type": "integer",
                    "description": "PDF 내 해당 페이지의 실제 번호"
                },
                "extractedText": {
                    "type": "string",
                    "description": "해당 이미지(PDF 페이지)에서 추출한 텍스트 markdown 형식으로 출력"
                }
            },
            "required": ["pageNumber", "extractedText"]
        }

        retries = 0
        while retries <= max_retries:
            try:
                response = await self.generate_text(
                    prompt,
                    response_schema=response_schema,
                    mode="vision",
                    image_data=image_data
                )
                self.logger.info("텍스트변환됨")
                return response

            except Exception as err:
                retries += 1
                if retries > max_retries:
                    self.logger.error(f"❌ Vision API {max_retries}회 재시도 실패: {err}")
                    return {
                        'result': {
                            'extractedText': '[Vision API 3회 오류]'
                        }
                    }

                self.logger.warning(f"⚠️ 이미지 텍스트 추출 실패, 재시도 {retries}/{max_retries}...")
                import asyncio
                await asyncio.sleep(3)

    async def get_search_type(
        self,
        title: str,
        toc: str,
        history_user_query: str,
        query: str,
        previous_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Determine search type and extract keywords (replaces getSearchType.js)

        Args:
            title: Book title
            toc: Table of contents (JSON string)
            history_user_query: Conversation history
            query: User query
            previous_context: Previous search context

        Returns:
            Dict with searchType, pages, coreKeywords, subKeywords, reason
        """
        # 🔥 Smart TOC filtering for different query types
        import re

        # 1. Detect conversation-reference queries (don't need TOC)
        # Patterns: "문제 1번", "1번 문제", "정답", "해설", "그거", "방금", etc.
        conversation_ref_patterns = [
            r'\d+번\s*(문제|정답|해설)',  # "1번 문제", "2번 정답"
            r'(문제|정답|해설)\s*\d+번',  # "문제 1번", "정답 2번"
            r'^(정답|해설|풀이)',  # "정답", "해설" (문장 시작)
            r'(그거|저거|방금|위|아래|이전)',  # 대화 참조
        ]

        is_conversation_ref = any(re.search(pattern, query, re.IGNORECASE) for pattern in conversation_ref_patterns)

        if is_conversation_ref:
            toc = ""  # Remove TOC for conversation-reference queries
            self.logger.info(f"🗣️ Conversation-reference query detected, TOC removed to save tokens")
        else:
            # 2. Detect chapter-specific queries and filter TOC
            chapter_pattern = r'(\d+)(?:-(\d+))?장'
            chapter_match = re.search(chapter_pattern, query)

            if chapter_match and toc:
                start_chapter = int(chapter_match.group(1))
                end_chapter = int(chapter_match.group(2)) if chapter_match.group(2) else start_chapter

                # Parse TOC JSON and filter for requested chapters
                try:
                    toc_lines = toc.strip().split('\n')
                    filtered_toc_lines = []
                    for line in toc_lines:
                        # Check if line contains any of the requested chapters
                        for ch in range(start_chapter, end_chapter + 1):
                            if f"{ch}장" in line:
                                filtered_toc_lines.append(line)
                                break

                    if filtered_toc_lines:
                        toc = '\n'.join(filtered_toc_lines)
                        self.logger.info(f"🎯 Filtered TOC to chapters {start_chapter}-{end_chapter} ({len(filtered_toc_lines)} entries, {len(toc)} chars)")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to filter TOC by chapters: {e}")

        prompt = f"""**🔍 질문 분석 및 검색 타입 결정**

질문: "{query}"

대화: {history_user_query[:300] if history_user_query else '없음'}
이전: {json.dumps(previous_context, ensure_ascii=False) if previous_context else '없음'}

**검색 타입:**
1. `toc`: 목차 탐색
2. `page`: 특정 페이지
3. `summary`: 챕터 요약 (예: "1장 요약" → 목차에서 1장 페이지 찾아 `pages` 지정!)
4. `keyword`: 키워드 검색
5. `semantic`: 의미 검색
6. `quiz`: 문제 출제
7. `followup`: 인사/일상/이전 대화 참조
8. `irrelevant`: 교재 완전 무관 (**극히 드물게, 확실할 때만!**)
   - 예: "오늘 날씨", "요리 방법", "영화 추천"
   - ⚠️ 심리학/성격 관련 용어는 절대 irrelevant 아님!
   - 예: MBTI, 성격검사, 심리이론 → keyword/semantic 검색!

**🔥 중요:**
- "1장 요약", "2장 내용" → `summary` + 목차에서 페이지 찾아 `pages` 지정!
- **비교/차이 질문 → `keyword` 검색!**
  - "A와 B의 차이", "A vs B", "A와 B 비교" → `keyword`
  - 예: "내향형과 외향형의 차이는?" → `keyword` (summary 아님!)
- 인사("안녕") → `followup`
- **키워드는 반드시 한국어로 생성!** (한국어 책 → 한국어 키워드)
  - 예: "내향형" (O), "introvert" (X)
  - 예: "자기효능감" (O), "self-efficacy" (X)

**JSON 출력:**
{{
  "searchType": "summary",
  "pages": "18-74",
  "coreKeywords": [],
  "subKeywords": [],
  "usePreviousContext": false,
  "reason": "1장 요약 요청"
}}

---
**책:** {title}
**목차:**
{toc}

**질문:** {query}
"""

        # 🔥 Dynamic TOC truncation to prevent token overflow
        # Estimate tokens: Korean text ~0.7 chars/token
        # Balance: 대형 TOC 지원 vs vLLM 토큰 제한 (8K context)
        max_toc_tokens = 6000  # Reserve tokens for TOC (optimized for ~200 entries)
        estimated_toc_tokens = int(len(toc) * 0.7)

        if estimated_toc_tokens > max_toc_tokens:
            max_toc_chars = int(max_toc_tokens / 0.7)
            toc_truncated = toc[:max_toc_chars]
            self.logger.warning(f"⚠️ TOC too long ({estimated_toc_tokens} tokens), truncating to {max_toc_tokens} tokens ({max_toc_chars} chars)")

            # Re-build prompt with truncated TOC
            prompt = f"""**🔍 질문 분석 및 검색 타입 결정**

질문: "{query}"

대화: {history_user_query[:300] if history_user_query else '없음'}
이전: {json.dumps(previous_context, ensure_ascii=False) if previous_context else '없음'}

**검색 타입:**
1. `toc`: 목차 탐색
2. `page`: 특정 페이지
3. `summary`: 챕터 요약 (예: "1장 요약" → 목차에서 1장 페이지 찾아 `pages` 지정!)
4. `keyword`: 키워드 검색
5. `semantic`: 의미 검색
6. `quiz`: 문제 출제
7. `followup`: 인사/일상/이전 대화 참조
8. `irrelevant`: 교재 완전 무관 (**극히 드물게, 확실할 때만!**)
   - 예: "오늘 날씨", "요리 방법", "영화 추천"
   - ⚠️ 심리학/성격 관련 용어는 절대 irrelevant 아님!
   - 예: MBTI, 성격검사, 심리이론 → keyword/semantic 검색!

**🔥 중요:**
- "1장 요약", "2장 내용" → `summary` + 목차에서 페이지 찾아 `pages` 지정!
- **비교/차이 질문 → `keyword` 검색!**
  - "A와 B의 차이", "A vs B", "A와 B 비교" → `keyword`
  - 예: "내향형과 외향형의 차이는?" → `keyword` (summary 아님!)
- 인사("안녕") → `followup`
- **키워드는 반드시 한국어로 생성!** (한국어 책 → 한국어 키워드)
  - 예: "내향형" (O), "introvert" (X)
  - 예: "자기효능감" (O), "self-efficacy" (X)

**JSON 출력:**
{{
  "searchType": "summary",
  "pages": "18-74",
  "coreKeywords": [],
  "subKeywords": [],
  "usePreviousContext": false,
  "reason": "1장 요약 요청"
}}

---
**책:** {title}
**목차 (일부, 전체 {len(toc)} 문자 중 {max_toc_chars} 문자):**
{toc_truncated}

**질문:** {query}
"""

        response_schema = {
            "type": "object",
            "properties": {
                "usePreviousContext": {
                    "type": "boolean",
                    "description": "이전 대화 컨텍스트를 재사용할지 여부"
                },
                "searchType": {
                    "type": "string",
                    "enum": ["page", "keyword", "summary", "toc", "irrelevant", "followup", "quiz", "semantic"],
                    "description": "검색에 사용할 방법"
                },
                "pages": {
                    "type": "string",
                    "description": "정보가 필요한 페이지 범위"
                },
                "coreKeywords": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string"},
                            "alternatives": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "subKeywords": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string"},
                            "alternatives": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "reason": {
                    "type": "string",
                    "description": "검색타입을 결정한 이유"
                }
            },
            "required": ["usePreviousContext", "searchType", "pages", "reason"]
        }

        try:
            response = await self.generate_text(prompt, response_schema=response_schema, mode="chat")
            return response
        except Exception as e:
            self.logger.error(f"Error in get_search_type: {e}")
            raise

    async def get_format_toc(self, manual_toc: str) -> str:
        """
        Format table of contents (replaces getFormatToc.js)

        Args:
            manual_toc: Raw TOC text

        Returns:
            Formatted TOC string
        """
        prompt = f"""
[입력된 목차데이터]를 분석하여 다음 규칙에 따라 '목차|페이지번호' 형식으로 정리해서 다시 출력해 주세요. 다른 출력 없이, 오직 '목차|페이지번호' 형식만 출력하세요.

목차 정리 규칙:


1. 목차의 내용을 분석해서, 부, 장, 절의 구분을 확인하세요. 텍스트 그대로가 아니라 의미적으로 파악해야 합니다.
책마다 부,장,절의 표시하는 방법이 다르며, 부,장,절의 표시가 없는 경우도 있습니다.

2. 입력된 목차데이터의 순서대로 출력하세요. 다양한 종류의 책으로 진행되므로, 각 책의 부,장,절의 규칙은 다를 수 있습니다.

3. 부와 장은 1부, 1장 이라고 통일해서 작성합니다.
예) 1부 부의제목텍스트
예) 1장 장의제목테스트

4. 절의 경우에는 해당 절이 속하는 장의 번호를 이용해서 순서대로 표시합니다.
예) 1장에 포함되었다면, 1-1 절의제목

5. 장이나 부 이전에 나오는 서문, 소개 등 독립적인 내용에는 장이나 절 표시를 하지 않고 제목만 기입합니다.
예시: 옮긴이 서문|5



### [입력된 목차데이터]
{manual_toc}
"""

        try:
            response = await self.generate_text(prompt, mode="chat")
            return response['result']
        except Exception as e:
            self.logger.error(f"Error in get_format_toc: {e}")
            raise

    async def extract_keywords(
        self,
        text: str,
        model: Optional[str] = None,
        num_keywords: int = 5
    ) -> List[str]:
        """
        Extract keywords from text (used by Chat API)

        This is a wrapper around the generate_text method,
        similar to the original extract_keywords in ollama_service.

        Args:
            text: Input text
            model: Model to use
            num_keywords: Number of keywords to extract

        Returns:
            List of keywords
        """
        prompt = f"""다음 텍스트에서 가장 중요한 키워드 {num_keywords}개를 추출해주세요.
키워드만 쉼표로 구분하여 나열해주세요.

텍스트:
{text}

키워드:"""

        try:
            model_name = model or self.model
            response = await self.client.generate(
                model=model_name,
                prompt=prompt,
                keep_alive=-1  # Keep model in memory indefinitely
            )

            # Parse keywords
            keywords_text = response['response'].strip()
            keywords = [k.strip() for k in keywords_text.split(',')]

            return keywords[:num_keywords]

        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
            raise

    async def simple_chat(
        self,
        message: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Simple chat function (used by Chat API /chat endpoint)

        This is simpler than get_answer() - no book-specific prompts.
        Just regular conversation with optional context.

        Args:
            message: User message
            context: Optional context (e.g., PDF content)
            conversation_history: Previous messages
            model: Model to use

        Returns:
            AI response text
        """
        try:
            model_name = model or self.model

            # Build messages
            messages = []

            # Add conversation history
            if conversation_history:
                messages.extend(conversation_history)

            # Build user message with context
            user_content = message
            if context:
                user_content = f"""다음 문서 내용을 참고하여 질문에 답변해주세요:

문서 내용:
{context}

질문: {message}"""

            messages.append({
                'role': 'user',
                'content': user_content
            })

            # Generate response using Ollama client chat API
            from ollama import AsyncClient
            client = AsyncClient(host=settings.OLLAMA_BASE_URL)

            response = await client.chat(
                model=model_name,
                messages=messages
            )

            return response['message']['content']

        except Exception as e:
            self.logger.error(f"Error in simple_chat: {e}")
            raise


# Create singleton instance
llm_service = LLMService()
