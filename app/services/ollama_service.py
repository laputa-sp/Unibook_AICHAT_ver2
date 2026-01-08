"""
Ollama Service
AI chatbot service using Ollama for local LLM
"""
import logging
from typing import List, Dict, Optional, AsyncGenerator
import ollama
from ollama import AsyncClient

from app.config import settings

logger = logging.getLogger(__name__)


class OllamaService:
    """Ollama AI service for chatbot functionality"""

    def __init__(self):
        self.client = AsyncClient(host=settings.OLLAMA_BASE_URL)
        self.model = settings.OLLAMA_MODEL
        self.logger = logging.getLogger(__name__)

    async def list_models(self) -> List[Dict]:
        """
        List all available Ollama models

        Returns:
            List of model information
        """
        try:
            response = await self.client.list()
            models = response.get('models', [])

            return [{
                'name': model.get('name'),
                'size': model.get('size'),
                'modified_at': model.get('modified_at'),
                'family': model.get('details', {}).get('family'),
                'parameter_size': model.get('details', {}).get('parameter_size')
            } for model in models]

        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            raise

    async def chat(
        self,
        message: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """
        Send a chat message to Ollama

        Args:
            message: User message
            context: Additional context (e.g., PDF content)
            conversation_history: Previous conversation messages
            model: Model to use (defaults to configured model)
            stream: Whether to stream the response

        Returns:
            AI response message
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

            # Generate response
            if stream:
                # For streaming, we'll return the full response for now
                full_response = ""
                async for part in await self.chat_stream(message, context, conversation_history, model_name):
                    full_response += part
                return full_response
            else:
                response = await self.client.chat(
                    model=model_name,
                    messages=messages
                )

                return response['message']['content']

        except Exception as e:
            self.logger.error(f"Error in chat: {e}")
            raise

    async def chat_stream(
        self,
        message: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
        model: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response from Ollama

        Args:
            message: User message
            context: Additional context
            conversation_history: Previous messages
            model: Model to use

        Yields:
            Response chunks
        """
        try:
            model_name = model or self.model

            # Build messages
            messages = []

            if conversation_history:
                messages.extend(conversation_history)

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

            # Stream response
            stream = await self.client.chat(
                model=model_name,
                messages=messages,
                stream=True
            )

            async for chunk in stream:
                if 'message' in chunk:
                    content = chunk['message'].get('content', '')
                    if content:
                        yield content

        except Exception as e:
            self.logger.error(f"Error in streaming chat: {e}")
            raise

    async def generate_summary(
        self,
        text: str,
        model: Optional[str] = None,
        max_length: int = 500
    ) -> str:
        """
        Generate a summary of the given text

        Args:
            text: Text to summarize
            model: Model to use
            max_length: Maximum summary length in words

        Returns:
            Summary text
        """
        try:
            model_name = model or self.model

            prompt = f"""다음 텍스트를 {max_length}자 이내로 요약해주세요. 핵심 내용만 간결하게 정리해주세요.

텍스트:
{text}

요약:"""

            response = await self.client.generate(
                model=model_name,
                prompt=prompt
            )

            return response['response'].strip()

        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            raise

    async def answer_question(
        self,
        question: str,
        context: str,
        model: Optional[str] = None
    ) -> str:
        """
        Answer a question based on provided context

        Args:
            question: User question
            context: Context information (e.g., PDF page content)
            model: Model to use

        Returns:
            Answer to the question
        """
        try:
            model_name = model or self.model

            prompt = f"""다음 문서 내용을 바탕으로 질문에 답변해주세요.
답변은 문서 내용에 근거하여 정확하고 간결하게 작성해주세요.
문서에 없는 내용은 추측하지 말고 "문서에 해당 정보가 없습니다"라고 답변해주세요.

문서 내용:
{context}

질문: {question}

답변:"""

            response = await self.client.generate(
                model=model_name,
                prompt=prompt
            )

            return response['response'].strip()

        except Exception as e:
            self.logger.error(f"Error answering question: {e}")
            raise

    async def extract_keywords(
        self,
        text: str,
        model: Optional[str] = None,
        num_keywords: int = 5
    ) -> List[str]:
        """
        Extract keywords from text

        Args:
            text: Input text
            model: Model to use
            num_keywords: Number of keywords to extract

        Returns:
            List of keywords
        """
        try:
            model_name = model or self.model

            prompt = f"""다음 텍스트에서 가장 중요한 키워드 {num_keywords}개를 추출해주세요.
키워드만 쉼표로 구분하여 나열해주세요.

텍스트:
{text}

키워드:"""

            response = await self.client.generate(
                model=model_name,
                prompt=prompt
            )

            # Parse keywords
            keywords_text = response['response'].strip()
            keywords = [k.strip() for k in keywords_text.split(',')]

            return keywords[:num_keywords]

        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
            raise

    async def check_model_availability(self, model_name: str) -> bool:
        """
        Check if a model is available

        Args:
            model_name: Name of the model to check

        Returns:
            True if model is available
        """
        try:
            models = await self.list_models()
            return any(m['name'] == model_name for m in models)

        except Exception as e:
            self.logger.error(f"Error checking model availability: {e}")
            return False

    async def pull_model(self, model_name: str) -> bool:
        """
        Pull/download a model from Ollama

        Args:
            model_name: Name of the model to pull

        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Pulling model: {model_name}")

            await self.client.pull(model_name)

            self.logger.info(f"Successfully pulled model: {model_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error pulling model {model_name}: {e}")
            return False


# Create singleton instance
ollama_service = OllamaService()
