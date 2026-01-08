"""
Chat Routes
AI chatbot API endpoints using Ollama
"""
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
import logging
import uuid
from typing import Dict
import re

from app.config import settings
from app.models.chat_models import (
    ChatRequest,
    ChatResponse,
    PDFQuestionRequest,
    PDFQuestionResponse,
    SummaryRequest,
    SummaryResponse,
    ModelsListResponse,
    ModelInfo,
    KeywordsRequest,
    KeywordsResponse,
    LegacyAPIRequest,
    LegacyAPIResponse
)
from app.services.ollama_service import ollama_service  # For list_models only
from app.services.llm_service import llm_service  # Main LLM service (Gemini prompts + Ollama)
from app.services.pdf_database import pdf_database
from app.utils.conversation_cache import ConversationCache

router = APIRouter()
logger = logging.getLogger(__name__)

# Conversation cache with LRU eviction and TTL
conversations = ConversationCache(max_size=1000, max_age_hours=24)

# Input validation limits
MAX_QUERY_LENGTH = 5000
MAX_SELECTED_TEXT_LENGTH = 10000


@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    """
    List all available Ollama models

    Returns a list of models that can be used for chat
    """
    try:
        models = await ollama_service.list_models()

        return ModelsListResponse(
            models=[ModelInfo(**m) for m in models],
            total=len(models),
            default_model=settings.OLLAMA_MODEL
        )

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing models: {str(e)}"
        )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a chat message to AI

    - **message**: Your message
    - **isbn**: (Optional) Reference a specific PDF
    - **page_number**: (Optional) Reference a specific page
    - **model**: (Optional) Specific model to use
    - **conversation_id**: (Optional) Continue a conversation
    """
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Get conversation history
        history = await conversations.get(conversation_id)

        # Get PDF context if ISBN is provided
        context = None
        context_used = False

        if request.isbn:
            pdf = await pdf_database.get_pdf_by_isbn(request.isbn)

            if not pdf:
                raise HTTPException(
                    status_code=404,
                    detail=f"PDF with ISBN {request.isbn} not found"
                )

            # Get page content
            if request.page_number:
                page_data = await pdf_database.get_page_content(
                    pdf['id'],
                    request.page_number
                )

                if page_data:
                    context = f"[페이지 {request.page_number}]\n{page_data['content']}"
                    context_used = True

            else:
                # Get first few pages as context (max 3 pages)
                pages = await pdf_database.get_pages_range(pdf['id'], 1, 3)

                if pages:
                    context = "\n\n".join([
                        f"[페이지 {p['page_number']}]\n{p['content']}"
                        for p in pages
                    ])
                    context_used = True

        # Get AI response using new LLM service
        response_text = await llm_service.simple_chat(
            message=request.message,
            context=context,
            conversation_history=history,
            model=request.model
        )

        # Update conversation history
        await conversations.append(conversation_id, [
            {'role': 'user', 'content': request.message},
            {'role': 'assistant', 'content': response_text}
        ])

        model_used = request.model or settings.OLLAMA_MODEL

        return ChatResponse(
            message=response_text,
            model=model_used,
            context_used=context_used,
            conversation_id=conversation_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat: {str(e)}"
        )


@router.post("/ask", response_model=PDFQuestionResponse)
async def ask_pdf_question(request: PDFQuestionRequest):
    """
    Ask a question about a PDF

    The AI will answer based on the PDF content.

    - **isbn**: ISBN of the PDF
    - **question**: Your question
    - **page_number**: (Optional) Specific page to query
    - **page_range**: (Optional) Page range (e.g., "1-10")
    """
    try:
        # Get PDF info
        pdf = await pdf_database.get_pdf_by_isbn(request.isbn)

        if not pdf:
            raise HTTPException(
                status_code=404,
                detail=f"PDF with ISBN {request.isbn} not found"
            )

        # Determine pages to query
        pages_to_query = []

        if request.page_number:
            pages_to_query = [request.page_number]

        elif request.page_range:
            # Parse page range (e.g., "1-10")
            try:
                start, end = map(int, request.page_range.split('-'))
                pages_to_query = list(range(start, min(end + 1, pdf['page_count'] + 1)))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid page range format. Use 'start-end' (e.g., '1-10')"
                )

        else:
            # Default: first 5 pages
            pages_to_query = list(range(1, min(6, pdf['page_count'] + 1)))

        # Get content from pages
        context_parts = []
        for page_num in pages_to_query:
            page_data = await pdf_database.get_page_content(pdf['id'], page_num)

            if page_data and page_data['content']:
                context_parts.append(
                    f"[페이지 {page_num}]\n{page_data['content']}"
                )

        if not context_parts:
            raise HTTPException(
                status_code=404,
                detail="No content found in specified pages"
            )

        context = "\n\n".join(context_parts)

        # Get PDF info for get_answer (needs title and TOC)
        # For now, we'll use a simplified version
        # TODO: Store and retrieve TOC from database

        # Get answer from AI using enhanced LLM service
        # Using get_answer with full Gemini-quality prompts
        response = await llm_service.get_answer(
            title=pdf.get('title', 'Unknown'),
            texts=context,
            history_user_query="[]",  # TODO: Add conversation history support
            question=request.question,
            toc=""  # TODO: Add TOC support
        )

        answer = response['result']

        model_used = request.model or settings.OLLAMA_MODEL

        return PDFQuestionResponse(
            question=request.question,
            answer=answer,
            isbn=request.isbn,
            pages_referenced=pages_to_query,
            model=model_used
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error answering question: {str(e)}"
        )


@router.post("/summarize", response_model=SummaryResponse)
async def summarize(request: SummaryRequest):
    """
    Summarize text or PDF content

    - **text**: Direct text to summarize (OR)
    - **isbn**: ISBN of PDF to summarize
    - **page_number**: Specific page (OR)
    - **page_range**: Page range (e.g., "1-10")
    - **max_length**: Maximum summary length
    """
    try:
        text_to_summarize = None

        if request.text:
            # Direct text
            text_to_summarize = request.text

        elif request.isbn:
            # Get PDF content
            pdf = await pdf_database.get_pdf_by_isbn(request.isbn)

            if not pdf:
                raise HTTPException(
                    status_code=404,
                    detail=f"PDF with ISBN {request.isbn} not found"
                )

            # Determine pages
            if request.page_number:
                page_data = await pdf_database.get_page_content(
                    pdf['id'],
                    request.page_number
                )

                if page_data:
                    text_to_summarize = page_data['content']

            elif request.page_range:
                try:
                    start, end = map(int, request.page_range.split('-'))
                    pages = await pdf_database.get_pages_range(
                        pdf['id'],
                        start,
                        end
                    )

                    text_to_summarize = "\n\n".join([p['content'] for p in pages])

                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid page range format"
                    )

            else:
                # Default: first 5 pages
                pages = await pdf_database.get_pages_range(
                    pdf['id'],
                    1,
                    min(5, pdf['page_count'])
                )

                text_to_summarize = "\n\n".join([p['content'] for p in pages])

        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'text' or 'isbn' must be provided"
            )

        if not text_to_summarize:
            raise HTTPException(
                status_code=404,
                detail="No text found to summarize"
            )

        # Generate summary using enhanced LLM service
        # Note: get_summary returns structured data, but we just need summary text
        # For simple summarization, we'll use a direct prompt
        prompt = f"""다음 텍스트를 {request.max_length}자 이내로 요약해주세요. 핵심 내용만 간결하게 정리해주세요.

텍스트:
{text_to_summarize}

요약:"""

        response = await llm_service.generate_text(prompt, mode="chat")
        summary = response['result']

        model_used = request.model or settings.OLLAMA_MODEL

        return SummaryResponse(
            summary=summary,
            original_length=len(text_to_summarize),
            summary_length=len(summary),
            model=model_used
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating summary: {str(e)}"
        )


@router.post("/keywords", response_model=KeywordsResponse)
async def extract_keywords(request: KeywordsRequest):
    """
    Extract keywords from text or PDF

    - **text**: Direct text (OR)
    - **isbn**: ISBN of PDF
    - **page_number**: Specific page
    - **num_keywords**: Number of keywords to extract
    """
    try:
        text_to_analyze = None
        source = 'text'

        if request.text:
            text_to_analyze = request.text

        elif request.isbn:
            pdf = await pdf_database.get_pdf_by_isbn(request.isbn)

            if not pdf:
                raise HTTPException(
                    status_code=404,
                    detail=f"PDF with ISBN {request.isbn} not found"
                )

            if request.page_number:
                page_data = await pdf_database.get_page_content(
                    pdf['id'],
                    request.page_number
                )

                if page_data:
                    text_to_analyze = page_data['content']
                    source = f'pdf_page_{request.page_number}'

            else:
                # Use first 3 pages
                pages = await pdf_database.get_pages_range(pdf['id'], 1, 3)
                text_to_analyze = "\n\n".join([p['content'] for p in pages])
                source = 'pdf'

        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'text' or 'isbn' must be provided"
            )

        if not text_to_analyze:
            raise HTTPException(
                status_code=404,
                detail="No text found to analyze"
            )

        # Extract keywords using enhanced LLM service
        keywords = await llm_service.extract_keywords(
            text=text_to_analyze,
            model=request.model,
            num_keywords=request.num_keywords
        )

        model_used = request.model or settings.OLLAMA_MODEL

        return KeywordsResponse(
            keywords=keywords,
            source=source,
            model=model_used
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting keywords: {str(e)}"
        )


@router.post("/v1/response/stream")
async def legacy_api_response_stream(request: LegacyAPIRequest):
    """
    Streaming version of legacy API endpoint

    Streams the answer in real-time using Server-Sent Events (SSE).
    Note: Streaming responses bypass the LLM cache for real-time delivery.

    - **isbn**: ISBN of the book
    - **userid**: User ID (stored for logging)
    - **sessionid**: Session ID (used as conversation_id)
    - **query**: User's question
    - **pageFrom**: (Optional) Starting page number
    - **selectedText**: (Optional) Text selected by user

    Returns:
    Stream of Server-Sent Events with format:
    data: {"chunk": "text piece", "done": false}
    data: {"chunk": "", "done": true, "metadata": {...}}
    """
    try:
        from app.services.qa_service import qa_service

        # Validate required fields
        if not request.userid or not request.userid.strip():
            raise HTTPException(
                status_code=400,
                detail="유저 아이디가 필요합니다."
            )

        if not request.sessionid or not request.sessionid.strip():
            raise HTTPException(
                status_code=400,
                detail="세션 아이디가 필요합니다."
            )

        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="사용자 발화(query)가 필요합니다."
            )

        # Input validation
        if len(request.query) > MAX_QUERY_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"질문이 너무 깁니다. 최대 {MAX_QUERY_LENGTH}자까지 가능합니다."
            )

        if request.selectedText and len(request.selectedText) > MAX_SELECTED_TEXT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"선택된 텍스트가 너무 깁니다. 최대 {MAX_SELECTED_TEXT_LENGTH}자까지 가능합니다."
            )

        # Sanitize input
        request.query = re.sub(r'<[^>]+>', '', request.query).strip()
        if request.selectedText:
            request.selectedText = re.sub(r'<[^>]+>', '', request.selectedText).strip()

        # Stream generator
        async def generate_stream():
            """Generate SSE stream with answer chunks"""
            import json

            try:
                # Get streaming response from QA service
                async for chunk_data in qa_service.handle_json_results_stream(
                    isbn=request.isbn,
                    query=request.query,
                    page_from=request.pageFrom,
                    selected_text=request.selectedText,
                    userid=request.userid,
                    sessionid=request.sessionid
                ):
                    # Send chunk as SSE
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

            except Exception as e:
                logger.error(f"Error in stream generation: {e}")
                error_data = {
                    "error": str(e),
                    "done": True
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in streaming API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )


@router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation history

    - **conversation_id**: ID of the conversation to delete
    """
    deleted = await conversations.delete(conversation_id)
    if deleted:
        return {"success": True, "message": "Conversation deleted"}
    else:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )


@router.get("/conversation/stats")
async def get_conversation_stats():
    """Get conversation cache statistics"""
    return conversations.stats()


@router.get("/cache/stats")
async def get_llm_cache_stats():
    """
    Get LLM response cache statistics

    Returns cache performance metrics including:
    - Size and capacity
    - Hit/miss counts and rate
    - Evictions and expirations
    - TTL configuration
    """
    from app.utils.llm_cache import get_llm_cache

    cache = get_llm_cache()
    return cache.get_stats()


# Legacy API Endpoint (for backward compatibility with existing apps)
@router.post("/v1/response", response_model=LegacyAPIResponse)
async def legacy_api_response(request: LegacyAPIRequest):
    """
    Legacy API endpoint for backward compatibility with Node.js version

    This endpoint maintains the same request/response format as the original
    Node.js API (/api/v1/response) so existing client apps don't need changes.

    Now uses the complete QA service that replicates Node.js logic:
    - 2-stage LLM: getSearchType() → getAnswer()
    - FTS keyword search with core/sub keywords
    - TOC-based search strategies
    - Context reuse across conversations

    - **isbn**: ISBN of the book
    - **userid**: User ID (stored for logging)
    - **sessionid**: Session ID (used as conversation_id)
    - **query**: User's question
    - **pageFrom**: (Optional) Starting page number
    - **selectedText**: (Optional) Text selected by user
    """
    try:
        # Import QA service
        from app.services.qa_service import qa_service

        # Validate required fields
        if not request.userid or not request.userid.strip():
            raise HTTPException(
                status_code=400,
                detail="유저 아이디가 필요합니다."
            )

        if not request.sessionid or not request.sessionid.strip():
            raise HTTPException(
                status_code=400,
                detail="세션 아이디가 필요합니다."
            )

        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="사용자 발화(query)가 필요합니다."
            )

        # Input validation - length limits
        if len(request.query) > MAX_QUERY_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"질문이 너무 깁니다. 최대 {MAX_QUERY_LENGTH}자까지 가능합니다."
            )

        if request.selectedText and len(request.selectedText) > MAX_SELECTED_TEXT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"선택된 텍스트가 너무 깁니다. 최대 {MAX_SELECTED_TEXT_LENGTH}자까지 가능합니다."
            )

        # Sanitize input - remove HTML/script tags
        request.query = re.sub(r'<[^>]+>', '', request.query).strip()
        if request.selectedText:
            request.selectedText = re.sub(r'<[^>]+>', '', request.selectedText).strip()

        # Use new QA service (complete Node.js flow)
        result = await qa_service.handle_json_results(
            isbn=request.isbn,
            query=request.query,
            page_from=request.pageFrom,
            selected_text=request.selectedText,
            userid=request.userid,
            sessionid=request.sessionid
        )

        infos = result['infos']
        answer = infos['answer']

        # Note: Conversation history is now saved in qa_service.py
        # to ensure it's in the same cache instance used for retrieval

        # Log the interaction with detailed metadata
        logger.info(
            f"Legacy API - ISBN: {request.isbn}, "
            f"User: {request.userid}, "
            f"Session: {request.sessionid}, "
            f"SearchType: {infos.get('검색유형')}, "
            f"Keywords: {infos.get('검색키워드', 'N/A')}, "
            f"Pages: {infos.get('페이지범위', 'N/A')}, "
            f"PagesFound: {len(infos.get('검색된페이지') or [])}, "
            f"LLM Time: {infos.get('검색요청LLM시간', 0)}ms + {infos.get('답변생성시간', 0)}ms, "
            f"DB Time: {infos.get('데이터베이스검색시간', 0)}ms, "
            f"Query length: {len(request.query)}, "
            f"Response length: {len(answer)}"
        )

        return LegacyAPIResponse(result=answer)

    except ValueError as e:
        # Handle specific errors (e.g., PDF not found)
        logger.error(f"Validation error in legacy API: {e}")
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in legacy API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )
