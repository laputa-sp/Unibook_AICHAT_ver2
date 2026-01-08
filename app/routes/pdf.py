"""
PDF Routes
API endpoints for PDF upload, extraction, and management
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse
from typing import Optional
import logging
from pathlib import Path
import shutil

from app.config import settings
from app.models.pdf_models import (
    PDFUploadRequest,
    PDFInfo,
    PDFListResponse,
    PDFExtractRequest,
    PDFExtractResponse,
    PDFDeleteRequest,
    PDFSearchRequest,
    PDFSearchResponse,
    PDFPage,
    ErrorResponse
)
from app.services.pdf_parser import pdf_parser
from app.services.pdf_database import pdf_database
from app.services.llm_service import LLMService

router = APIRouter()
logger = logging.getLogger(__name__)
llm_service = LLMService()


# Background task for LLM model warmup
async def warmup_llm_model():
    """Warm up LLM model by sending a simple request to load it into memory"""
    try:
        logger.info("🔥 Warming up LLM model...")
        # Simple prompt to trigger model loading
        await llm_service.client.generate(
            model=llm_service.model,
            prompt="안녕하세요",
            keep_alive=-1
        )
        logger.info("✅ LLM model warmed up successfully")
    except Exception as e:
        logger.warning(f"⚠️ LLM warmup failed (non-critical): {e}")


# Background task for PDF processing
async def process_pdf_background(pdf_id: str, pdf_path: Path):
    """Background task to extract and save PDF content + generate embeddings"""
    try:
        logger.info(f"Starting background processing for PDF {pdf_id}")

        # Extract all pages
        pages = await pdf_parser.extract_text_from_pdf(pdf_path)

        # Save pages to database in batches
        batch_size = 100
        for i in range(0, len(pages), batch_size):
            batch = pages[i:i + batch_size]
            await pdf_database.save_pages_batch(pdf_id, batch)

        logger.info(f"Completed processing {len(pages)} pages for PDF {pdf_id}")

        # Generate embeddings if Qdrant is enabled
        if settings.QDRANT_ENABLED:
            logger.info(f"🔮 임베딩 생성 시작 for PDF {pdf_id}")
            await generate_embeddings_for_pdf(pdf_id, pages)

    except Exception as e:
        logger.error(f"Error processing PDF {pdf_id}: {e}")


async def generate_embeddings_for_pdf(pdf_id: str, pages: list):
    """
    PDF 페이지에서 임베딩 생성 및 Qdrant 저장

    Args:
        pdf_id: PDF UUID
        pages: 페이지 리스트 [{"page_number": int, "content": str}, ...]
    """
    try:
        from app.services.chunk_service import get_chunk_service
        from app.services.embedding_service import get_embedding_service
        from app.services.qdrant_service import get_qdrant_service

        chunk_service = get_chunk_service(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        embedding_service = get_embedding_service()
        qdrant_service = get_qdrant_service()

        # 1. Qdrant 컬렉션 생성
        logger.info(f"📦 Qdrant 컬렉션 생성: {pdf_id}")
        qdrant_service.create_collection(pdf_id, recreate=True)

        # 2. 페이지를 청크로 분할
        logger.info(f"✂️  텍스트 청크 분할 중...")
        all_chunks = chunk_service.split_pages_batch(pdf_id, pages)

        if not all_chunks:
            logger.warning("청크가 생성되지 않음, 임베딩 스킵")
            return

        logger.info(f"📊 총 {len(all_chunks)}개 청크 생성됨")

        # 3. 배치 임베딩 생성
        logger.info(f"🧠 임베딩 생성 중 ({len(all_chunks)}개 청크)...")
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        embeddings = embedding_service.embed_batch(
            chunk_texts,
            batch_size=32,
            max_length=512,
            show_progress=True
        )

        # 4. Qdrant에 저장
        logger.info(f"💾 Qdrant에 저장 중...")
        success = qdrant_service.upsert_chunks(pdf_id, all_chunks, embeddings)

        if success:
            logger.info(f"✅ 임베딩 생성 완료: {len(embeddings)}개 벡터 저장됨")
        else:
            logger.error(f"❌ Qdrant 저장 실패")

    except Exception as e:
        logger.error(f"❌ 임베딩 생성 실패: {e}")
        # 실패해도 PDF 업로드 자체는 성공으로 처리 (임베딩은 선택적 기능)


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(..., description="PDF file to upload"),
    isbn: str = Form(..., description="ISBN of the book"),
    page_count: int = Form(..., description="Number of pages"),
    file_size: int = Form(..., description="File size in bytes")
):
    """
    Upload a PDF file

    - **pdf_file**: The PDF file to upload
    - **isbn**: ISBN number (10 or 13 digits)
    - **page_count**: Total number of pages
    - **file_size**: File size in bytes
    """
    pdf_id = None
    pdf_path = None

    try:
        # 1. File size validation
        MAX_FILE_SIZE = settings.MAX_UPLOAD_SIZE  # 50MB from config
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"파일 크기가 너무 큽니다. 최대 {MAX_FILE_SIZE // (1024*1024)}MB까지 가능합니다."
            )

        # 2. File extension validation
        if not pdf_file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="PDF 파일만 업로드 가능합니다."
            )

        # 3. File content validation (PDF Magic Number: %PDF-)
        content_chunk = await pdf_file.read(1024)
        await pdf_file.seek(0)  # Reset file pointer

        if not content_chunk.startswith(b'%PDF-'):
            raise HTTPException(
                status_code=400,
                detail="유효한 PDF 파일이 아닙니다."
            )

        # 4. ISBN validation
        isbn_cleaned = isbn.replace('-', '').replace(' ', '')
        if not isbn_cleaned.isdigit() or len(isbn_cleaned) not in [10, 13]:
            raise HTTPException(
                status_code=400,
                detail="Invalid ISBN format"
            )

        # 5. Check if PDF with this ISBN already exists
        existing_pdf = await pdf_database.get_pdf_by_isbn(isbn_cleaned)
        if existing_pdf:
            raise HTTPException(
                status_code=409,
                detail=f"PDF with ISBN {isbn_cleaned} already exists"
            )

        # 6. Sanitize filename
        import os
        import re
        safe_filename = os.path.basename(pdf_file.filename)
        safe_filename = re.sub(r'[^\w\s.-]', '', safe_filename)
        title = safe_filename.replace('.pdf', '').strip()

        # 7. Register PDF in database
        pdf_id = await pdf_database.register_pdf(
            isbn=isbn_cleaned,
            title=title,
            file_path="",  # Will update after saving file
            page_count=page_count,
            file_size=file_size
        )

        # 8. Save PDF file
        pdf_dir = settings.UPLOAD_DIR / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / f"{pdf_id}.pdf"

        # Save in chunks (for large files)
        CHUNK_SIZE = 1024 * 1024  # 1MB
        try:
            with open(pdf_path, "wb") as buffer:
                await pdf_file.seek(0)
                while chunk := await pdf_file.read(CHUNK_SIZE):
                    buffer.write(chunk)
        finally:
            await pdf_file.close()

        # 9. Update file path in database
        await pdf_database.update_pdf_file_path(pdf_id, str(pdf_path))

        # 10. Schedule background task for processing
        background_tasks.add_task(process_pdf_background, pdf_id, pdf_path)

        logger.info(f"PDF uploaded successfully: {pdf_id} - {title}")

        return {
            "success": True,
            "message": "PDF uploaded successfully. Processing in background.",
            "pdf_id": pdf_id,
            "isbn": isbn_cleaned,
            "title": title
        }

    except HTTPException:
        raise
    except Exception as e:
        # Rollback: Delete DB record and file if created
        logger.error(f"Upload failed, rolling back: {e}", exc_info=True)

        if pdf_id:
            try:
                await pdf_database.delete_pdf_by_id(pdf_id)
                logger.info(f"Rolled back DB record for PDF {pdf_id}")
            except Exception as rollback_error:
                logger.error(f"Failed to rollback DB: {rollback_error}")

        if pdf_path and pdf_path.exists():
            try:
                pdf_path.unlink()
                logger.info(f"Rolled back file {pdf_path}")
            except Exception as rollback_error:
                logger.error(f"Failed to rollback file: {rollback_error}")

        raise HTTPException(
            status_code=500,
            detail="파일 업로드 중 오류가 발생했습니다."
        )


@router.get("/list", response_model=PDFListResponse)
async def list_pdfs():
    """
    Get list of all PDFs

    Returns a list of all registered PDFs with their metadata
    """
    try:
        pdfs = await pdf_database.get_all_pdfs()

        return PDFListResponse(
            pdfs=[PDFInfo(**pdf) for pdf in pdfs],
            total=len(pdfs)
        )

    except Exception as e:
        logger.error(f"Error listing PDFs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing PDFs: {str(e)}"
        )


@router.get("/info/{isbn}")
async def get_pdf_info(isbn: str, background_tasks: BackgroundTasks):
    """
    Get PDF information by ISBN

    Returns detailed information about a specific PDF
    """
    try:
        pdf = await pdf_database.get_pdf_by_isbn(isbn)

        if not pdf:
            raise HTTPException(
                status_code=404,
                detail=f"PDF with ISBN {isbn} not found"
            )

        # Get additional stats
        stats = await pdf_database.get_pdf_stats(pdf['id'])

        # Warm up LLM model in background for faster first response
        background_tasks.add_task(warmup_llm_model)

        return {
            **pdf,
            **stats
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting PDF info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting PDF info: {str(e)}"
        )


@router.post("/extract", response_model=PDFExtractResponse)
async def extract_pdf_content(request: PDFExtractRequest):
    """
    Extract text content from PDF

    - **isbn**: ISBN of the PDF
    - **page_number**: Extract specific page (optional)
    - **start_page**: Start page for range extraction (optional)
    - **end_page**: End page for range extraction (optional)
    """
    try:
        # Get PDF info
        pdf = await pdf_database.get_pdf_by_isbn(request.isbn)

        if not pdf:
            raise HTTPException(
                status_code=404,
                detail=f"PDF with ISBN {request.isbn} not found"
            )

        # Get pages from database
        if request.page_number:
            # Single page
            page_data = await pdf_database.get_page_content(pdf['id'], request.page_number)
            if page_data:
                pages = [PDFPage(**page_data)]
            else:
                pages = []

        elif request.start_page and request.end_page:
            # Range of pages
            pages_data = await pdf_database.get_pages_range(
                pdf['id'],
                request.start_page,
                request.end_page
            )
            pages = [PDFPage(**p) for p in pages_data]

        else:
            # All pages (limited to first 100)
            pages_data = await pdf_database.get_pages_range(
                pdf['id'],
                1,
                min(100, pdf['page_count'])
            )
            pages = [PDFPage(**p) for p in pages_data]

        return PDFExtractResponse(
            isbn=request.isbn,
            title=pdf['title'],
            pages=pages,
            total_pages=pdf['page_count'],
            extracted_pages=len(pages)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting PDF content: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting PDF content: {str(e)}"
        )


@router.post("/search", response_model=PDFSearchResponse)
async def search_pdf(request: PDFSearchRequest):
    """
    Search for text in PDF

    - **isbn**: ISBN of the PDF
    - **query**: Search query
    - **limit**: Maximum number of results (default: 10)
    """
    try:
        # Get PDF info
        pdf = await pdf_database.get_pdf_by_isbn(request.isbn)

        if not pdf:
            raise HTTPException(
                status_code=404,
                detail=f"PDF with ISBN {request.isbn} not found"
            )

        # Search in database
        results = await pdf_database.search_pages(
            pdf['id'],
            request.query,
            request.limit
        )

        # Format results
        search_results = []
        for r in results:
            search_results.append({
                'page_number': r['page_number'],
                'content': r['content'][:500],  # Limit content length
                'relevance_score': 1.0  # FTS rank would go here
            })

        return PDFSearchResponse(
            isbn=request.isbn,
            query=request.query,
            results=search_results,
            total_results=len(search_results)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching PDF: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching PDF: {str(e)}"
        )


@router.delete("/delete/{isbn}")
async def delete_pdf(isbn: str):
    """
    Delete PDF by ISBN

    Removes the PDF and all associated data from the system
    """
    try:
        # Get PDF info
        pdf = await pdf_database.get_pdf_by_isbn(isbn)

        if not pdf:
            raise HTTPException(
                status_code=404,
                detail=f"PDF with ISBN {isbn} not found"
            )

        # Delete PDF file
        pdf_path = Path(pdf['file_path'])
        if pdf_path.exists():
            pdf_path.unlink()

        # Delete from database
        deleted = await pdf_database.delete_pdf(isbn)

        if deleted:
            return {
                "success": True,
                "message": f"PDF {isbn} deleted successfully"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete PDF from database"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting PDF: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting PDF: {str(e)}"
        )


@router.get("/page/{isbn}/{page_number}")
async def get_page_content(isbn: str, page_number: int):
    """
    Get content of a specific page

    - **isbn**: ISBN of the PDF
    - **page_number**: Page number to retrieve
    """
    try:
        # Get PDF info
        pdf = await pdf_database.get_pdf_by_isbn(isbn)

        if not pdf:
            raise HTTPException(
                status_code=404,
                detail=f"PDF with ISBN {isbn} not found"
            )

        # Validate page number
        if page_number < 1 or page_number > pdf['page_count']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid page number. Must be between 1 and {pdf['page_count']}"
            )

        # Get page content
        page_data = await pdf_database.get_page_content(pdf['id'], page_number)

        if not page_data:
            raise HTTPException(
                status_code=404,
                detail=f"Page {page_number} not found or not yet processed"
            )

        return {
            "success": True,
            "isbn": isbn,
            "title": pdf['title'],
            "page": PDFPage(**page_data),
            "total_pages": pdf['page_count']
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting page content: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting page content: {str(e)}"
        )
