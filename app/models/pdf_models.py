"""
PDF Models
Pydantic models for PDF-related operations
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from pathlib import Path


class PDFUploadRequest(BaseModel):
    """PDF upload request model"""
    isbn: str = Field(..., description="ISBN of the book", min_length=10, max_length=20)
    page_count: int = Field(..., gt=0, description="Number of pages in the PDF")
    file_size: int = Field(..., gt=0, description="File size in bytes")

    @validator('isbn')
    def validate_isbn(cls, v):
        """Validate ISBN format"""
        # Remove hyphens and spaces
        cleaned = v.replace('-', '').replace(' ', '')
        if not cleaned.isdigit():
            raise ValueError('ISBN must contain only digits')
        if len(cleaned) not in [10, 13]:
            raise ValueError('ISBN must be 10 or 13 digits')
        return cleaned


class PDFInfo(BaseModel):
    """PDF information model"""
    id: str
    isbn: str
    title: str
    file_path: str
    page_count: int
    file_size: int
    has_toc: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class PDFListResponse(BaseModel):
    """PDF list response model"""
    pdfs: List[PDFInfo]
    total: int


class PDFExtractRequest(BaseModel):
    """PDF text extraction request"""
    isbn: str
    page_number: Optional[int] = Field(None, ge=1, description="Specific page to extract")
    start_page: Optional[int] = Field(None, ge=1, description="Start page for range extraction")
    end_page: Optional[int] = Field(None, ge=1, description="End page for range extraction")


class PDFPage(BaseModel):
    """PDF page content model"""
    page_number: int
    content: str
    word_count: int = 0


class PDFExtractResponse(BaseModel):
    """PDF extraction response"""
    isbn: str
    title: str
    pages: List[PDFPage]
    total_pages: int
    extracted_pages: int


class PDFDeleteRequest(BaseModel):
    """PDF delete request"""
    isbn: str


class PDFSearchRequest(BaseModel):
    """PDF search request"""
    isbn: str
    query: str
    limit: int = Field(10, ge=1, le=100)


class PDFSearchResult(BaseModel):
    """PDF search result"""
    page_number: int
    content: str
    relevance_score: float = 0.0


class PDFSearchResponse(BaseModel):
    """PDF search response"""
    isbn: str
    query: str
    results: List[PDFSearchResult]
    total_results: int


class PDFProcessStatus(BaseModel):
    """PDF processing status"""
    isbn: str
    status: str  # pending, processing, completed, failed
    progress: int = Field(0, ge=0, le=100)
    message: Optional[str] = None
    pages_processed: int = 0
    total_pages: int = 0


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    message: str
    details: Optional[dict] = None
