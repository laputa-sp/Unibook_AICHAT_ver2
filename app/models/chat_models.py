"""
Chat Models
Pydantic models for AI chatbot operations
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ChatMessage(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    """Chat request from user"""
    message: str = Field(..., min_length=1, description="User message")
    isbn: Optional[str] = Field(None, description="ISBN of PDF to reference")
    page_number: Optional[int] = Field(None, ge=1, description="Specific page to reference")
    model: Optional[str] = Field(None, description="Ollama model to use")
    stream: bool = Field(False, description="Stream response")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")


class ChatResponse(BaseModel):
    """Chat response from AI"""
    message: str = Field(..., description="AI response")
    model: str = Field(..., description="Model used")
    context_used: bool = Field(False, description="Whether PDF context was used")
    conversation_id: str = Field(..., description="Conversation ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PDFQuestionRequest(BaseModel):
    """Question about PDF content"""
    isbn: str = Field(..., description="ISBN of the PDF")
    question: str = Field(..., min_length=1, description="Question about the PDF")
    page_number: Optional[int] = Field(None, ge=1, description="Specific page to query")
    page_range: Optional[str] = Field(None, description="Page range (e.g., '1-10')")
    model: Optional[str] = Field(None, description="Model to use")


class PDFQuestionResponse(BaseModel):
    """Answer to PDF question"""
    question: str
    answer: str
    isbn: str
    pages_referenced: List[int] = []
    model: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SummaryRequest(BaseModel):
    """Request to summarize text"""
    isbn: Optional[str] = Field(None, description="ISBN of PDF to summarize")
    text: Optional[str] = Field(None, description="Text to summarize directly")
    page_number: Optional[int] = Field(None, ge=1, description="Specific page")
    page_range: Optional[str] = Field(None, description="Page range")
    max_length: int = Field(500, ge=100, le=2000, description="Max summary length")
    model: Optional[str] = Field(None, description="Model to use")


class SummaryResponse(BaseModel):
    """Summary result"""
    summary: str
    original_length: int
    summary_length: int
    model: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelInfo(BaseModel):
    """Ollama model information"""
    name: str
    size: Optional[int] = None
    modified_at: Optional[str] = None
    family: Optional[str] = None
    parameter_size: Optional[str] = None


class ModelsListResponse(BaseModel):
    """List of available models"""
    models: List[ModelInfo]
    total: int
    default_model: str


class ConversationHistory(BaseModel):
    """Conversation history"""
    conversation_id: str
    messages: List[ChatMessage]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class KeywordsRequest(BaseModel):
    """Request to extract keywords"""
    text: Optional[str] = Field(None, description="Text to extract keywords from")
    isbn: Optional[str] = Field(None, description="ISBN of PDF")
    page_number: Optional[int] = Field(None, ge=1, description="Specific page")
    num_keywords: int = Field(5, ge=1, le=20, description="Number of keywords to extract")
    model: Optional[str] = Field(None, description="Model to use")


class KeywordsResponse(BaseModel):
    """Keywords extraction result"""
    keywords: List[str]
    source: str  # 'text' or 'pdf'
    model: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Legacy API Models (for backward compatibility)
class LegacyAPIRequest(BaseModel):
    """Legacy Node.js API request format"""
    isbn: str = Field(..., description="ISBN of the book")
    userid: str = Field(..., description="User ID")
    sessionid: str = Field(..., description="Session ID")
    query: str = Field(..., min_length=1, description="User query")
    pageFrom: Optional[int] = Field(None, description="Starting page number")
    selectedText: Optional[str] = Field(None, description="Selected text from PDF")


class LegacyAPIResponse(BaseModel):
    """Legacy Node.js API response format"""
    result: str = Field(..., description="AI response text")
