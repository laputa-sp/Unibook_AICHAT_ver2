"""
Application Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings"""

    # Application
    APP_NAME: str = "E-Book PDF AI Chatbot"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "uploads/app.db")

    # Session
    SESSION_SECRET: str = os.getenv("SESSION_SECRET", "your-secret-key-change-this-in-production")
    SESSION_MAX_AGE: int = 24 * 60 * 60  # 24 hours in seconds

    # LLM Engine Selection
    LLM_ENGINE: str = os.getenv("LLM_ENGINE", "ollama")  # "ollama" or "vllm"

    # Ollama
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama2")

    # vLLM (OpenAI-compatible API)
    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
    VLLM_MODEL_NAME: str = os.getenv("VLLM_MODEL_NAME", "gpt-oss:20b")
    VLLM_API_KEY: str = os.getenv("VLLM_API_KEY", "EMPTY")

    # LLM Fallback Configuration
    ENABLE_LLM_FALLBACK: bool = os.getenv("ENABLE_LLM_FALLBACK", "true").lower() == "true"
    FALLBACK_ENGINE: str = os.getenv("FALLBACK_ENGINE", "ollama")  # Fallback to this engine on failure

    # Qdrant Vector Database
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_ENABLED: bool = os.getenv("QDRANT_ENABLED", "true").lower() == "true"

    # Embedding Settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1024"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # File Upload
    UPLOAD_DIR: Path = Path("uploads")
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: set = {".pdf"}

    # Static files
    STATIC_DIR: Path = Path("public")
    TEMPLATES_DIR: Path = Path("app/templates")

    def __init__(self):
        """Ensure required directories exist"""
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        (self.UPLOAD_DIR / "images").mkdir(parents=True, exist_ok=True)
        self.STATIC_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
