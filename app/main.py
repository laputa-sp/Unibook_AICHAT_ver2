"""
Main FastAPI Application
"""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import logging

from app.config import settings
from app.routes import health, pdf, chat, monitoring, query_logs
from app.services.pdf_database import pdf_database
from app.utils.query_logger import get_query_logger
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Configure logging with Korean timezone
import time
import datetime

class KSTFormatter(logging.Formatter):
    """Custom formatter that uses Korean Standard Time"""
    def formatTime(self, record, datefmt=None):
        # Convert to KST (UTC+9)
        kst = datetime.datetime.fromtimestamp(record.created) + datetime.timedelta(hours=9)
        if datefmt:
            s = kst.strftime(datefmt)
        else:
            s = kst.strftime("%Y-%m-%d %H:%M:%S")
        return s

# Set up logging with KST (clear existing handlers to avoid duplicates)
logging.root.handlers.clear()
handler = logging.StreamHandler()
handler.setFormatter(KSTFormatter('%(asctime)s [KST] - %(name)s - %(levelname)s - %(message)s'))
logging.root.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
logging.root.addHandler(handler)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
if settings.STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")

# Upload images
if (settings.UPLOAD_DIR / "images").exists():
    app.mount(
        "/uploads/images",
        StaticFiles(directory=str(settings.UPLOAD_DIR / "images")),
        name="uploads"
    )

# Templates
templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(pdf.router, prefix="/api/pdf", tags=["PDF"])
app.include_router(chat.router, prefix="/api/chat", tags=["AI Chat"])
app.include_router(monitoring.router, tags=["Monitoring"])
app.include_router(query_logs.router, tags=["Query Logs"])


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database and warm up LLM models on startup"""
    logger.info("Initializing database...")
    await pdf_database.init_database()
    logger.info("Database initialized successfully")

    logger.info("Initializing query logger...")
    query_logger = get_query_logger()
    await query_logger.init_database()
    logger.info("Query logger initialized successfully")

    # Warm up Ollama model (preload into memory)
    logger.info("Warming up Ollama model...")
    try:
        from app.services.llm_service import llm_service
        # Send a small dummy request to load model into memory
        await llm_service.ollama_client.chat(
            model=llm_service.ollama_model,
            messages=[{'role': 'user', 'content': '안녕'}]
        )
        logger.info(f"✅ Ollama model '{llm_service.ollama_model}' warmed up successfully")
    except Exception as e:
        logger.warning(f"⚠️ Failed to warm up Ollama model: {e}")
        logger.warning("Model will be loaded on first request")


@app.get("/")
async def root(request: Request):
    """Root endpoint"""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint

    Exposes LLM performance metrics for Prometheus scraping.
    Includes latency, request counts, token usage, etc.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={"error": "Not Found", "message": str(exc)}
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
