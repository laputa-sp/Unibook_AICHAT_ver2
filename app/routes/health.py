"""
Health Check Routes
"""
from fastapi import APIRouter, status
from datetime import datetime
import aiosqlite
import httpx
from typing import Dict, Any

from app.config import settings

router = APIRouter()


async def check_database() -> Dict[str, Any]:
    """Check database connection"""
    try:
        async with aiosqlite.connect(settings.DATABASE_URL) as db:
            await db.execute("SELECT 1")
            return {
                "status": "healthy",
                "message": "Database connection successful"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}"
        }


async def check_ollama() -> Dict[str, Any]:
    """Check Ollama service connection"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [model.get("name") for model in data.get("models", [])]
                return {
                    "status": "healthy",
                    "message": "Ollama service is running",
                    "models": models,
                    "configured_model": settings.OLLAMA_MODEL
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": f"Ollama service returned status {response.status_code}"
                }
    except httpx.ConnectError:
        return {
            "status": "unavailable",
            "message": "Cannot connect to Ollama service. Make sure Ollama is running.",
            "hint": "Run 'ollama serve' to start the service"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Ollama health check failed: {str(e)}"
        }


async def check_vllm() -> Dict[str, Any]:
    """Check vLLM service connection"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check vLLM health endpoint
            response = await client.get(f"{settings.VLLM_BASE_URL}/health")
            if response.status_code == 200:
                # Try to get model info
                try:
                    models_response = await client.get(f"{settings.VLLM_BASE_URL}/v1/models")
                    if models_response.status_code == 200:
                        models_data = models_response.json()
                        models = [m.get("id") for m in models_data.get("data", [])]
                    else:
                        models = []
                except:
                    models = []

                return {
                    "status": "healthy",
                    "message": "vLLM service is running",
                    "models": models,
                    "configured_model": settings.VLLM_MODEL_NAME,
                    "base_url": settings.VLLM_BASE_URL
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": f"vLLM service returned status {response.status_code}"
                }
    except httpx.ConnectError:
        return {
            "status": "unavailable",
            "message": "Cannot connect to vLLM service. Make sure vLLM server is running.",
            "hint": f"Run 'vllm serve {settings.VLLM_MODEL_NAME} --host 0.0.0.0 --port 8000' to start the service",
            "base_url": settings.VLLM_BASE_URL
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"vLLM health check failed: {str(e)}"
        }


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint

    Returns the health status of the application and its dependencies:
    - Application status
    - Database connection
    - Ollama service status
    - vLLM service status
    """
    # Check database
    db_status = await check_database()

    # Check Ollama
    ollama_status = await check_ollama()

    # Check vLLM
    vllm_status = await check_vllm()

    # Determine overall status (DB required, at least one LLM engine must be healthy)
    db_healthy = db_status["status"] == "healthy"
    llm_healthy = (
        ollama_status["status"] == "healthy" or
        vllm_status["status"] == "healthy"
    )

    all_healthy = db_healthy and llm_healthy
    overall_status = "healthy" if all_healthy else "degraded"

    # Determine which LLM engine is active
    active_engine = settings.LLM_ENGINE
    fallback_available = settings.ENABLE_LLM_FALLBACK

    response = {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "application": {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "running",
            "active_llm_engine": active_engine,
            "fallback_enabled": fallback_available
        },
        "services": {
            "database": db_status,
            "ollama": ollama_status,
            "vllm": vllm_status
        }
    }

    return response


@router.get("/health/live", status_code=status.HTTP_200_OK)
async def liveness_probe():
    """
    Liveness probe for Kubernetes/Docker
    Simple check to see if the application is alive
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/health/ready", status_code=status.HTTP_200_OK)
async def readiness_probe():
    """
    Readiness probe for Kubernetes/Docker
    Checks if the application is ready to serve traffic
    """
    # Check critical dependencies
    db_status = await check_database()

    if db_status["status"] != "healthy":
        return {
            "status": "not_ready",
            "timestamp": datetime.utcnow().isoformat(),
            "reason": "Database not available"
        }

    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat()
    }
