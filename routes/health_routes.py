import os
import platform
import psutil
from fastapi import APIRouter
from config.settings import settings
from utils.logger import logger

router = APIRouter()


@router.get("/health")
async def health_check():
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    total_mem = psutil.virtual_memory().total
    cpu_count = os.cpu_count()

    return {
        "success": True,
        "server": {
            "status": "running",
            "node": platform.python_version(),
            "platform": f"{platform.system()} ({platform.machine()})",
            "uptime": format_uptime(process.create_time()),
            "cpu": f"{process.cpu_percent():.2f}% ({cpu_count} cores)",
            "memory": f"{mem.rss / 1024 / 1024:.2f} MB / {total_mem / 1024 / 1024 / 1024:.2f} GB",
            "url": f"http://localhost:{settings.PORT}",
        },
        "ai": {
            "mode": settings.AI_USED,
            "model": settings.CROP_AI_MODEL if settings.AI_USED == "local" else "models/gemini-3.5-flash" if settings.AI_USED == "gemini" else "HDM AI",
            "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
            "model_exists": os.path.exists(settings.CROP_AI_MODEL),
        },
        "mern_server": {
            "url": settings.MERN_URL,
            "connected": check_mern_connection(),
        },
        "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
    }


def format_uptime(create_time):
    import time
    seconds = time.time() - create_time
    d = int(seconds // 86400)
    h = int((seconds % 86400) // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{d}d {h}h {m}m {s}s"


def check_mern_connection():
    import httpx
    try:
        response = httpx.get(f"{settings.MERN_URL}/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False