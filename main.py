import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings
from routes import sensor_routes, crop_routes, health_routes, model_routes, chat_routes
from keep_alive import keep_alive
from utils.logger import logger

settings.validate()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"\x1b[36m🚀 FarmVexa AI Engine\x1b[0m → Port {settings.PORT} [\x1b[32m{settings.AI_USED}\x1b[0m]")
    logger.info(f"🚀 {settings.APP_NAME} started on port {settings.PORT}")
    logger.info(f"🤖 AI Mode: {settings.AI_USED}")
    logger.info(f"🔗 MERN Server: {settings.MERN_URL}")

    if settings.AI_USED == "local":
        if os.path.exists(settings.CROP_AI_MODEL):
            logger.info(f"✅ Model found: {settings.CROP_AI_MODEL}")
        else:
            logger.warning(f"⚠️  Model not found: {settings.CROP_AI_MODEL}")

    keep_alive.start()

    yield

    logger.info("👋 FarmVexa AI Engine shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    description="AI-Powered Farm Intelligence Engine",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.MERN_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_routes.router, prefix="/api", tags=["Health"])
app.include_router(sensor_routes.router, prefix="/api", tags=["Sensor Analysis"])
app.include_router(crop_routes.router, prefix="/api", tags=["Crop Analysis"])
app.include_router(model_routes.router, prefix="/api", tags=["Models"])
app.include_router(chat_routes.router, prefix="/api", tags=["Farmer Chat"])

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.TEMP_DIR, exist_ok=True)


@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "status": "running",
        "ai_mode": settings.AI_USED,
        "endpoints": {
            "health": "/api/health",
            "analyze_sensors": "/api/analyze/sensors",
            "analyze_crop": "/api/analyze/crop",
            "model_details": "/api/models/details",
            "train_model": "/api/models/train",
            "farmer_chat": "/api/chat"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=settings.PORT, reload=True)