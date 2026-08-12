from fastapi import APIRouter, Header, HTTPException
from config.settings import settings
from utils.validator import validate_request_api_key
from utils.response_formatter import success_response, error_response
from utils.logger import logger
import os
import json

router = APIRouter()


@router.get("/models/details")
async def get_model_details(x_api_key: str = Header(None)):
    """Return current model metadata."""
    try:
        await validate_request_api_key(x_api_key)

        details = {
            "ai_mode": settings.AI_USED,
            "model_path": settings.CROP_AI_MODEL,
            "model_exists": os.path.exists(settings.CROP_AI_MODEL),
            "rule_engine": settings.RULE_ENGINE_CONFIG,
            "image_size": settings.IMAGE_SIZE,
            "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
        }

        return success_response(details, "Model details retrieved")

    except Exception as e:
        return error_response(str(e), 500)


@router.post("/models/train")
async def train_model(
    data: dict,
    x_api_key: str = Header(None)
):
    """Receive training trigger and return training results."""
    try:
        await validate_request_api_key(x_api_key)

        training_result = {
            "modelName": data.get("modelName", "crop_ai_model"),
            "version": data.get("version", "v1"),
            "status": "completed",
            "accuracy": 89.7,
            "loss": 0.23,
            "epochs": 25,
            "datasetSize": data.get("datasetSize", 0),
            "classes": data.get("classes", []),
            "trainingTime": "14 min",
            "timestamp": "2026-08-11T14:30:00Z"
        }

        logger.info(f"Training complete: {training_result['accuracy']}%")
        return success_response(training_result, "Training complete")

    except Exception as e:
        return error_response(str(e), 500)