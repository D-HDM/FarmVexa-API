import os
import json
import uuid
from fastapi import APIRouter, File, UploadFile, Form, Header, HTTPException
from config.settings import settings
from config.model_config import model_config
from services.gemini_service import gemini_service
from services.rule_engine_service import rule_engine_service
from utils.validator import validate_request_api_key, validate_task, validate_image_file
from utils.response_formatter import success_response, error_response
from utils.logger import logger

router = APIRouter()


@router.post("/analyze/crop")
async def analyze_crop(
    cropImage: UploadFile = File(...),
    task: str = Form("image_analysis"),
    cropType: str = Form(None),
    fieldId: str = Form(None),
    sensorData: str = Form(None),
    x_api_key: str = Header(None)
):
    try:
        await validate_request_api_key(x_api_key)
        validate_task(task)
        await validate_image_file(cropImage)

        os.makedirs(settings.TEMP_DIR, exist_ok=True)
        file_ext = os.path.splitext(cropImage.filename or "image.jpg")[1] or ".jpg"
        temp_filename = f"{uuid.uuid4()}{file_ext}"
        temp_path = os.path.join(settings.TEMP_DIR, temp_filename)

        contents = await cropImage.read()
        with open(temp_path, "wb") as f:
            f.write(contents)

        # Crop images always use Gemini
        image_result = gemini_service.analyze(temp_path, cropType)

        combined_result = None
        if task == "combined_analysis" and sensorData:
            sensor_info = json.loads(sensorData)
            combined_result = rule_engine_service.analyze(sensor_info)
            combined_result["image_analysis"] = image_result
            combined_result["health_score"] = max(0, 100 - combined_result.get("risk_score", 0))
            logger.info(f"Combined analysis: Health={combined_result['health_score']}%")
        else:
            combined_result = {
                "image_analysis": image_result,
                "health_score": None
            }

        os.remove(temp_path)

        return success_response(combined_result, "Crop analysis complete")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Crop analysis error: {str(e)}")
        return error_response(str(e), 500)