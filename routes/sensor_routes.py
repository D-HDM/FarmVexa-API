from fastapi import APIRouter, Header, HTTPException
from services.rule_engine_service import rule_engine_service
from utils.validator import validate_request_api_key, validate_task
from utils.response_formatter import success_response, error_response
from utils.logger import logger

router = APIRouter()


@router.post("/analyze/sensors")
async def analyze_sensors(
    data: dict,
    x_api_key: str = Header(None)
):
    try:
        await validate_request_api_key(x_api_key)
        task = data.get("task", "sensor_analysis")
        validate_task(task)

        readings = data.get("readings", {})
        historical = data.get("historicalData", [])

        result = rule_engine_service.analyze(readings, historical)

        logger.info(f"Sensor analysis: Risk={result['risk_level']}, Score={result['risk_score']}")
        return success_response(result, "Sensor analysis complete")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sensor analysis error: {str(e)}")
        return error_response(str(e), 500)