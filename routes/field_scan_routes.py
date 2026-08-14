import os
import json
import uuid
from fastapi import APIRouter, Header, HTTPException
from config.settings import settings
from utils.gemini_key_fallback import get_key_pair, is_key_limit_error, has_backup
from utils.validator import validate_request_api_key
from utils.response_formatter import success_response, error_response
from utils.logger import logger
import google.generativeai as genai
import requests

router = APIRouter()


class FieldScanService:
    def __init__(self):
        self.primary_key, self.backup_key = get_key_pair("fieldscan")

    def _analyze_frame_with_key(self, api_key, image_url, crop_type):
        """Analyze a single field scan frame with a specific API key."""
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("models/gemini-3.5-flash")

        prompt = f"""You are a crop field scan AI. Analyze this frame from a {crop_type} field.

Tasks:
1. Is there any disease visible? Name it or say "Healthy".
2. Are there weeds visible?
3. Are there pests visible?
4. Give a health score (0-100).

Return ONLY a JSON object with no markdown or extra text:
{{
    "disease": "Disease name or 'Healthy'",
    "confidence": 95,
    "severity": "low/moderate/high",
    "symptoms": "Brief description of visible symptoms",
    "recommendation": "Treatment recommendation",
    "weeds": true/false,
    "pests": true/false,
    "healthScore": 65
}}"""

        # Download image from URL
        response = requests.get(image_url, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Failed to download frame: {response.status_code}")

        image_data = response.content

        result = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_data}
        ])

        response_text = result.text.strip()
        if response_text.startswith("```"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()

        return json.loads(response_text)

    def analyze_frame(self, image_url, crop_type):
        """Analyze frame with primary key, fallback to backup on limit error. Returns (result, key_used)."""
        # Try primary key
        try:
            logger.info("Using primary FieldScan Gemini key")
            result = self._analyze_frame_with_key(self.primary_key, image_url, crop_type)
            return result, "fieldscan_primary"

        except Exception as e:
            if is_key_limit_error(e) and has_backup("fieldscan"):
                logger.warning(f"Primary FieldScan Gemini key failed: {e}. Trying backup...")
                # Try backup key
                try:
                    logger.info("Using backup FieldScan Gemini key")
                    result = self._analyze_frame_with_key(self.backup_key, image_url, crop_type)
                    return result, "fieldscan_backup"
                except Exception as e2:
                    logger.error(f"Backup FieldScan Gemini key also failed: {e2}")
                    raise Exception(f"All FieldScan Gemini API keys exhausted: {e2}")
            else:
                logger.error(f"FieldScan Gemini analysis failed: {e}")
                raise e


field_scan_service = FieldScanService()


@router.post("/analyze/field-scan")
async def analyze_field_scan(data: dict, x_api_key: str = Header(None)):
    """Field scan analysis - batch of frames with GPS tags."""
    try:
        await validate_request_api_key(x_api_key)

        field_id = data.get("fieldId", "")
        crop_type = data.get("cropType", "")
        frames = data.get("frames", [])
        max_gemini_calls = data.get("maxGeminiCalls", 30)
        pre_filter_enabled = data.get("preFilterEnabled", True)

        if not field_id:
            return error_response("fieldId is required", 400)
        if not crop_type:
            return error_response("cropType is required", 400)
        if not frames or len(frames) == 0:
            return error_response("frames are required", 400)

        logger.info(f"Field scan: field={field_id}, crop={crop_type}, frames={len(frames)}, maxGeminiCalls={max_gemini_calls}")

        results = []
        analyzed_count = 0
        skipped_count = 0
        total_frames = len(frames)
        keys_used = []

        for frame in frames:
            image_url = frame.get("imageUrl", "")
            lat = frame.get("lat")
            lng = frame.get("lng")
            timestamp = frame.get("timestamp", "")

            if not image_url:
                skipped_count += 1
                continue

            if analyzed_count >= max_gemini_calls:
                skipped_count += 1
                continue

            try:
                analysis, key_used = field_scan_service.analyze_frame(image_url, crop_type)
                analyzed_count += 1
                keys_used.append(key_used)

                results.append({
                    "imageUrl": image_url,
                    "lat": lat,
                    "lng": lng,
                    "timestamp": timestamp,
                    "analysis": analysis,
                    "keyUsed": key_used,
                })

            except Exception as e:
                logger.error(f"Frame analysis failed: {e}")
                skipped_count += 1
                continue

        # Build summary
        diseases = []
        healthy_count = 0
        disease_count = 0
        weed_hotspots = []
        pest_areas = 0

        for r in results:
            analysis = r.get("analysis", {})
            disease = analysis.get("disease", "Healthy")

            if disease == "Healthy":
                healthy_count += 1
            else:
                disease_count += 1
                diseases.append({
                    "name": disease,
                    "severity": analysis.get("severity", "low"),
                    "location": {"lat": r.get("lat"), "lng": r.get("lng")},
                })

            if analysis.get("weeds"):
                weed_hotspots.append({
                    "lat": r.get("lat"),
                    "lng": r.get("lng"),
                    "type": "Weeds",
                })

            if analysis.get("pests"):
                pest_areas += 1

        healthy_percentage = (healthy_count / len(results) * 100) if results else 0

        summary = {
            "healthyCount": healthy_count,
            "healthyPercentage": round(healthy_percentage, 1),
            "diseaseCount": disease_count,
            "diseases": diseases,
            "weeds": {
                "pressure": "Low" if len(weed_hotspots) < 3 else "Moderate" if len(weed_hotspots) < 10 else "High",
                "hotspots": weed_hotspots,
            },
            "pests": {
                "activity": "None" if pest_areas == 0 else "Low" if pest_areas < 3 else "Moderate" if pest_areas < 10 else "High",
                "affectedAreas": pest_areas,
            },
        }

        # Key usage summary
        key_summary = {
            "fieldscan_primary": keys_used.count("fieldscan_primary"),
            "fieldscan_backup": keys_used.count("fieldscan_backup"),
        }

        return success_response({
            "totalFrames": total_frames,
            "analyzedFrames": analyzed_count,
            "skippedFrames": skipped_count,
            "results": results,
            "summary": summary,
            "keyUsage": key_summary,
        }, "Field scan complete")

    except Exception as e:
        logger.error(f"Field scan error: {str(e)}")
        return error_response(str(e), 500)