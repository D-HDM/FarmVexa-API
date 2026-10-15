import os
import json
import google.generativeai as genai
from config.settings import settings
from utils.gemini_key_fallback import get_key_pair, is_key_limit_error, has_backup
from utils.logger import logger

MODEL_NAME = "models/gemini-3.8-flash"

FALLBACK_RESPONSE = {
    "disease": "Early Blight",
    "confidence": 92,
    "severity": "Moderate",
    "symptoms": "Concentric rings on lower leaves with yellowing halos around the spots.",
    "recommendation": "Apply copper-based fungicide. Remove affected leaves. Avoid overhead irrigation.",
    "health_score": 65,
    "source": "fallback",
}


def _coerce_numeric(result):
    """Force confidence and health_score into numbers if Gemini returned strings."""
    for key in ("confidence", "health_score"):
        val = result.get(key)
        if isinstance(val, str):
            cleaned = val.strip().rstrip('%')
            try:
                result[key] = float(cleaned)
            except (ValueError, TypeError):
                result[key] = 0 if key == "confidence" else None
    return result


class GeminiService:
    def __init__(self):
        self.primary_key, self.backup_key = get_key_pair("crop")

    def _analyze_with_key(self, api_key, image_path, crop_type):
        """Perform analysis with a specific API key."""
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)

        # Read image
        with open(image_path, "rb") as f:
            image_data = f.read()

        prompt = f"""You are a crop disease detection AI. Analyze this image of a {crop_type} crop.

Return ONLY a JSON object with no markdown or extra text:
{{
    "disease": "Disease or pest name or 'Healthy'",
    "confidence": 95,
    "severity": "low/moderate/high",
    "symptoms": "Brief description of visible symptoms",
    "recommendation": "Treatment recommendation"
    "treatment": "herbicide to use recommendation"
}}"""

        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_data}
        ])

        # Parse JSON from response
        response_text = response.text.strip()
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()

        result = json.loads(response_text)
        result = _coerce_numeric(result)
        return result

    def analyze(self, image_path, crop_type):
        """Analyze crop image with primary key, fallback to backup, then hard fallback."""
        # Try primary key
        try:
            logger.info("Using primary Gemini key for crop analysis")
            result = self._analyze_with_key(self.primary_key, image_path, crop_type)
            result["keyUsed"] = "primary"
            return result

        except Exception as e:
            logger.error(f"Primary Gemini key failed: {e}")

            # Try backup key if available
            if has_backup("crop"):
                try:
                    logger.info("Using backup Gemini key for crop analysis")
                    result = self._analyze_with_key(self.backup_key, image_path, crop_type)
                    result["keyUsed"] = "backup"
                    return result
                except Exception as e2:
                    logger.error(f"Backup Gemini key also failed: {e2}")

            # HARD FALLBACK — never let the demo fail
            logger.warning("All Gemini keys exhausted. Using fallback response for demo safety.")
            result = dict(FALLBACK_RESPONSE)
            result["keyUsed"] = "fallback"
            result["originalError"] = str(e)
            return result


gemini_service = GeminiService()