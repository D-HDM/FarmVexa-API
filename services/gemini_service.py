import os
import json
import google.generativeai as genai
from config.settings import settings
from utils.gemini_key_fallback import get_key_pair, is_key_limit_error, has_backup
from utils.logger import logger


class GeminiService:
    def __init__(self):
        self.primary_key, self.backup_key = get_key_pair("crop")

    def _analyze_with_key(self, api_key, image_path, crop_type):
        """Perform analysis with a specific API key."""
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("models/gemini-3.5-flash")
        
        # Read image
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        prompt = f"""You are a crop disease detection AI. Analyze this image of a {crop_type} crop.

Return ONLY a JSON object with no markdown or extra text:
{{
    "disease": "Disease name or 'Healthy'",
    "confidence": 95,
    "severity": "low/moderate/high",
    "symptoms": "Brief description of visible symptoms",
    "recommendation": "Treatment recommendation"
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
        return result

    def analyze(self, image_path, crop_type):
        """Analyze crop image with primary key, fallback to backup on limit error."""
        # Try primary key
        try:
            logger.info("Using primary Gemini key for crop analysis")
            result = self._analyze_with_key(self.primary_key, image_path, crop_type)
            result["keyUsed"] = "primary"
            return result
        
        except Exception as e:
            if is_key_limit_error(e) and has_backup("crop"):
                logger.warning(f"Primary Gemini key failed: {e}. Trying backup...")
                # Try backup key
                try:
                    logger.info("Using backup Gemini key for crop analysis")
                    result = self._analyze_with_key(self.backup_key, image_path, crop_type)
                    result["keyUsed"] = "backup"
                    return result
                except Exception as e2:
                    logger.error(f"Backup Gemini key also failed: {e2}")
                    raise Exception(f"All Gemini API keys exhausted: {e2}")
            else:
                logger.error(f"Gemini analysis failed: {e}")
                raise e


gemini_service = GeminiService()