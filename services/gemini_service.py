import base64
import json
import re
from config.settings import settings
from utils.logger import logger


class GeminiService:
    def __init__(self):
        self.client = None
        self.model_name = "models/gemini-3.5-flash"

    def _get_client(self):
        if self.client is None:
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.client = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini client initialized: {self.model_name}")
        return self.client

    def analyze(self, image_path: str, crop_type: str = None):
        """Analyze crop image using Gemini Vision."""
        model = self._get_client()

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        prompt = f"""You are a crop disease expert. Analyze this {crop_type or 'plant'} leaf image.
Identify any disease, pest damage, nutrient deficiency, or abnormalities.

Return ONLY a JSON object with these fields:
- disease: disease name or "Healthy"
- confidence: 0-100 percentage
- severity: "low", "moderate", or "high"
- symptoms: brief description of visible symptoms
- recommendation: short actionable advice for farmer

If the leaf is healthy, set disease to "Healthy"."""

        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_data}
        ])

        result = self._parse_response(response.text)
        logger.info(f"Gemini analysis: {result.get('disease')} ({result.get('confidence')}%)")
        return result

    def _parse_response(self, text: str):
        """Extract JSON from Gemini response."""
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {
            "disease": "Unknown",
            "confidence": 0,
            "severity": "unknown",
            "symptoms": "Could not parse AI response",
            "recommendation": "Please retry analysis"
        }


gemini_service = GeminiService()