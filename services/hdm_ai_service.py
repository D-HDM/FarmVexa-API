import httpx
from config.settings import settings
from utils.logger import logger


class HdmAIService:
    def __init__(self):
        self.url = settings.HDM_AI_URL
        self.api_key = settings.HDM_AI_API_KEY

    async def chat(self, message: str, system_prompt: str = ""):
        """Send chat message to HDM AI."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.url,
                    json={
                        "message": message,
                        "system_prompt": system_prompt,
                    },
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"HDM AI chat: {data.get('data', {}).get('reply', '')[:50]}...")
                    return data
                else:
                    logger.error(f"HDM AI error: {response.status_code} - {response.text}")
                    return {
                        "success": False,
                        "data": {"reply": "AI service temporarily unavailable. Please try again."},
                    }

        except Exception as e:
            logger.error(f"HDM AI request failed: {str(e)}")
            return {
                "success": False,
                "data": {"reply": "AI service temporarily unavailable. Please try again later."},
            }


hdm_ai_service = HdmAIService()