from fastapi import APIRouter, Header, HTTPException
from config.settings import settings
from services.hdm_ai_service import hdm_ai_service
from utils.validator import validate_request_api_key
from utils.response_formatter import success_response, error_response
from utils.logger import logger
import google.generativeai as genai

router = APIRouter()

SYSTEM_GUARD = """You are FarmVexa AI, an agricultural assistant for farmers.

You ONLY answer questions related to:
- Crop farming, livestock, poultry
- Crop diseases, pests, and treatment
- Soil health and management
- Irrigation and water management
- Fertilizers and nutrients
- Weather and climate impacts on farming
- Harvesting and post-harvest handling
- Farm equipment and tools
- Farm business and market tips
- Animal health, vaccination, breeding
- Farm management and operations

If a user asks anything outside agriculture (sports, politics, technology, entertainment, coding, etc.), respond ONLY with:
"I'm your FarmVexa agricultural assistant. I can only help with farming-related questions. Please ask me about crops, soil, livestock, pests, diseases, or any other farming topic."

Do not answer non-agricultural questions under any circumstances.
Keep responses practical, concise, and helpful for farmers.
Use simple language that farmers can understand.
If you don't know something, say so honestly."""


def get_chat_response_gemini(message: str, system_prompt: str = None):
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel("models/gemini-3.5-flash")
    
    full_prompt = SYSTEM_GUARD
    if system_prompt:
        full_prompt += "\n\n" + system_prompt
    full_prompt += f"\n\nFarmer's question: {message}"
    
    response = model.generate_content(full_prompt)
    return response.text


@router.post("/chat")
async def farmer_chat(data: dict, x_api_key: str = Header(None)):
    """FarmVexa farmer chat - agriculture questions with farm context."""
    try:
        await validate_request_api_key(x_api_key)
        
        message = data.get("message", "")
        system_prompt = data.get("systemPrompt", "")
        
        if not message:
            return error_response("Message is required", 400)
        
        if settings.AI_USED == "hdm":
            result = await hdm_ai_service.chat(message, system_prompt)
            reply = result.get("data", {}).get("reply", "No response")
        else:
            reply = get_chat_response_gemini(message, system_prompt)
        
        logger.info(f"Chat: '{message[:50]}...' → Answered via {settings.AI_USED}")
        return success_response({"reply": reply, "model": settings.AI_USED}, "Chat response")
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return error_response(str(e), 500)