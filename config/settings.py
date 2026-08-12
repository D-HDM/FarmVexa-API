import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Server
    PORT: int = int(os.getenv("PORT", 8000))
    APP_NAME: str = "FarmVexa AI Engine"
    APP_URL: str = os.getenv("APP_URL", "")
    KEEP_ALIVE_ENABLED: bool = os.getenv("KEEP_ALIVE_ENABLED", "false").lower() == "true"

    # API Key (Shared MERN ↔ Python)
    INTERNAL_API_KEY: str = os.getenv("INTERNAL_API_KEY", "")

    # MERN Server
    MERN_URL: str = os.getenv("MERN_URL", "http://localhost:5000")

    # AI Config
    AI_USED: str = os.getenv("AI_USED", "local")
    HDM_AI_API_KEY: str = os.getenv("HDM_AI_API_KEY", "")
    HDM_AI_URL: str = os.getenv("HDM_AI_URL", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Model Paths
    CROP_AI_MODEL: str = os.getenv("CROP_AI_MODEL", "models/crop_ai_model_v1.h5")
    RULE_ENGINE_CONFIG: str = os.getenv("RULE_ENGINE_CONFIG", "models/rule_engine_config.json")
    DISEASE_CLASSES: str = os.getenv("DISEASE_CLASSES", "models/disease_classes.json")
    CROP_TYPES: str = os.getenv("CROP_TYPES", "models/crop_types.json")

    # Computer Vision
    IMAGE_SIZE: int = int(os.getenv("IMAGE_SIZE", 224))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.75))

    # Uploads
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    TEMP_DIR: str = os.getenv("TEMP_DIR", "temp")

    @classmethod
    def validate(cls):
        if not cls.INTERNAL_API_KEY:
            raise ValueError("INTERNAL_API_KEY is required in .env")
        if cls.AI_USED == "gemini" and not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required when AI_USED=gemini")
        if cls.AI_USED == "hdm" and not cls.HDM_AI_API_KEY:
            raise ValueError("HDM_AI_API_KEY is required when AI_USED=hdm")
        return True


settings = Settings()