from fastapi import UploadFile, HTTPException
from config.settings import settings


async def validate_request_api_key(api_key: str):
    """Check shared API key."""
    if api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key")


def validate_task(task: str):
    """Check if task type is valid."""
    valid_tasks = ["sensor_analysis", "image_analysis", "combined_analysis"]
    if task not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Invalid task. Must be one of: {valid_tasks}")


async def validate_image_file(file: UploadFile):
    """Validate uploaded image file."""
    if not file:
        raise HTTPException(status_code=400, detail="No image file provided")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > 10:
        raise HTTPException(status_code=400, detail=f"Image too large: {size_mb:.1f}MB (max 10MB)")

    await file.seek(0)
    return True