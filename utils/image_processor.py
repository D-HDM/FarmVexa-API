import cv2
import numpy as np
from PIL import Image
from config.settings import settings


def preprocess_image(image_path: str):
    """Load, resize, normalize image for model prediction."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (settings.IMAGE_SIZE, settings.IMAGE_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def validate_image(file_bytes: bytes):
    """Check if uploaded file is a valid image."""
    try:
        img = Image.open(file_bytes)
        img.verify()
        return True
    except Exception:
        return False


def get_image_size_mb(file_bytes: bytes):
    """Return file size in MB."""
    return len(file_bytes) / (1024 * 1024)