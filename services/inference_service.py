import numpy as np
from config.settings import settings
from config.model_config import model_config
from utils.image_processor import preprocess_image
from utils.logger import logger


class InferenceService:
    def __init__(self):
        self.model = None

    def _load_model(self):
        """Lazy load the TensorFlow model."""
        if self.model is None:
            import tensorflow as tf
            logger.info(f"Loading model: {settings.CROP_AI_MODEL}")
            self.model = tf.keras.models.load_model(settings.CROP_AI_MODEL)
            logger.info("Model loaded successfully")
        return self.model

    def predict(self, image_path: str, crop_type: str = None):
        """Run inference on a crop image."""
        model = self._load_model()

        img = preprocess_image(image_path)
        predictions = model.predict(img, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        diseases = model_config.get_diseases(crop_type) if crop_type else []
        disease_name = diseases[predicted_class] if predicted_class < len(diseases) else "Unknown"

        high_confidence = confidence >= settings.CONFIDENCE_THRESHOLD

        return {
            "disease": disease_name,
            "confidence": round(confidence * 100, 1),
            "high_confidence": high_confidence,
            "severity": self._estimate_severity(confidence),
            "raw_prediction": predictions[0].tolist()
        }

    def _estimate_severity(self, confidence: float):
        if confidence >= 0.85:
            return "high"
        elif confidence >= 0.70:
            return "moderate"
        else:
            return "low"


inference_service = InferenceService()