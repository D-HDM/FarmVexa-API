import json
import os
from config.settings import settings


class ModelConfig:
    def __init__(self):
        self.crop_types = self._load_json(settings.CROP_TYPES, ["tomato", "maize", "potato", "bean"])
        self.disease_classes = self._load_json(settings.DISEASE_CLASSES, {})

    def _load_json(self, path, default):
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return default

    def get_diseases(self, crop_type: str):
        return self.disease_classes.get(crop_type, self.disease_classes.get("default", []))

    def get_supported_crops(self):
        return self.crop_types.get("supported_crops", [])

    def is_crop_supported(self, crop_type: str):
        return crop_type in self.get_supported_crops()


model_config = ModelConfig()