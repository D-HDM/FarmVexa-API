import json
import os
from config.settings import settings
from utils.logger import logger


class RuleEngineService:
    def __init__(self):
        self.rules = {}
        self._load_rules()

    def _load_rules(self):
        """Load rule engine configuration."""
        path = settings.RULE_ENGINE_CONFIG
        if os.path.exists(path):
            with open(path, "r") as f:
                self.rules = json.load(f)
            logger.info(f"Rules loaded from {path}")
        else:
            logger.warning(f"Rule config not found: {path}. Using defaults.")
            self.rules = self._default_rules()

    def _default_rules(self):
        return {
            "thresholds": {
                "soil_moisture_low": 20,
                "soil_moisture_high": 80,
                "temperature_high": 35,
                "temperature_low": 5,
                "humidity_high": 85,
                "disease_risk_humidity": 75,
                "disease_risk_temperature": 28
            },
            "trend_rules": {
                "declining_days": 3,
                "rising_days": 3
            }
        }

    def analyze(self, current_readings: dict, historical_data: list = None):
        """Analyze sensor data against rules."""
        thresholds = self.rules.get("thresholds", {})
        alerts = []
        risk_score = 0

        temperature = current_readings.get("temperature")
        humidity = current_readings.get("humidity")
        soil_moisture = current_readings.get("soilMoisture")
        light_level = current_readings.get("lightLevel")

        # Soil moisture checks
        if soil_moisture is not None:
            low = thresholds.get("soil_moisture_low", 20)
            high = thresholds.get("soil_moisture_high", 80)
            if soil_moisture < low:
                alerts.append(f"Soil moisture critically low ({soil_moisture}%)")
                risk_score += 30
            elif soil_moisture > high:
                alerts.append(f"Soil moisture too high ({soil_moisture}%)")
                risk_score += 15

        # Temperature checks
        if temperature is not None:
            temp_high = thresholds.get("temperature_high", 35)
            temp_low = thresholds.get("temperature_low", 5)
            if temperature > temp_high:
                alerts.append(f"High temperature ({temperature}°C)")
                risk_score += 15
            elif temperature < temp_low:
                alerts.append(f"Low temperature ({temperature}°C)")
                risk_score += 15

        # Disease risk conditions
        if temperature and humidity:
            risk_temp = thresholds.get("disease_risk_temperature", 28)
            risk_hum = thresholds.get("disease_risk_humidity", 75)
            if temperature > risk_temp and humidity > risk_hum:
                alerts.append("Conditions favorable for disease spread")
                risk_score += 25

        # Trend analysis
        if historical_data:
            trend_alerts = self._analyze_trends(historical_data, thresholds)
            alerts.extend(trend_alerts)
            risk_score += len(trend_alerts) * 10

        # Determine risk level
        risk_level = "HIGH" if risk_score >= 50 else "MEDIUM" if risk_score >= 25 else "LOW"

        return {
            "risk_score": min(risk_score, 100),
            "risk_level": risk_level,
            "alerts": alerts,
            "recommendation": self._generate_recommendation(risk_level, alerts)
        }

    def _analyze_trends(self, historical_data: list, thresholds: dict):
        alerts = []
        if len(historical_data) < 3:
            return alerts

        trend_rules = self.rules.get("trend_rules", {})
        declining_days = trend_rules.get("declining_days", 3)

        recent_moisture = [d.get("soilMoisture") for d in historical_data[-declining_days:] if d.get("soilMoisture") is not None]
        if len(recent_moisture) >= declining_days:
            if all(recent_moisture[i] > recent_moisture[i + 1] for i in range(len(recent_moisture) - 1)):
                alerts.append(f"Soil moisture declining for {declining_days}+ days")

        return alerts

    def _generate_recommendation(self, risk_level: str, alerts: list):
        if risk_level == "HIGH":
            return "Immediate action required. Inspect crops and irrigation immediately."
        elif risk_level == "MEDIUM":
            return "Monitor conditions closely. Consider preventive measures."
        else:
            return "Farm conditions are normal. Continue regular monitoring."


rule_engine_service = RuleEngineService()