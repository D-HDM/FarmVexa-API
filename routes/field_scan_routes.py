import os
import json
import uuid
import time
import cv2
import numpy as np
import requests
import google.generativeai as genai
from fastapi import APIRouter, Header, HTTPException
from config.settings import settings
from utils.gemini_key_fallback import get_key_pair, is_key_limit_error, has_backup
from utils.validator import validate_request_api_key
from utils.response_formatter import success_response, error_response
from utils.logger import logger

router = APIRouter()

BATCH_SIZE = 16  # Gemini max images per request


class FieldScanService:
    def __init__(self):
        self.primary_key, self.backup_key = get_key_pair("fieldscan")

    def _download_image(self, image_url):
        """Download image from URL."""
        response = requests.get(image_url, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Failed to download frame: {response.status_code}")
        return response.content

    def _prefilter_frame(self, image_data):
        """Score a frame using OpenCV. Returns (should_skip, score, reason)."""
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return True, 0, "invalid"

            # 1. Blur detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < 50:
                return True, 0, "blurry"

            # 2. Green coverage (healthy vegetation)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
            green_percentage = (np.sum(green_mask > 0) / green_mask.size) * 100

            # 3. Yellow/brown detection (possible disease)
            yellow_mask = cv2.inRange(hsv, (20, 40, 40), (35, 255, 255))
            brown_mask = cv2.inRange(hsv, (10, 40, 40), (20, 255, 200))
            disease_mask = cv2.bitwise_or(yellow_mask, brown_mask)
            disease_percentage = (np.sum(disease_mask > 0) / disease_mask.size) * 100

            # Score: higher = more suspicious
            # Disease spots increase score, high green coverage decreases
            score = disease_percentage * 10 - (green_percentage - 50) * 0.5
            score = max(0, score)

            # Skip if very healthy (high green, low disease)
            if disease_percentage < 2 and green_percentage > 70:
                return True, score, "healthy"

            # Skip if no vegetation at all
            if green_percentage < 10:
                return True, score, "no_vegetation"

            return False, score, None

        except Exception as e:
            logger.warning(f"Pre-filter error: {e}")
            return False, 0, None  # Don't skip if pre-filter fails

    def _analyze_batch_with_key(self, api_key, frames_data, crop_type):
        """Analyze a batch of frames (max 16) with a specific API key."""
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("models/gemini-3.5-flash")

        prompt = f"""You are a crop field scan AI. Analyze {len(frames_data)} photos from a {crop_type} field.

For each photo, return:
1. disease: Disease name or "Healthy"
2. confidence: 0-100
3. severity: low/moderate/high
4. symptoms: Brief description
5. recommendation: Treatment advice
6. weeds: true/false
7. pests: true/false
8. healthScore: 0-100

Return ONLY a JSON array (no markdown) with {len(frames_data)} objects:
[
    {{"photoIndex": 0, "disease": "...", "confidence": 95, "severity": "moderate", "symptoms": "...", "recommendation": "...", "weeds": false, "pests": false, "healthScore": 65}},
    ...
]"""

        content = [prompt] + [
            {"mime_type": "image/jpeg", "data": img_data} for img_data in frames_data
        ]

        response = model.generate_content(content)
        response_text = response.text.strip()
        if response_text.startswith("```"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()

        return json.loads(response_text)

    def analyze_batch(self, frames, crop_type):
        """Analyze batch with primary key, fallback to backup. Returns (results, key_used)."""
        try:
            logger.info(f"[Gemini Batch] Using primary FieldScan key — {len(frames)} frames")
            results = self._analyze_batch_with_key(self.primary_key, frames, crop_type)
            return results, "fieldscan_primary"
        except Exception as e:
            if is_key_limit_error(e) and has_backup("fieldscan"):
                logger.warning(f"Primary FieldScan key failed: {e}. Trying backup...")
                try:
                    logger.info(f"[Gemini Batch] Using backup FieldScan key — {len(frames)} frames")
                    results = self._analyze_batch_with_key(self.backup_key, frames, crop_type)
                    return results, "fieldscan_backup"
                except Exception as e2:
                    logger.error(f"Backup FieldScan key also failed: {e2}")
                    raise Exception(f"All FieldScan Gemini API keys exhausted: {e2}")
            else:
                raise e


field_scan_service = FieldScanService()


@router.post("/analyze/field-scan")
async def analyze_field_scan(data: dict, x_api_key: str = Header(None)):
    """Field scan analysis — batch of frames with GPS + pre-filter + batching."""
    try:
        await validate_request_api_key(x_api_key)

        field_id = data.get("fieldId", "")
        crop_type = data.get("cropType", "")
        frames = data.get("frames", [])
        pre_filter_enabled = data.get("preFilterEnabled", True)
        pre_filter_percentage = data.get("preFilterPercentage", 60)  # Default 60%

        if not field_id:
            return error_response("fieldId is required", 400)
        if not crop_type:
            return error_response("cropType is required", 400)
        if not frames or len(frames) == 0:
            return error_response("frames are required", 400)

        start_time = time.time()
        total_frames = len(frames)
        logger.info(f"[Field Scan] Received: {total_frames} frames | field={field_id} | crop={crop_type}")

        # === STEP 1: Download all frames + pre-filter ===
        skip_reasons = {"blurry": 0, "duplicate": 0, "healthy": 0, "no_vegetation": 0, "invalid": 0, "download_failed": 0}

        scored_frames = []
        skipped_frames = 0

        for i, frame in enumerate(frames):
            image_url = frame.get("imageUrl", "")
            if not image_url:
                skipped_frames += 1
                skip_reasons["invalid"] += 1
                continue

            try:
                # Download image
                image_data = field_scan_service._download_image(image_url)

                # Pre-filter
                if pre_filter_enabled:
                    should_skip, score, reason = field_scan_service._prefilter_frame(image_data)
                    if should_skip:
                        skipped_frames += 1
                        if reason in skip_reasons:
                            skip_reasons[reason] += 1
                        continue
                else:
                    score = 0

                scored_frames.append({
                    "frame": frame,
                    "image_data": image_data,
                    "score": score,
                })

            except Exception as e:
                skipped_frames += 1
                skip_reasons["download_failed"] += 1
                logger.warning(f"[Field Scan] Frame {i+1} download failed: {e}")

        logger.info(f"[Pre-filter] Received: {total_frames} | Kept: {len(scored_frames)} | Skipped: {skipped_frames}")
        logger.info(f"[Pre-filter] Skip reasons: {skip_reasons}")

        # === STEP 2: Sort by score (suspicious first) and take top X% ===
        if pre_filter_enabled and scored_frames:
            scored_frames.sort(key=lambda x: x["score"], reverse=True)
            keep_count = max(1, int(len(scored_frames) * (pre_filter_percentage / 100)))
            filtered_out = len(scored_frames) - keep_count
            scored_frames = scored_frames[:keep_count]
            skipped_frames += filtered_out
            logger.info(f"[Pre-filter] Top {pre_filter_percentage}% selected: {keep_count} frames for Gemini (dropped {filtered_out})")

        selected_frames = scored_frames

        # === STEP 3: Batch and send to Gemini ===
        if len(selected_frames) == 0:
            logger.warning("[Field Scan] No frames to analyze after pre-filter")
            return success_response({
                "totalFrames": total_frames,
                "analyzedFrames": 0,
                "skippedFrames": skipped_frames,
                "geminiRequests": 0,
                "results": [],
                "summary": {
                    "healthyCount": 0,
                    "healthyPercentage": 0,
                    "diseaseCount": 0,
                    "diseases": [],
                    "weeds": {"pressure": "Low", "hotspots": []},
                    "pests": {"activity": "None", "affectedAreas": 0},
                },
                "keyUsage": {"fieldscan_primary": 0, "fieldscan_backup": 0},
                "skipReasons": skip_reasons,
            }, "Field scan complete (no frames to analyze)")

        # Split into batches of 16
        batches = [selected_frames[i:i + BATCH_SIZE] for i in range(0, len(selected_frames), BATCH_SIZE)]
        total_batches = len(batches)
        logger.info(f"[Gemini Batch] Total batches: {total_batches} | Batch size: {BATCH_SIZE}")

        all_results = []
        keys_used = []

        for batch_idx, batch in enumerate(batches):
            logger.info(f"[Gemini Batch] Request {batch_idx + 1}/{total_batches} — analyzing {len(batch)} frames...")

            batch_start = time.time()
            batch_frame_data = [item["image_data"] for item in batch]
            batch_frames = [item["frame"] for item in batch]

            try:
                batch_results, key_used = field_scan_service.analyze_batch(batch_frame_data, crop_type)
                keys_used.append(key_used)
                batch_duration = time.time() - batch_start

                # Merge results with frame GPS data
                for result in batch_results:
                    photo_index = result.get("photoIndex", 0)
                    if photo_index < len(batch_frames):
                        frame_info = batch_frames[photo_index]
                        all_results.append({
                            "imageUrl": frame_info.get("imageUrl", ""),
                            "lat": frame_info.get("lat"),
                            "lng": frame_info.get("lng"),
                            "timestamp": frame_info.get("timestamp", ""),
                            "analysis": result,
                            "keyUsed": key_used,
                        })

                logger.info(f"[Gemini Batch] Request {batch_idx + 1}/{total_batches} — done ({batch_duration:.1f}s)")

            except Exception as e:
                logger.error(f"[Gemini Batch] Request {batch_idx + 1}/{total_batches} — FAILED: {e}")
                # Continue with other batches
                continue

        # === STEP 4: Build summary ===
        diseases = []
        healthy_count = 0
        disease_count = 0
        weed_hotspots = []
        pest_areas = 0

        for r in all_results:
            analysis = r.get("analysis", {})
            disease = analysis.get("disease", "Healthy")

            if disease == "Healthy":
                healthy_count += 1
            else:
                disease_count += 1
                diseases.append({
                    "name": disease,
                    "severity": analysis.get("severity", "low"),
                    "location": {"lat": r.get("lat"), "lng": r.get("lng")},
                })

            if analysis.get("weeds"):
                weed_hotspots.append({
                    "lat": r.get("lat"),
                    "lng": r.get("lng"),
                    "type": "Weeds",
                })

            if analysis.get("pests"):
                pest_areas += 1

        healthy_percentage = (healthy_count / len(all_results) * 100) if all_results else 0

        summary = {
            "healthyCount": healthy_count,
            "healthyPercentage": round(healthy_percentage, 1),
            "diseaseCount": disease_count,
            "diseases": diseases,
            "weeds": {
                "pressure": "Low" if len(weed_hotspots) < 3 else "Moderate" if len(weed_hotspots) < 10 else "High",
                "hotspots": weed_hotspots,
            },
            "pests": {
                "activity": "None" if pest_areas == 0 else "Low" if pest_areas < 3 else "Moderate" if pest_areas < 10 else "High",
                "affectedAreas": pest_areas,
            },
        }

        total_duration = time.time() - start_time

        logger.info(f"[Field Scan] Complete: {len(all_results)} analyzed | {skipped_frames} skipped | {len(batches)} Gemini requests | {total_duration:.0f}s total")

        return success_response({
            "totalFrames": total_frames,
            "preFilteredFrames": len(selected_frames),
            "skippedFrames": skipped_frames,
            "analyzedFrames": len(all_results),
            "geminiRequests": len(batches),
            "batchSize": BATCH_SIZE,
            "duration": round(total_duration, 1),
            "skipReasons": skip_reasons,
            "results": all_results,
            "summary": summary,
            "keyUsage": {
                "fieldscan_primary": keys_used.count("fieldscan_primary"),
                "fieldscan_backup": keys_used.count("fieldscan_backup"),
            },
        }, "Field scan complete")

    except Exception as e:
        logger.error(f"Field scan error: {str(e)}")
        return error_response(str(e), 500)