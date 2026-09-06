"""
ANPR (Automatic Number Plate Recognition) stage.

Triggered only when an "Accident" detection clears its confidence
threshold in detect_hazard.py — not run on every frame.

Two-step process:
1. Send the full frame to a Roboflow model trained to LOCATE plates
   (returns a bounding box, not text).
2. Crop that bounding box out of the frame and run PaddleOCR (via
   recognize_plate() in ocr.py) on the crop to actually READ the
   characters.
"""

import os
import io
import json
import uuid
import contextlib

import cv2
from roboflow import Roboflow

from config import (
    ANPR_MODEL_CONFIG, ANPR_DETECTION_CONFIDENCE, ANPR_OVERLAP,
    ANPR_OCR_MIN_CONFIDENCE, ANPR_CROP_PADDING_PX, PLATES_OUTPUT_JSON,
)
from ocr import recognize_plate

_anpr_model = None  # loaded once, reused across calls


def load_anpr_model():
    """Load the Roboflow ANPR (plate-location) model once."""
    global _anpr_model
    if _anpr_model is not None:
        return _anpr_model

    cfg = ANPR_MODEL_CONFIG
    if not cfg.get("api_key"):
        raise ValueError(
            "Missing ROBOFLOW_API_KEY_ANPR in your .env file — "
            "the ANPR model needs its own API key."
        )

    with contextlib.redirect_stdout(io.StringIO()):
        rf = Roboflow(api_key=cfg["api_key"])
        project = rf.workspace(cfg["workspace"]).project(cfg["project"])
        version = project.version(cfg["version"])
        available = version.models()

    if not available:
        raise RuntimeError(
            f"No trained ANPR model found at "
            f"{cfg['workspace']}/{cfg['project']}/v{cfg['version']}."
        )

    _anpr_model = available[0]
    return _anpr_model


def detect_plate_boxes(filepath):
    """Run the Roboflow ANPR model on the full frame, return a list of
    bounding boxes (Roboflow's raw prediction dicts: x, y, width, height,
    confidence, class)."""
    model = load_anpr_model()
    with contextlib.redirect_stdout(io.StringIO()):
        prediction = model.predict(
            filepath, confidence=ANPR_DETECTION_CONFIDENCE, overlap=ANPR_OVERLAP
        ).json()
    return prediction.get("predictions", [])


def crop_plate(image, box, padding=ANPR_CROP_PADDING_PX):
    """Crop the plate region out of the full frame, with a small pixel
    padding margin, since OCR tends to fail on tightly-cropped plates that
    clip characters right at the edge.

    box: Roboflow's prediction dict — x/y are the CENTER of the box,
    width/height are its full size.
    """
    img_h, img_w = image.shape[:2]

    cx, cy = box["x"], box["y"]
    bw, bh = box["width"], box["height"]

    x1 = max(0, int(cx - bw / 2) - padding)
    y1 = max(0, int(cy - bh / 2) - padding)
    x2 = min(img_w, int(cx + bw / 2) + padding)
    y2 = min(img_h, int(cy + bh / 2) + padding)

    return image[y1:y2, x1:x2]


def ocr_plate(crop):
    """Run PaddleOCR (via recognize_plate) on a cropped plate image.

    Returns (plate_text, ocr_confidence_pct). PaddleOCR's own confidence is
    a 0-1 float, converted here to a 0-100 percentage so it's directly
    comparable to ANPR_OCR_MIN_CONFIDENCE and every other confidence value
    used elsewhere in this pipeline.
    """
    result = recognize_plate(crop)
    plate_text = result["text"]
    confidence_pct = result["confidence"] * 100
    return plate_text, confidence_pct


def process_accident_frame(filepath, frame_meta, accident_event_id):
    """Full pipeline for one accident-flagged frame: detect plate boxes,
    crop + OCR each one, return a list of plate reading dicts (empty list
    if no plate was found or OCR confidence was too low to trust).

    frame_meta: the manifest entry for this frame (timestamp, location, etc.)
    accident_event_id: links this plate reading back to the accident event
    that triggered it, for cross-referencing later.
    """
    boxes = detect_plate_boxes(filepath)
    if not boxes:
        return []

    image = cv2.imread(filepath)
    if image is None:
        print(f"Could not read image for ANPR: {filepath}")
        return []

    results = []
    for box in boxes:
        crop = crop_plate(image, box)
        if crop.size == 0:
            continue

        plate_text, ocr_confidence = ocr_plate(crop)

        if not plate_text or ocr_confidence < ANPR_OCR_MIN_CONFIDENCE:
            continue  # too unreliable to record

        results.append({
            "plate_id": str(uuid.uuid4()),
            "accident_event_id": accident_event_id,
            "plate_number": plate_text,
            "plate_detection_confidence": round(box["confidence"] * 100, 1),
            "ocr_confidence": round(ocr_confidence, 1),
            "timestamp": frame_meta.get("timestamp"),
            "location": frame_meta.get("location"),
            "frame_unique_id": frame_meta.get("unique_id"),
            "source_frame": os.path.basename(filepath),
        })

        print(f"🔍 Plate read: {plate_text} (OCR conf: {ocr_confidence:.0f}%) "
              f"near accident {accident_event_id[:8]}...")

    return results


def save_plates(all_plate_results, output_json=PLATES_OUTPUT_JSON):
    with open(output_json, "w") as f:
        json.dump(all_plate_results, f, indent=2)
    print(f"Saved {len(all_plate_results)} plate reading(s) -> {output_json}")
