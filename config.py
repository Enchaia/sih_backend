"""
Central configuration for the road-hazard detection pipeline.

Everything under "unchanged" below is exactly what you already had in
extract.py / detect_hazard.py — same values, just moved here so both files
share one source of truth instead of duplicating numbers.

Everything under "new" is a starting-point threshold I picked using common
values for these techniques. I don't have your actual footage to tune
against, so treat these as a first pass and adjust after watching how they
behave on your 3-4 test videos.
"""

# Fill in the workspace + project slugs for your two trained models shown in
# your Roboflow dashboard:
#   - "debr-road-water-pot-traffic-1-yolo11n-t1"  -> the combined defect model
#   - "traffic-light-7auyu-wdg0w-1-yolo11n-t1"    -> the traffic-light model
# The dashboard "Name"/"ID" columns are display names, not necessarily the

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file and loads its values into environment variables/URL for the slug.
MODEL_CONFIGS = [
    {"name": "mix_class", "api_key": os.environ.get("ROBOFLOW_API_KEY_1"), "workspace": "sanskar-s-workspace", "project": "debr-road-water-pot-traffic", "version": 1},
    {"name": "traffic_light", "api_key": os.environ.get("ROBOFLOW_API_KEY_2"), "workspace": "sibika", "project": "traffic-light-detection-yo9o4-3o2hz", "version": 2},
]

ANPR_MODEL_CONFIG = {
    "name": "anpr_plate_detector",
    "api_key": os.environ.get("ROBOFLOW_API_KEY_ANPR"),
    "workspace": "lethargic-wanderer",   # ← from your ANPR project's URL
    "project": "anpr-model",       # ← from your ANPR project's URL
    "version": 1,                          # ← from your ANPR project's URL
}

OVERLAP = 30
MAX_WORKERS = 8  # total parallel API calls across BOTH models combined

# Roboflow's API still needs ONE confidence value per request (it's a
# request parameter, not per-class), so we send a low floor to the API to
# make sure we don't lose anything, then filter more strictly per-class
# afterward using this table. Add/adjust class names to match exactly what
# your models output (check a sample prediction's "class" field if unsure).
API_REQUEST_CONFIDENCE = 25  # low floor sent to Roboflow itself — filtering happens locally after
 
CLASS_CONFIDENCE_THRESHOLDS = {
    "Pothole": 65,
    "water_logging": 60,
    "debris": 55,
    "garbage": 60,
    "garbage-overflow": 60,
    "Accident": 40,
    "traffic_light": 60,
}
DEFAULT_CLASS_CONFIDENCE = 50 

ANPR_DETECTION_CONFIDENCE = 40   # confidence floor sent to Roboflow for plate LOCATION
ANPR_OVERLAP = 30                # same idea as OVERLAP, but for the ANPR model's own request
ANPR_OCR_MIN_CONFIDENCE = 60     # min PaddleOCR confidence (0-100 pct) to trust a plate reading
ANPR_CROP_PADDING_PX = 5         # pixels of padding added around the plate crop before OCR

# Optional: relax every threshold slightly at night, since low light makes
# even real detections score lower confidence than the same object in
# daylight. Subtracted from the class threshold when timestamp hour is
# within NIGHT_HOURS. Tune this after watching real night footage.

NIGHT_HOURS = range(19, 6)  # 7 PM to 6 AM, wraps past midnight — handled in code, not by this range directly
NIGHT_CONFIDENCE_RELAXATION = 10
 
# ---- Driver alert signs (new) ----
# These aren't "hazards" in the pothole/accident sense — they're regulatory
# signs the driver needs to be actively reminded of the moment they're seen.
# Kept in a separate table from CLASS_CONFIDENCE_THRESHOLDS so their alert
# logic (immediate audio/visual cue) can be handled differently from
# hazard logging/dedup.

DRIVER_ALERT_CLASSES = {
    "Stop": {"confidence": 60, "message": "STOP sign ahead"},
    "Speed_Limit_120": {"confidence": 55, "message": "Speed limit 120 ahead"},
    "Speed_Limit_90": {"confidence": 55, "message": "Speed limit 90 ahead"},
    "Speed_Limit_40": {"confidence": 55, "message": "Speed limit 40 ahead"},
}
# ---- Frame extraction (unchanged) ----
FPS_TARGET = 3
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# ---- Lens health / blur gate (new) ----
# Laplacian variance below this = frame likely blurred, glare-washed, or a
# dirty/wet/obstructed lens. Frame is NOT skipped, just tagged, since it may
# still hold a real hazard.
BLUR_VARIANCE_THRESHOLD = 60.0

# If this many CONSECUTIVE frames are flagged degraded, it's more likely a
# camera problem than one bad frame from a bump — raise a health alert
# instead of quietly reporting "no hazards" the whole time.
CONSECUTIVE_DEGRADED_ALERT = 15  # ~5 seconds at 3fps

# ---- Event dedup (new) ----
# Detections of the same class within this many meters and this many
# seconds of each other are treated as one physical hazard, not separate
# alerts, as the bus drives past it across many frames.
DEDUP_RADIUS_METERS = 8
DEDUP_TIME_WINDOW_SEC = 4

# ---- Confined test route (new) ----
# Named waypoints for the pilot corridor, in order, used ONLY to simulate a
# plausible GPS trail for offline test videos — standing in for real GPS
# until the bus has a live GPS/OBD feed. This keeps every test video's fake
# coordinates confined to this actual corridor instead of jumping around all
# of Delhi/India.
#
# IMPORTANT: these lat/lon values are approximate central points for each
# named area, not verified precise bus-stop coordinates. Refine them with
# real reference points (Google Maps pins, or your actual route survey)
# before this represents anything but a rough test corridor.
ROUTE_WAYPOINTS = [
    {"name": "Old Delhi", "lat": 28.6562, "lon": 77.2410},
    {"name": "Chandni Chowk", "lat": 28.6506, "lon": 77.2303},
    {"name": "Sadar Bazar", "lat": 28.6660, "lon": 77.2110},
    {"name": "Kamla Nagar", "lat": 28.6784, "lon": 77.2064},
    {"name": "Azadpur", "lat": 28.7069, "lon": 77.1746},
    {"name": "Jahangirpuri", "lat": 28.7280, "lon": 77.1636},
    {"name": "Badli", "lat": 28.7434, "lon": 77.1329},
]

# ---- Storage ----
FRAMES_DIR = "frames"
EVENTS_DIR = "events"       # confirmed, deduplicated hazard events
OUTPUT_JSON = "results.json"
MANIFEST_JSON = "frames_manifest.json"
PLATES_OUTPUT_JSON = "plates.json"   # ANPR plate-reading results

# ---- Network gap handling (new) ----
RETRY_ATTEMPTS = 3          # retry API call this many times before giving up
QUEUE_FILE = "pending_frames.json"   # stores frames that failed even after retries
