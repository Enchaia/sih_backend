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

# ---- Roboflow model config (unchanged values, new structure) ----
API_KEY = "YOUR_API_KEY"

# Fill in the workspace + project slugs for your two trained models shown in
# your Roboflow dashboard:
#   - "debr-road-water-pot-traffic-1-yolo11n-t1"  -> the combined defect model
#   - "traffic-light-7auyu-wdg0w-1-yolo11n-t1"    -> the traffic-light model
# The dashboard "Name"/"ID" columns are display names, not necessarily the
# exact slug the SDK needs — check the project's own page/URL for the slug.
MODEL_CONFIGS = [
    {"name": "defect_multiclass", "workspace": "your-workspace-name", "project": "project-one", "version": 1},
    {"name": "traffic_light", "workspace": "your-workspace-name", "project": "project-two", "version": 1},
]

CONFIDENCE = 40
OVERLAP = 30
MAX_WORKERS = 8  # total parallel API calls across BOTH models combined

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

# ---- Network gap handling (new) ----
RETRY_ATTEMPTS = 3          # retry API call this many times before giving up
QUEUE_FILE = "pending_frames.json"   # stores frames that failed even after retries