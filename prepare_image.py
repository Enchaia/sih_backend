"""
Driver script for feeding a single PHOTO (not video) into the hazard
detection pipeline. Used when the demo upload is an image instead of a
dashcam video — bypasses extract_frames.py entirely (no CLAHE, no lens
health check, no overlay, no fps sampling). The photo is copied as-is into
FRAMES_DIR and handed straight to detect_hazard.py.

A manifest entry is still written (unique id, timestamp, location) because
detect_hazard.py's event dedup requires both to register a hazard event —
the location comes from the same BusLocationProvider used for videos, at
progress_fraction=0.0 (its starting point), and the timestamp is "now".

Usage:
    python prepare_image.py --image path/to/photo.jpg [--zone "Old Delhi"] [--bus-data bus_data.json]
"""

import argparse
import json
import os
import shutil
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from config import FRAMES_DIR, MANIFEST_JSON
from bus_location_provider import BusLocationProvider

IST = ZoneInfo("Asia/Kolkata")


def prepare_image(image_path, output_dir=FRAMES_DIR, location_provider=None, zone=None, bus_data_path="bus_data.json"):
    os.makedirs(output_dir, exist_ok=True)

    location_provider = location_provider or BusLocationProvider(bus_data_path=bus_data_path, zone=zone)
    location = location_provider.get_location(0.0)
    capture_time = datetime.now(IST).replace(tzinfo=None)

    ext = os.path.splitext(image_path)[1].lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"  # detect_hazard.py only picks up these extensions
    filename = f"frame_00000{ext}"
    dest_path = os.path.join(output_dir, filename)
    shutil.copyfile(image_path, dest_path)

    manifest = [{
        "unique_id": str(uuid.uuid4()),
        "filename": filename,
        "timestamp": capture_time.isoformat(),
        "location": location,
        "lens_degraded": False,
        "blur_score": None,
        "camera_health_alert": False,
    }]

    with open(os.path.join(output_dir, MANIFEST_JSON), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Using bus {location_provider.bus['bus_id']} in zone {location_provider.bus['zone']}")
    print(f"Prepared 1 photo frame -> {dest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input photo")
    parser.add_argument("--zone", default=None, help="Zone to filter bus (e.g., 'Old Delhi')")
    parser.add_argument("--bus-data", default="bus_data.json", help="Path to bus data JSON")
    args = parser.parse_args()

    prepare_image(args.image, zone=args.zone, bus_data_path=args.bus_data)


if __name__ == "__main__":
    main()
