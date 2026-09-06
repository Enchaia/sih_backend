"""
Frame extraction stage.

Same CLAHE brightening as your original extract.py — unchanged. New on top
of that:

- flags blurred/lens-obstructed frames instead of silently treating them
  like any other frame
- writes a metadata manifest (unique id, real timestamp, location, lens
  status) alongside the frames, so the detection stage doesn't have to
  recompute any of this later
- optionally overlays metadata text onto the saved frames (if overlay_metadata=True)

No frame-similarity redundancy check here anymore — at fps_target=3,
consecutive sampled frames are far enough apart in time that a fixed
pixel-difference threshold doesn't reliably tell "static" from "just
driving", so it added noise without a real signal.
"""

import os
import json
import uuid
import cv2
from datetime import datetime, timedelta

from config import (
    FPS_TARGET, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE,
    FRAMES_DIR, MANIFEST_JSON, CONSECUTIVE_DEGRADED_ALERT,
)
from frame_quality import lens_health
from bus_location_provider import BusLocationProvider   # <-- changed import


def auto_brighten_clahe(frame, clip_limit=CLAHE_CLIP_LIMIT, tile_grid_size=CLAHE_TILE_GRID_SIZE):
    """Brighten dark frames while improving local contrast using CLAHE. (unchanged)"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def overlay_metadata(frame, timestamp_str, lat, lon, source, bus_id=None, zone=None):
    """Draw metadata text on the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (0, 255, 0)  # green

    lines = [
        f"Time : {timestamp_str}",
        f"Lat  : {lat:.6f}",
        f"Lon  : {lon:.6f}",
        f"Source: {source}",
    ]
    if bus_id:
        lines.append(f"Bus  : {bus_id}")
    if zone:
        lines.append(f"Zone : {zone}")

    y0 = 30
    dy = 25
    for i, line in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(frame, line, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


def extract_frames(video_path, output_dir=FRAMES_DIR, fps_target=FPS_TARGET,
                   location_provider=None, video_start_time=None,
                   draw_overlay=False):
    """
    location_provider: anything with .get_location(progress_fraction) -> dict.
        Defaults to BusLocationProvider (uses bus_data.json).
    video_start_time: real-world datetime the video/recording started.
        Defaults to "now" — pass the actual capture start time for real runs
        so manifest timestamps reflect when the footage was actually shot,
        not when you happened to run this script.
    overlay_metadata: if True, draws timestamp, coordinates, source, and
        optional bus/zone on the saved frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    location_provider = location_provider or BusLocationProvider()   # <-- changed default
    video_start_time = video_start_time or datetime.now()

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps_target
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_interval = max(1, int(video_fps / fps_target))

    frame_count, saved_count = 0, 0
    consecutive_degraded = 0
    manifest = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            enhanced = auto_brighten_clahe(frame)

            degraded, blur_score = lens_health(enhanced)
            consecutive_degraded = consecutive_degraded + 1 if degraded else 0
            camera_health_alert = consecutive_degraded >= CONSECUTIVE_DEGRADED_ALERT

            progress = frame_count / max(total_frames - 1, 1)
            location = location_provider.get_location(progress)
            elapsed_sec = frame_count / video_fps
            capture_time = video_start_time + timedelta(seconds=elapsed_sec)
            timestamp_str = capture_time.strftime("%Y-%m-%d %H:%M:%S")

            # Overlay metadata if requested
            if draw_overlay and location:
                lat = location.get("lat", 0.0)
                lon = location.get("lon", 0.0)
                source = location.get("source", "unknown")
                bus_id = location.get("bus_id")
                zone = location.get("zone")
                enhanced = overlay_metadata(enhanced, timestamp_str, lat, lon, source, bus_id, zone)

            filename = f"frame_{saved_count:05d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, enhanced)

            manifest.append({
                "unique_id": str(uuid.uuid4()),
                "filename": filename,
                "timestamp": capture_time.isoformat(),
                "location": location,
                "lens_degraded": degraded,
                "blur_score": blur_score,
                "camera_health_alert": camera_health_alert,
            })

            saved_count += 1

        frame_count += 1

    cap.release()

    with open(os.path.join(output_dir, MANIFEST_JSON), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {saved_count} frames (CLAHE-enhanced) to {output_dir}")
    if any(m["camera_health_alert"] for m in manifest):
        print("WARNING: sustained blur/lens-obstruction detected during this video — check the camera.")


if __name__ == "__main__":
    extract_frames("dashcam.mp4")
