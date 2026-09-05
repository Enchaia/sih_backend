"""
Detection stage.

Same two-model, parallel-API-call design as your original detect_hazard.py
— unchanged: same CONFIDENCE, OVERLAP, MAX_WORKERS. New on top of that:

- reads the manifest written by extract_frames.py, so every detection is
  tagged with the frame's real unique id, timestamp, and location
- handles network gaps: retries the API call a few times, and if it still
  fails, the frame is saved to a queue file (`pending_frames.json`) to be
  processed later when connectivity returns (store-and-forward)
- collapses repeated detections of the same hazard across consecutive
  frames into one event instead of one alert per frame

Honest caveat: model.predict() below still calls Roboflow's hosted API per
frame, which means detection itself needs internet the whole time. That's
fine for testing, but it does NOT satisfy "must keep detecting through
network gaps" — for real offline operation you'd export these two trained
models (Roboflow supports exporting YOLOv11 weights) and run inference
locally via ultralytics/ONNX instead of the hosted API. The queue mechanism
here ensures no data is lost during outages, but detection still requires
online access when the queue is processed.
"""

import os
import json
import uuid
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from roboflow import Roboflow

from config import (
    API_KEY, MODEL_CONFIGS, CONFIDENCE, OVERLAP, MAX_WORKERS,
    FRAMES_DIR, MANIFEST_JSON, OUTPUT_JSON, EVENTS_DIR,
    DEDUP_RADIUS_METERS, DEDUP_TIME_WINDOW_SEC,
    RETRY_ATTEMPTS, QUEUE_FILE,
)
from route_geo import haversine_m

# Thread-safe collection of frames that failed even after retries
_pending_lock = threading.Lock()
_pending_frames = []


def load_models():
    """Load all configured Roboflow models once, return as a dict keyed by name. (unchanged)"""
    rf = Roboflow(api_key=API_KEY)
    models = {}
    for cfg in MODEL_CONFIGS:
        project = rf.workspace(cfg["workspace"]).project(cfg["project"])
        models[cfg["name"]] = project.version(cfg["version"]).model
    return models


def load_manifest(frames_dir):
    path = os.path.join(frames_dir, MANIFEST_JSON)
    if not os.path.exists(path):
        print(f"No manifest found at {path} — run extract_frames.py first.")
        return {}
    with open(path) as f:
        entries = json.load(f)
    return {e["filename"]: e for e in entries}


def predict_with_model(model_name, model, filepath):
    """Send one frame to ONE model with retries, tag each detection.

    Returns (filename, model_name, detections). On final failure, the frame
    is added to the pending queue and detections is empty.
    """
    last_exception = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            prediction = model.predict(filepath, confidence=CONFIDENCE, overlap=OVERLAP).json()
            detections = prediction.get("predictions", [])
            for d in detections:
                d["source_model"] = model_name
            return os.path.basename(filepath), model_name, detections
        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt+1}/{RETRY_ATTEMPTS} failed for {filepath} with {model_name}: {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                import time
                time.sleep(2 ** attempt)  # simple exponential backoff

    # All retries failed → add to pending queue
    with _pending_lock:
        _pending_frames.append({"filepath": filepath, "model": model_name})
    print(f"Giving up on {filepath} with {model_name}. Added to pending queue.")
    return os.path.basename(filepath), model_name, []


def run_detection(frames_dir=FRAMES_DIR, output_json=OUTPUT_JSON, max_workers=MAX_WORKERS):
    models = load_models()
    manifest = load_manifest(frames_dir)

    filepaths = [
        os.path.join(frames_dir, f)
        for f in sorted(os.listdir(frames_dir))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not filepaths:
        print(f"No frames found in '{frames_dir}'. Run extract_frames.py first.")
        return

    total_tasks = len(filepaths) * len(models)
    print(f"Sending {len(filepaths)} frames to {len(models)} models "
          f"({total_tasks} total API calls, {max_workers} parallel workers)...")

    all_results = {os.path.basename(fp): [] for fp in filepaths}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for fp in filepaths:
            for model_name, model in models.items():
                future = executor.submit(predict_with_model, model_name, model, fp)
                futures[future] = (fp, model_name)

        completed = 0
        for future in as_completed(futures):
            filename, model_name, detections = future.result()
            meta = manifest.get(filename, {})
            for d in detections:
                d["event_id"] = str(uuid.uuid4())
                d["frame_unique_id"] = meta.get("unique_id")
                d["timestamp"] = meta.get("timestamp")
                d["location"] = meta.get("location")
                d["lens_degraded"] = meta.get("lens_degraded")
            all_results[filename].extend(detections)
            completed += 1
            if detections:
                classes = [d["class"] for d in detections]
                print(f"[{completed}/{total_tasks}] {filename} ({model_name}): {classes}")
            else:
                print(f"[{completed}/{total_tasks}] {filename} ({model_name}): nothing detected")

    sorted_results = {k: all_results[k] for k in sorted(all_results.keys())}
    with open(output_json, "w") as f:
        json.dump(sorted_results, f, indent=2)

    total_detections = sum(len(v) for v in sorted_results.values())
    print(f"\nDone. {total_detections} total raw hazard detections across {len(filepaths)} frames "
          f"(combined from {len(models)} models).")

    events = dedup_to_events(sorted_results)
    os.makedirs(EVENTS_DIR, exist_ok=True)
    events_path = os.path.join(EVENTS_DIR, "events.json")
    with open(events_path, "w") as f:
        json.dump(events, f, indent=2)

    print(f"Collapsed into {len(events)} deduplicated hazard events -> {events_path}")
    print(f"Raw per-frame results saved to {output_json}")

    # Write pending frames queue (if any)
    if _pending_frames:
        queue_path = os.path.join(frames_dir, QUEUE_FILE)
        with open(queue_path, "w") as f:
            json.dump(_pending_frames, f, indent=2)
        print(f"Saved {len(_pending_frames)} pending frames to {queue_path}")
    else:
        print("No pending frames (all API calls succeeded).")


def dedup_to_events(sorted_results):
    """Collapse repeated detections of the same class, close in space and
    time, across consecutive frames into a single event (keeps the highest-
    confidence detection as the representative one, and counts how many
    frames agreed — a simple multi-frame-consensus signal you can threshold
    on downstream, e.g. require frame_count >= 2 before treating it as
    confirmed rather than a single blurry-frame false positive)."""
    flat = [d for detections in sorted_results.values() for d in detections
            if d.get("location") and d.get("timestamp")]
    flat.sort(key=lambda d: d["timestamp"])

    events = []
    for d in flat:
        matched = None
        for e in events:
            if e["class"] != d["class"]:
                continue
            same_time_window = abs(_seconds_between(e["timestamp"], d["timestamp"])) <= DEDUP_TIME_WINDOW_SEC
            same_place = haversine_m(
                e["location"]["lat"], e["location"]["lon"],
                d["location"]["lat"], d["location"]["lon"],
            ) <= DEDUP_RADIUS_METERS
            if same_time_window and same_place:
                matched = e
                break
        if matched:
            matched["frame_count"] += 1
            if d["confidence"] > matched["confidence"]:
                matched.update({
                    "confidence": d["confidence"], "timestamp": d["timestamp"],
                    "location": d["location"], "representative_frame": d["frame_unique_id"],
                })
        else:
            events.append({
                "event_id": d["event_id"], "class": d["class"], "confidence": d["confidence"],
                "timestamp": d["timestamp"], "location": d["location"],
                "representative_frame": d["frame_unique_id"], "frame_count": 1,
            })
    return events


def _seconds_between(ts_a, ts_b):
    return (datetime.fromisoformat(ts_b) - datetime.fromisoformat(ts_a)).total_seconds()


if __name__ == "__main__":
    run_detection()