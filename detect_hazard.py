import os
import json
import uuid
import io
import threading
import contextlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from roboflow import Roboflow

from config import (
    MODEL_CONFIGS, API_REQUEST_CONFIDENCE, OVERLAP, MAX_WORKERS,
    CLASS_CONFIDENCE_THRESHOLDS, DEFAULT_CLASS_CONFIDENCE,
    NIGHT_CONFIDENCE_RELAXATION, DRIVER_ALERT_CLASSES,
    FRAMES_DIR, MANIFEST_JSON, OUTPUT_JSON, EVENTS_DIR,
    DEDUP_RADIUS_METERS, DEDUP_TIME_WINDOW_SEC,
    RETRY_ATTEMPTS, QUEUE_FILE,
)
from geo_utils import haversine_m  # make sure this filename matches the actual file in your project

# Thread-safe collection of frames that failed even after retries
_pending_lock = threading.Lock()
_pending_frames = []


def load_models():
    """Load all configured Roboflow models once, return as a dict keyed by
    name. Each model gets its OWN Roboflow client, since they may live under
    different workspaces with different API keys.

    Uses version.models() instead of the deprecated version.model attribute,
    which can silently return None even when a trained model exists.
    """
    models = {}
    for cfg in MODEL_CONFIGS:
        with contextlib.redirect_stdout(io.StringIO()):
            rf = Roboflow(api_key=cfg["api_key"])
            project = rf.workspace(cfg["workspace"]).project(cfg["project"])
            version = project.version(cfg["version"])
            available = version.models()

        if not available:
            raise RuntimeError(
                f"No trained model found for '{cfg['name']}' "
                f"({cfg['workspace']}/{cfg['project']}/v{cfg['version']}). "
                f"Check the Models tab on Roboflow for this project."
            )

        # available is expected to be a list of trained model objects on
        # this version — take the first one unless you have multiple and
        # need to pick a specific one.
        models[cfg["name"]] = available[0]

    return models


def load_manifest(frames_dir):
    path = os.path.join(frames_dir, MANIFEST_JSON)
    if not os.path.exists(path):
        print(f"No manifest found at {path} — run extract_frames.py first.")
        return {}
    with open(path) as f:
        entries = json.load(f)
    return {e["filename"]: e for e in entries}


def is_nighttime(timestamp_str):
    """Returns True if the given ISO timestamp falls between 7 PM and 6 AM."""
    if not timestamp_str:
        return False
    hour = datetime.fromisoformat(timestamp_str).hour
    return hour >= 19 or hour < 6


def required_confidence_for(class_name, timestamp_str=None):
    """Look up this class's minimum confidence, relaxed slightly at night."""
    threshold = CLASS_CONFIDENCE_THRESHOLDS.get(class_name, DEFAULT_CLASS_CONFIDENCE)
    if is_nighttime(timestamp_str):
        threshold -= NIGHT_CONFIDENCE_RELAXATION
    return threshold


def filter_by_class_confidence(detections, timestamp_str=None):
    """Keep only detections that meet their OWN class's confidence bar."""
    kept = []
    for d in detections:
        confidence_pct = d["confidence"] * 100  # Roboflow returns 0-1
        min_required = required_confidence_for(d["class"], timestamp_str)
        if confidence_pct >= min_required:
            d["threshold_used"] = min_required
            kept.append(d)
    return kept


def check_driver_alerts(detections):
    """Fire an immediate driver alert for sign classes (stop, speed-limit),
    independent of the hazard confidence table above."""
    for d in detections:
        class_name = d["class"]
        confidence_pct = d["confidence"] * 100
        if class_name in DRIVER_ALERT_CLASSES:
            alert_cfg = DRIVER_ALERT_CLASSES[class_name]
            if confidence_pct >= alert_cfg["confidence"]:
                trigger_alert(alert_cfg["message"], class_name, confidence_pct)


def trigger_alert(message, class_name, confidence_pct):
    """Replace with real audio/dashboard/buzzer logic once you have hardware."""
    print(f"🔔 DRIVER ALERT: {message} (confidence: {confidence_pct:.0f}%)")


def predict_with_model(model_name, model, filepath, timestamp_str=None):
    """Send one frame to ONE model with retries, filter by per-class
    confidence, and check for driver-alert signs.

    Returns (filename, model_name, detections). On final failure, the frame
    is added to the pending queue and detections is empty.
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                prediction = model.predict(
                    filepath, confidence=API_REQUEST_CONFIDENCE, overlap=OVERLAP
                ).json()
            raw_detections = prediction.get("predictions", [])
            for d in raw_detections:
                d["source_model"] = model_name

            check_driver_alerts(raw_detections)
            filtered = filter_by_class_confidence(raw_detections, timestamp_str)

            # Only print when something actually cleared its threshold
            if filtered:
                for d in filtered:
                    print(f"🚧 {d['class']} detected ({d['confidence']*100:.0f}%) "
                          f"in {os.path.basename(filepath)} [{model_name}]")

            return os.path.basename(filepath), model_name, filtered

        except Exception as e:
            # Keep failure visibility — these matter even in quiet mode
            print(f"Attempt {attempt+1}/{RETRY_ATTEMPTS} failed for {filepath} with {model_name}: {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                import time
                time.sleep(2 ** attempt)  # exponential backoff

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
    all_results = {os.path.basename(fp): [] for fp in filepaths}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for fp in filepaths:
            filename = os.path.basename(fp)
            meta = manifest.get(filename, {})
            frame_timestamp = meta.get("timestamp")  # now real, from extract_frames.py's manifest

            for model_name, model in models.items():
                future = executor.submit(predict_with_model, model_name, model, fp, frame_timestamp)
                futures[future] = (fp, model_name)

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

    sorted_results = {k: all_results[k] for k in sorted(all_results.keys())}
    with open(output_json, "w") as f:
        json.dump(sorted_results, f, indent=2)

    total_detections = sum(len(v) for v in sorted_results.values())

    events = dedup_to_events(sorted_results)
    os.makedirs(EVENTS_DIR, exist_ok=True)
    events_path = os.path.join(EVENTS_DIR, "events.json")
    with open(events_path, "w") as f:
        json.dump(events, f, indent=2)

    print(f"\nDone: {total_detections} detections, {len(events)} unique hazard events.")

    if _pending_frames:
        queue_path = os.path.join(frames_dir, QUEUE_FILE)
        with open(queue_path, "w") as f:
            json.dump(_pending_frames, f, indent=2)
        print(f"{len(_pending_frames)} frames failed and were queued for retry -> {queue_path}")


def dedup_to_events(sorted_results):
    """Collapse repeated detections of the same class, close in space and
    time, across consecutive frames into a single event."""
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
