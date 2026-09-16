#!/usr/bin/env python3
"""
Wrapper for the road-hazard detection pipeline.

CLI usage (unchanged):
    python run.py <video-file>

Server usage:
    uvicorn run:app --port 8001 --reload
"""

import os
import sys
import time
import json
import shutil
import venv
import subprocess
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

BASE_DIR = Path(__file__).resolve().parent
VENV_DIR = BASE_DIR / "venv"
FRAMES_DIR = BASE_DIR / "frames"
REPORTS_DIR = BASE_DIR / "reports"
UPLOADS_DIR = BASE_DIR / "uploads"
EVENTS_JSON = BASE_DIR / "events" / "events.json"
PLATES_JSON = BASE_DIR / "plates.json"
HAZARD_REPORT_PDF = BASE_DIR / "reports" / "hazard-report.pdf"
INIT_SCRIPT = BASE_DIR / "init.py"
DETECT_SCRIPT = BASE_DIR / "detect_hazard.py"
PREPARE_IMAGE_SCRIPT = BASE_DIR / "prepare_image.py"
FRAMES_WAIT_TIMEOUT = 60

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8001")


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_venv():
    if VENV_DIR.exists():
        return
    print(f"Creating virtual environment at {VENV_DIR} ...")
    venv.EnvBuilder(with_pip=True).create(VENV_DIR)


def run_step(python_exe: Path, script: Path, *args):
    cmd = [str(python_exe), str(script), *args]
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"'{script.name}' failed (exit code {result.returncode}).")


def wait_for_frames_dir():
    print(f"\nWaiting for {FRAMES_DIR} ...")
    waited = 0
    while not FRAMES_DIR.is_dir():
        time.sleep(1)
        waited += 1
        if waited >= FRAMES_WAIT_TIMEOUT:
            raise RuntimeError(f"'{FRAMES_DIR}' was not created after {FRAMES_WAIT_TIMEOUT}s.")
    print(f"{FRAMES_DIR} found.")


def run_pipeline_for_file(input_path: Path) -> dict:
    """video -> init.py (extract + overlay frames) -> wait -> detect_hazard.py
    photo -> prepare_image.py (single frame, no extraction) -> detect_hazard.py
    Either way, writes events.json and, only if events were found,
    reports/hazard-report.pdf."""
    ensure_venv()
    python_exe = venv_python()

    # Report PDF from a previous run shouldn't leak into this run's response
    # if this run detects nothing.
    if HAZARD_REPORT_PDF.exists():
        HAZARD_REPORT_PDF.unlink()

    # Clear stale frames from a previous run so leftover files never get
    # re-detected alongside this run's frame(s) — extract_frames.py and
    # prepare_image.py both overwrite frame_NNNNN.* sequentially but never
    # delete leftovers from a PRIOR run that produced more frames.
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)

    ext = input_path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        run_step(python_exe, PREPARE_IMAGE_SCRIPT, "--image", str(input_path))
    else:
        run_step(python_exe, INIT_SCRIPT, "--video", str(input_path))
        wait_for_frames_dir()

    run_step(python_exe, DETECT_SCRIPT)

    events = json.loads(EVENTS_JSON.read_text()) if EVENTS_JSON.exists() else []
    plates = json.loads(PLATES_JSON.read_text()) if PLATES_JSON.exists() else []

    # Attach a browsable URL to each event, pointing at the exact frame
    # (already has time/lat/lon/bus/zone burned into it) used for detection.
    for event in events:
        frame_filename = event.get("frame_filename")
        if frame_filename:
            event["frame_url"] = f"{PUBLIC_BASE_URL}/frames/{frame_filename}"

    result = {"events": events, "plates": plates}

    # Only surface a report_url if detect_hazard.py actually wrote the PDF
    # (i.e. this run found at least one hazard event).
    if HAZARD_REPORT_PDF.exists():
        result["report_url"] = f"{PUBLIC_BASE_URL}/reports/hazard-report.pdf"

    return result


# ---------------- CLI mode (unchanged behaviour) ----------------

def main():
    if len(sys.argv) < 2:
        sys.exit(f"Usage: python {Path(sys.argv[0]).name} <video-or-photo-file>")

    input_path = Path(sys.argv[1])
    if not input_path.is_file():
        sys.exit(f"Error: file not found: {input_path}")

    required_scripts = (DETECT_SCRIPT,)
    required_scripts += (PREPARE_IMAGE_SCRIPT,) if input_path.suffix.lower() in IMAGE_EXTENSIONS else (INIT_SCRIPT,)
    for required in required_scripts:
        if not required.exists():
            sys.exit(f"Error: missing required script: {required}")

    if not (BASE_DIR / ".env").exists():
        print("Warning: no .env file found. Roboflow API keys are read from it.")

    try:
        result = run_pipeline_for_file(input_path)
    except RuntimeError as exc:
        sys.exit(str(exc))

    print(f"\nPipeline complete. {len(result['events'])} event(s), {len(result['plates'])} plate(s).")
    if result.get("report_url"):
        print(f"Report: {result['report_url']}")
    else:
        print("No hazards detected — no report generated.")


# ---------------- Server mode (for frontend upload) ----------------

app = FastAPI(title="UrbanLens Demo Runner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

FRAMES_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
app.mount("/frames", StaticFiles(directory=FRAMES_DIR), name="frames")
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")


@app.post("/api/run-demo")
async def run_demo_endpoint(file: UploadFile = File(...)):
    allowed = VIDEO_EXTENSIONS + IMAGE_EXTENSIONS
    if not file.filename.lower().endswith(allowed):
        raise HTTPException(400, "Please upload a video (.mp4/.mov/.avi) or a photo (.jpg/.jpeg/.png).")

    UPLOADS_DIR.mkdir(exist_ok=True)
    dest = UPLOADS_DIR / file.filename
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        result = await run_in_threadpool(run_pipeline_for_file, dest)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc

    return result


if __name__ == "__main__":
    main()
