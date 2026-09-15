#!/usr/bin/env python3
"""
Wrapper for the road-hazard detection pipeline.

CLI usage (unchanged):
    python run.py <video-file>

Server usage (new):
    uvicorn run:app --port 8001 --reload
    -> exposes POST /api/run-demo for the frontend's "Run demo" upload button.

Both paths call the exact same ensure_venv() / run_step() logic, so the
actual pipeline (init.py -> extract_frames.py -> detect_hazard.py) is
untouched.
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
from starlette.concurrency import run_in_threadpool

BASE_DIR = Path(__file__).resolve().parent
VENV_DIR = BASE_DIR / "venv"
FRAMES_DIR = BASE_DIR / "frames"
UPLOADS_DIR = BASE_DIR / "uploads"
EVENTS_JSON = BASE_DIR / "events" / "events.json"
PLATES_JSON = BASE_DIR / "plates.json"
INIT_SCRIPT = BASE_DIR / "init.py"
DETECT_SCRIPT = BASE_DIR / "detect_hazard.py"
FRAMES_WAIT_TIMEOUT = 60  # seconds


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


def run_pipeline_for_video(video_path: Path) -> dict:
    """Same three steps run.py always did — video -> init.py -> wait -> detect_hazard.py —
    just returned as data instead of printed to a terminal."""
    ensure_venv()
    python_exe = venv_python()

    run_step(python_exe, INIT_SCRIPT, "--video", str(video_path))
    wait_for_frames_dir()
    run_step(python_exe, DETECT_SCRIPT)

    events = json.loads(EVENTS_JSON.read_text()) if EVENTS_JSON.exists() else []
    plates = json.loads(PLATES_JSON.read_text()) if PLATES_JSON.exists() else []
    return {"events": events, "plates": plates}


# ---------------- CLI mode (unchanged behaviour) ----------------

def main():
    if len(sys.argv) < 2:
        sys.exit(f"Usage: python {Path(sys.argv[0]).name} <video-file>")

    video_path = Path(sys.argv[1])
    if not video_path.is_file():
        sys.exit(f"Error: video file not found: {video_path}")

    for required in (INIT_SCRIPT, DETECT_SCRIPT):
        if not required.exists():
            sys.exit(f"Error: missing required script: {required}")

    if not (BASE_DIR / ".env").exists():
        print("Warning: no .env file found. Roboflow API keys are read from it.")

    try:
        result = run_pipeline_for_video(video_path)
    except RuntimeError as exc:
        sys.exit(str(exc))

    print(f"\nPipeline complete. {len(result['events'])} event(s), {len(result['plates'])} plate(s).")


# ---------------- Server mode (new, for frontend upload) ----------------

app = FastAPI(title="UrbanLens Demo Runner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.post("/api/run-demo")
async def run_demo_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi")):
        raise HTTPException(400, "Please upload a .mp4, .mov, or .avi file.")

    UPLOADS_DIR.mkdir(exist_ok=True)
    dest = UPLOADS_DIR / file.filename
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        # subprocess.run() blocks, so run it off the event loop
        result = await run_in_threadpool(run_pipeline_for_video, dest)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc

    return result


if __name__ == "__main__":
    main()