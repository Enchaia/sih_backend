#!/usr/bin/env python3
"""
Wrapper for the road-hazard detection pipeline.

Usage:
    python run.py <video-file>

1. Creates venv/ if it doesn't exist.
2. Runs init.py --video <video-file> USING THE VENV's interpreter.
3. Waits for frames/ to exist.
4. Runs detect_hazard.py, also using the venv's interpreter.

Note: this script itself doesn't need to run inside the venv — it launches
init.py and detect_hazard.py as subprocesses using the venv's own python
binary, so their imports (roboflow, cv2, paddleocr, etc.) resolve correctly
regardless of what interpreter you used to launch run.py.
"""

import os
import sys
import time
import venv
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_DIR = BASE_DIR / "venv"
FRAMES_DIR = BASE_DIR / "frames"
INIT_SCRIPT = BASE_DIR / "init.py"
DETECT_SCRIPT = BASE_DIR / "detect_hazard.py"
FRAMES_WAIT_TIMEOUT = 60  # seconds


def venv_python() -> Path:
    """Path to the python executable INSIDE venv/, cross-platform."""
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_venv():
    if VENV_DIR.exists():
        return
    print(f"Creating virtual environment at {VENV_DIR} ...")
    venv.EnvBuilder(with_pip=True).create(VENV_DIR)


def run_step(python_exe: Path, script: Path, *args):
    """Run a script with the venv's python, streaming output live.
    Raises SystemExit if the script fails, instead of continuing silently.
    """
    cmd = [str(python_exe), str(script), *args]
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"'{script.name}' failed (exit code {result.returncode}). Stopping.")


def wait_for_frames_dir():
    print(f"\nWaiting for {FRAMES_DIR} ...")
    waited = 0
    while not FRAMES_DIR.is_dir():
        time.sleep(1)
        waited += 1
        if waited >= FRAMES_WAIT_TIMEOUT:
            sys.exit(f"'{FRAMES_DIR}' was not created after {FRAMES_WAIT_TIMEOUT}s.")
    print(f"{FRAMES_DIR} found.")


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

    ensure_venv()
    python_exe = venv_python()

    run_step(python_exe, INIT_SCRIPT, "--video", str(video_path))
    wait_for_frames_dir()
    run_step(python_exe, DETECT_SCRIPT)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
