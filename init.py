"""
Driver script for running frame extraction with bus data JSON.

Usage:
    python example.py --video path/to/video.mp4 [--zone "Old Delhi"] [--fps 2] [--bus-data bus_data.json] [--start-time "2026-09-06T10:00:00"]

If --zone is omitted, a random bus is chosen.
If --start-time is not given, the CURRENT real-world time (converted to IST)
is used as the video start time — not a static value from the JSON file.
This means every run reflects "now", regardless of what machine/timezone
the script is actually executed on.
"""

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from extract_frames import extract_frames
from bus_location_provider import BusLocationProvider

IST = ZoneInfo("Asia/Kolkata")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--zone", default=None, help="Zone to filter bus (e.g., 'Old Delhi')")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second to extract")
    parser.add_argument("--bus-data", default="bus_data.json", help="Path to bus data JSON")
    parser.add_argument("--start-time", default=None, help="Optional real-world start time (ISO format)")
    args = parser.parse_args()

    # Create the location provider (loads bus data and picks a bus)
    provider = BusLocationProvider(bus_data_path=args.bus_data, zone=args.zone)

    # Determine video start time
    if args.start_time:
        video_start = datetime.fromisoformat(args.start_time)
    else:
        # Dynamic: always "right now", correctly converted to IST regardless
        # of the server/machine's own local timezone setting.
        video_start = datetime.now(IST).replace(tzinfo=None)

    print(f"Using bus {provider.bus['bus_id']} in zone {provider.bus['zone']}")
    print(f"Video start time (IST, live): {video_start.isoformat()}")

    # Run extraction with metadata overlay enabled
    extract_frames(
        video_path=args.video,
        fps_target=args.fps,
        location_provider=provider,
        video_start_time=video_start,
        draw_overlay=True
    )


if __name__ == "__main__":
    main()
