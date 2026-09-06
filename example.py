"""
Driver script for running frame extraction with bus data JSON.

Usage:
    python example.py --video path/to/video.mp4 [--zone "Old Delhi"] [--fps 2] [--bus-data bus_data.json] [--start-time "2026-09-06T10:00:00"]

If --zone is omitted, a random bus is chosen.
If --start-time is not given, the bus's timestamp from the JSON is used as the video start time.
"""

import argparse
from datetime import datetime
from extract_frames import extract_frames
from bus_location_provider import BusLocationProvider


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
        # Use the bus's timestamp from the data
        bus_time_str = provider.bus["timestamp"].replace('Z', '+00:00')
        video_start = datetime.fromisoformat(bus_time_str).replace(tzinfo=None)

    print(f"Using bus {provider.bus['bus_id']} in zone {provider.bus['zone']}")
    print(f"Video start time: {video_start.isoformat()}")

    # Run extraction with metadata overlay enabled
    extract_frames(
        video_path=args.video,
        fps_target=args.fps,
        location_provider=provider,
        video_start_time=video_start,
        overlay_metadata=True
    )


if __name__ == "__main__":
    main()