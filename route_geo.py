"""
Location metadata for the pipeline.

Two providers, same interface (.get_location(...) -> dict with lat/lon):

- SimulatedRouteLocationProvider: for testing with pre-recorded videos where
  there's no real GPS. Interpolates a plausible lat/lon along the fixed
  pilot corridor (config.ROUTE_WAYPOINTS), proportional to how far through
  the video you are. Confines every test video's fake coordinates to the
  actual route instead of scattering them across all of Delhi/India.

- GPSLocationProvider: for the real bus. Reads live GPS fixes and falls back
  to dead reckoning (last known fix + constant-velocity extrapolation from
  heading/speed) when the signal drops (tunnels, underpasses, urban
  canyons), so an event never goes out with no location at all.

Swap which one extract_frames.py uses with one line — nothing downstream
needs to know which is active, since both return the same shape of dict.
"""

import math
from config import ROUTE_WAYPOINTS


def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters between two lat/lon points."""
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class SimulatedRouteLocationProvider:
    """Interpolates a fake-but-plausible GPS trail along a fixed route.

    Use only for offline testing of pre-recorded videos. `progress_fraction`
    is 0-1, how far through the current video you are (frame_index /
    total_frames), so the simulated position moves start-to-end along the
    corridor at a steady pace across the video's length.
    """

    def __init__(self, waypoints=None):
        self.waypoints = waypoints or ROUTE_WAYPOINTS
        self._segment_lengths = []
        self._total_length = 0.0
        for i in range(len(self.waypoints) - 1):
            a, b = self.waypoints[i], self.waypoints[i + 1]
            d = haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])
            self._segment_lengths.append(d)
            self._total_length += d

    def get_location(self, progress_fraction):
        progress_fraction = min(max(progress_fraction, 0.0), 1.0)
        target_dist = progress_fraction * self._total_length

        covered = 0.0
        for i, seg_len in enumerate(self._segment_lengths):
            reached_end = i == len(self._segment_lengths) - 1
            if covered + seg_len >= target_dist or reached_end:
                a, b = self.waypoints[i], self.waypoints[i + 1]
                seg_progress = 0.0 if seg_len == 0 else (target_dist - covered) / seg_len
                seg_progress = min(max(seg_progress, 0.0), 1.0)
                lat = a["lat"] + (b["lat"] - a["lat"]) * seg_progress
                lon = a["lon"] + (b["lon"] - a["lon"]) * seg_progress
                nearest_name = a["name"] if seg_progress < 0.5 else b["name"]
                return {"lat": round(lat, 6), "lon": round(lon, 6), "near": nearest_name, "source": "simulated"}
            covered += seg_len

        last = self.waypoints[-1]
        return {"lat": last["lat"], "lon": last["lon"], "near": last["name"], "source": "simulated"}


class GPSLocationProvider:
    """Real GPS provider for the live bus, with dead-reckoning fallback.

    Wire `read_gps_fix` to your actual GPS/OBD hardware — it should be a
    zero-arg callable returning (lat, lon, speed_mps, heading_deg) on a good
    fix, or None when there's no fix right now.
    """

    def __init__(self, read_gps_fix):
        self._read_gps_fix = read_gps_fix
        self._last_fix = None  # (lat, lon, speed_mps, heading_deg, t_seconds)
        self._lost_since = None

    def get_location(self, t_seconds):
        fix = self._read_gps_fix()
        if fix is not None:
            lat, lon, speed_mps, heading_deg = fix
            self._last_fix = (lat, lon, speed_mps, heading_deg, t_seconds)
            self._lost_since = None
            return {"lat": round(lat, 6), "lon": round(lon, 6), "source": "gps"}

        if self._last_fix is None:
            return None  # never had a fix yet — nothing to reckon from

        if self._lost_since is None:
            self._lost_since = t_seconds

        lat, lon, speed_mps, heading_deg, last_t = self._last_fix
        elapsed = t_seconds - last_t
        dist_m = speed_mps * elapsed
        heading_rad = math.radians(heading_deg)
        # Flat-earth projection — fine for the short gaps dead reckoning is
        # meant to cover; don't rely on this for minutes-long GPS outages.
        dlat = (dist_m * math.cos(heading_rad)) / 111320
        dlon = (dist_m * math.sin(heading_rad)) / (111320 * math.cos(math.radians(lat)))
        return {
            "lat": round(lat + dlat, 6),
            "lon": round(lon + dlon, 6),
            "source": "dead_reckoning",
            "signal_lost_for_sec": round(t_seconds - self._lost_since, 1),
        }