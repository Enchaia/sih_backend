"""
Location provider that uses bus data from a JSON file (bus_data.json)
and simulates a small random walk starting from the selected bus's
coordinates. This replaces the SimulatedRouteLocationProvider and removes
the need for the route_geo.py corridor interpolation.
"""

import json
import random
import numpy as np


class BusLocationProvider:
    def __init__(self, bus_data_path="bus_data.json", zone=None, step_size_m=5):
        """
        bus_data_path: path to JSON file containing OTD bus list (e.g., the sample you posted)
        zone: optional zone name to filter; if None, a random bus is chosen
        step_size_m: maximum random displacement per frame in meters
        """
        with open(bus_data_path, "r") as f:
            data = json.load(f)
        buses = data["buses"]
        self.bus = self._pick_bus(buses, zone)
        self.start_lat = self.bus["latitude"]
        self.start_lon = self.bus["longitude"]
        self.step_size_m = step_size_m
        self.path = None  # generated lazily on first call

    def _pick_bus(self, buses, zone):
        if zone:
            candidates = [b for b in buses if b["zone"].lower() == zone.lower()]
            if not candidates:
                print(f"No buses in zone '{zone}', picking random")
                candidates = buses
        else:
            candidates = buses
        return random.choice(candidates)

    def _generate_path(self, num_points):
        """Random walk from start point."""
        path = [(self.start_lat, self.start_lon)]
        lat, lon = self.start_lat, self.start_lon
        for _ in range(num_points - 1):
            angle = random.uniform(0, 2 * np.pi)
            dist = random.uniform(0, self.step_size_m)
            dlat = dist * np.cos(angle) / 111320
            dlon = dist * np.sin(angle) / (111320 * np.cos(np.radians(lat)))
            lat += dlat
            lon += dlon
            path.append((lat, lon))
        return path

    def get_location(self, progress_fraction):
        """
        progress_fraction: float 0..1 indicating how far through the video we are.
        Returns a dict with lat, lon, source, bus_id, zone.
        """
        if self.path is None:
            # Assume up to 1000 points; enough for typical short test videos
            self.path = self._generate_path(1000)

        idx = int(progress_fraction * (len(self.path) - 1))
        idx = max(0, min(idx, len(self.path) - 1))
        lat, lon = self.path[idx]
        return {
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "source": "otd_simulated",
            "bus_id": self.bus["bus_id"],
            "zone": self.bus["zone"],
        }