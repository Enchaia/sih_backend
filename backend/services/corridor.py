"""Approximate monitored-zone geometry. Replace with official GIS polygons when supplied."""
from __future__ import annotations

from typing import Final
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

Zone = tuple[str, tuple[float, float], tuple[tuple[float, float], ...]]

# Each coordinate is (latitude, longitude). Shapely receives (longitude, latitude).
MONITORED_AREAS: Final[tuple[Zone, ...]] = (
    ('Old Delhi', (28.6562, 77.2285), ((28.648,77.217),(28.648,77.240),(28.665,77.242),(28.669,77.222))),
    ('Chandni Chowk', (28.6506, 77.2303), ((28.642,77.222),(28.642,77.241),(28.655,77.244),(28.660,77.226))),
    ('Sadar Bazar', (28.6585, 77.2057), ((28.650,77.195),(28.650,77.216),(28.668,77.219),(28.672,77.200))),
    ('Kamla Nagar', (28.6825, 77.2044), ((28.673,77.193),(28.673,77.216),(28.691,77.219),(28.696,77.198))),
    ('Azadpur', (28.7072, 77.1805), ((28.697,77.169),(28.697,77.192),(28.716,77.194),(28.721,77.174))),
    ('Jahangirpuri', (28.7244, 77.1630), ((28.714,77.151),(28.714,77.175),(28.733,77.178),(28.738,77.157))),
    ('Badli', (28.7457, 77.1385), ((28.735,77.126),(28.735,77.151),(28.754,77.154),(28.760,77.133))),
)

ROUTE: Final = [(center[1], center[0]) for _, center, _ in MONITORED_AREAS]
ZONE_POLYGONS: Final = {name: Polygon([(lng, lat) for lat, lng in boundary]) for name, _, boundary in MONITORED_AREAS}
# ~1.15 km flexibility in Delhi latitude, also covering connections between adjacent zones.
CORRIDOR: Final = unary_union([*ZONE_POLYGONS.values(), LineString(ROUTE).buffer(0.0105)])

def is_within_monitored_corridor(latitude: float, longitude: float) -> bool:
    return CORRIDOR.covers(Point(longitude, latitude))

def monitored_zone(latitude: float, longitude: float) -> str | None:
    point = Point(longitude, latitude)
    for name, polygon in ZONE_POLYGONS.items():
        if polygon.covers(point):
            return name
    if not CORRIDOR.covers(point):
        return None
    # The connector between zones is still part of the corridor; assign nearest zone.
    return min(MONITORED_AREAS, key=lambda area: point.distance(Point(area[1][1], area[1][0])))[0]
