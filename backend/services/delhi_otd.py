"""Delhi OTD GTFS-Realtime VehiclePositions client with GPS-derived speed."""
from __future__ import annotations

import os
import math
from datetime import UTC, datetime
import httpx
from google.transit import gtfs_realtime_pb2
from .corridor import is_within_monitored_corridor, monitored_zone

OTD_URL = 'https://otd.delhi.gov.in/api/realtime/VehiclePositions.pb'

# Cache: {vehicle_id: (lat, lng, timestamp)}
_position_cache: dict[str, tuple[float, float, datetime]] = {}

class DelhiOTDError(RuntimeError):
    pass

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two coordinates."""
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * r * math.asin(math.sqrt(a))

def _calculate_speed(vehicle_id: str, lat: float, lng: float, 
                     timestamp: datetime | None) -> tuple[float | None, str | None]:
    """Calculate speed from position change if feed doesn't provide it."""
    if not timestamp:
        return None, None
    
    prev = _position_cache.get(vehicle_id)
    _position_cache[vehicle_id] = (lat, lng, timestamp)
    
    if not prev:
        return None, None  # First sighting
    
    prev_lat, prev_lng, prev_time = prev
    time_diff = (timestamp - prev_time).total_seconds()
    
    # Need at least 5 seconds between updates for accurate calculation
    if time_diff < 5 or time_diff > 300:
        return None, None
    
    distance_m = _haversine_m(prev_lat, prev_lng, lat, lng)
    speed_kmh = round((distance_m / time_diff) * 3.6, 1)
    
    # Filter GPS noise (buses don't go >120 km/h)
    if speed_kmh > 120 or speed_kmh < 0:
        return 0.0, 'gps-derived' if speed_kmh < 1 else None
        
    return speed_kmh, 'gps-derived'

async def fetch_live_buses() -> tuple[str, int, list[dict[str, object]]]:
    api_key = os.getenv('DELHI_OTD_API_KEY')
    if not api_key:
        raise DelhiOTDError('Delhi OTD API key is not configured on the server.')
    
    timeout = float(os.getenv('DELHI_OTD_TIMEOUT_SECONDS', '12'))
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                OTD_URL, 
                params={'key': api_key}, 
                headers={'Accept': 'application/x-protobuf'}
            )
            response.raise_for_status()
    except httpx.HTTPError as error:
        raise DelhiOTDError('Delhi OTD VehiclePositions service is unavailable.') from error

    feed = gtfs_realtime_pb2.FeedMessage()
    try:
        feed.ParseFromString(response.content)
    except Exception as error:
        raise DelhiOTDError('Delhi OTD returned an invalid GTFS-Realtime response.') from error

    buses: list[dict[str, object]] = []
    vehicle_positions_received = 0
    now = datetime.now(UTC)
    
    for entity in feed.entity:
        if not entity.HasField('vehicle') or not entity.vehicle.HasField('position'):
            continue
            
        vehicle_positions_received += 1
        vehicle = entity.vehicle
        latitude = float(vehicle.position.latitude)
        longitude = float(vehicle.position.longitude)
        
        if not is_within_monitored_corridor(latitude, longitude):
            continue
            
        vehicle_id = vehicle.vehicle.id or entity.id or 'unknown'
        
        # Parse timestamp
        if vehicle.timestamp:
            timestamp = datetime.fromtimestamp(vehicle.timestamp, UTC)
        else:
            timestamp = now
            
        # Get speed: prefer feed, fallback to GPS calculation
        speed = None
        speed_source = None
        
        if vehicle.position.HasField('speed'):
            speed = round(float(vehicle.position.speed) * 3.6, 1)
            speed_source = 'feed'
        else:
            speed, speed_source = _calculate_speed(vehicle_id, latitude, longitude, timestamp)
        
        buses.append({
            'bus_id': vehicle_id,
            'label': vehicle.vehicle.label or None,
            'latitude': latitude,
            'longitude': longitude,
            'speed': speed,
            'speed_source': speed_source,
            'bearing': float(vehicle.position.bearing) if vehicle.position.HasField('bearing') else None,
            'timestamp': timestamp.isoformat().replace('+00:00', 'Z'),
            'zone': monitored_zone(latitude, longitude),
        })
    
    # Cleanup stale cache entries (10 min old)
    cutoff = now.timestamp() - 600
    stale = [k for k, (_, _, ts) in _position_cache.items() if ts.timestamp() < cutoff]
    for k in stale:
        del _position_cache[k]
    
    return now.isoformat().replace('+00:00','Z'), vehicle_positions_received, buses