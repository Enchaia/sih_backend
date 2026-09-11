"""Delhi OTD GTFS-Realtime VehiclePositions client; credentials stay server-side."""
from __future__ import annotations

import os
import math
from datetime import UTC, datetime
import httpx
from google.transit import gtfs_realtime_pb2
from .corridor import is_within_monitored_corridor, monitored_zone

OTD_URL = 'https://otd.delhi.gov.in/api/realtime/VehiclePositions.pb'

# Cache previous positions to calculate GPS-derived speed
# Format: {vehicle_id: (lat, lng, timestamp)}
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
    """
    Calculate speed from GPS position change if feed doesn't provide it.
    Returns (speed_kmh, speed_source).
    """
    if not timestamp:
        return None, None
    
    prev = _position_cache.get(vehicle_id)
    _position_cache[vehicle_id] = (lat, lng, timestamp)
    
    if not prev:
        return None, None  # First sighting, no previous position
    
    prev_lat, prev_lng, prev_time = prev
    time_diff = (timestamp - prev_time).total_seconds()
    
    # Ignore if time gap too small (<5s) or too large (>5min - likely stale data)
    if time_diff < 5 or time_diff > 300:
        return None, None
    
    distance_m = _haversine_m(prev_lat, prev_lng, lat, lng)
    speed_ms = distance_m / time_diff
    speed_kmh = round(speed_ms * 3.6, 1)
    
    # Filter unrealistic jumps (GPS noise)
    if speed_kmh > 120:  # Buses don't go faster than 120 km/h
        return None, None
        
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
            
        # Try feed speed first, fallback to GPS-derived
        speed = None
        speed_source = None
        
        if vehicle.position.HasField('speed'):
            speed = round(float(vehicle.position.speed) * 3.6, 1)  # m/s → km/h
            speed_source = 'feed'
        else:
            speed, speed_source = _calculate_speed(vehicle_id, latitude, longitude, timestamp)
        
        buses.append({
            'bus_id': vehicle_id,
            'label': vehicle.vehicle.label or None,
            'latitude': latitude,
            'longitude': longitude,
            'speed': speed,
            'speed_source': speed_source,  # 'feed' or 'gps-derived'
            'bearing': float(vehicle.position.bearing) if vehicle.position.HasField('bearing') else None,
            'timestamp': timestamp.isoformat().replace('+00:00', 'Z'),
            'zone': monitored_zone(latitude, longitude),
        })
    
    # Cleanup old cache entries (buses not seen for 10 minutes)
    cutoff = now.timestamp() - 600
    stale_ids = [vid for vid, (_, _, ts) in _position_cache.items() 
                 if ts.timestamp() < cutoff]
    for vid in stale_ids:
        del _position_cache[vid]
    
    timestamp_str = now.isoformat().replace('+00:00', 'Z')
    return timestamp_str, vehicle_positions_received, buses
