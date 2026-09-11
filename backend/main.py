from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.delhi_otd import DelhiOTDError, fetch_live_buses

load_dotenv()
app = FastAPI(title='UrbanLens Live Transit API', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173','https://cuddly-robot-4qjv9xqrj74jcq6qw-5173.app.github.dev'], allow_credentials=False, allow_methods=['GET'], allow_headers=['*'])

class LiveBus(BaseModel):
    bus_id: str | None
    label: str | None
    latitude: float
    longitude: float
    speed: float | None
    bearing: float | None
    timestamp: str | None
    zone: str | None

class LiveBusesResponse(BaseModel):
    timestamp: str
    corridor: str = 'Old Delhi-Badli'
    total_live_vehicles_received: int
    corridor_buses_returned: int
    buses: list[LiveBus]

@app.get('/api/health')
async def health() -> dict[str, str]:
    return {'status':'ok'}

@app.get('/api/live-buses', response_model=LiveBusesResponse)
async def live_buses() -> LiveBusesResponse:
    try:
        timestamp, vehicle_count, buses = await fetch_live_buses()
        return LiveBusesResponse(timestamp=timestamp, total_live_vehicles_received=vehicle_count, corridor_buses_returned=len(buses), buses=buses)
    except DelhiOTDError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
