#!/usr/bin/env python3
"""
Campus cart mission server — runs on EC2, not on the Pi.

Owns everything that needs the campus map and none of what needs a deadline:

  here  -> the campus graph, networkx routing, the operator's map, mission
           state, position history. Heavy, latency-tolerant, and none of it
           belongs on a 2 GB Pi.
  Pi    -> steering, throttle, obstacle stops, geofence. All local, all still
           working if this server is on fire.

The cart POLLS this server; nothing here ever connects to the cart. Campus NAT
blocks inbound connections, so an outbound poll is the only thing that works
without asking IT for a port forward.

A mission is handed over as a complete waypoint list. Once the cart has it, the
delivery finishes with or without us — a server outage must not strand a cart
in a corridor.

Run:
    pip install -r requirements.txt
    export CART_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
    python app.py --graph campus_graph.graphml

Put it behind nginx with TLS before using it for real; the token is sent as a
bearer header and is only as private as the transport.
"""
import argparse
import itertools
import logging
import math
import os
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mission-server")

STATIC_DIR = Path(__file__).resolve().parent / "static"
TOKEN = os.environ.get("CART_TOKEN")
STALE_TELEMETRY_SECS = 20.0

app = FastAPI(title="Campus Cart Mission Server")

_state_lock = threading.Lock()
_graph = None
_missions = {}     # cart_id -> {"mission_id", "route", "destination", "created"}
_telemetry = {}    # cart_id -> last reported telemetry
_mission_seq = itertools.count(1)


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------

def require_token(authorization: str = Header(default="")):
    """
    A request here can dispatch a physical vehicle across a campus, so the
    endpoint is not left open. Set CART_TOKEN in the environment; leaving it
    unset is allowed only for local development and is logged loudly.
    """
    if not TOKEN:
        return
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="bad or missing token")


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------

def load_graph(path):
    global _graph
    import networkx as nx
    _graph = nx.read_graphml(path)
    logger.info("campus graph loaded: %d nodes, %d edges",
                _graph.number_of_nodes(), _graph.number_of_edges())


def _node_latlon(node):
    d = _graph.nodes[node]
    return float(d["y"]), float(d["x"])      # OSMnx: y=lat, x=lon


def _haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(a))


def _nearest_node(lat, lon):
    return min(_graph.nodes, key=lambda n: _haversine_m(lat, lon, *_node_latlon(n)))


def compute_route(from_lat, from_lon, to_lat, to_lon):
    """(lat, lon) pair -> waypoint list, or None if unreachable."""
    import networkx as nx
    src, dst = _nearest_node(from_lat, from_lon), _nearest_node(to_lat, to_lon)
    try:
        path = nx.shortest_path(_graph, src, dst, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    return [list(_node_latlon(n)) for n in path]


def route_length_m(route):
    return sum(_haversine_m(*route[i], *route[i + 1])
               for i in range(len(route) - 1))


# --------------------------------------------------------------------------
# API — cart side (polled by MissionClient)
# --------------------------------------------------------------------------

class Telemetry(BaseModel):
    cart_id: str = "cart-1"
    mission_id: int | None = None
    lat: float = 0.0
    lon: float = 0.0
    has_fix: bool = False
    safe: bool = False
    arrived: bool = False
    corridor_clear: bool = False
    sonar_stop: bool = False
    mode: str = "user"
    ts: float = 0.0


@app.post("/api/telemetry", dependencies=[Depends(require_token)])
def post_telemetry(t: Telemetry):
    with _state_lock:
        _telemetry[t.cart_id] = {**t.model_dump(), "received": time.time()}
        mission = _missions.get(t.cart_id)
        # the cart is the authority on arrival; it is the one that got there
        if mission and t.arrived and t.mission_id == mission["mission_id"]:
            logger.info("cart %s completed mission %s", t.cart_id, t.mission_id)
            _missions.pop(t.cart_id, None)
    return {"ok": True}


@app.get("/api/mission", dependencies=[Depends(require_token)])
def get_mission(cart_id: str = "cart-1"):
    with _state_lock:
        mission = _missions.get(cart_id)
    if not mission:
        return {"mission_id": None, "route": []}
    return {"mission_id": mission["mission_id"], "route": mission["route"]}


# --------------------------------------------------------------------------
# API — operator side (the phone app)
# --------------------------------------------------------------------------

class Dispatch(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    cart_id: str = "cart-1"


@app.post("/api/dispatch", dependencies=[Depends(require_token)])
def dispatch(d: Dispatch):
    if _graph is None:
        raise HTTPException(503, "no campus graph loaded")

    with _state_lock:
        tel = _telemetry.get(d.cart_id)

    # routing starts from where the cart actually is, so a stale or missing
    # position would silently route from the wrong place
    if not tel or not tel.get("has_fix"):
        raise HTTPException(409, "cart has no GPS fix yet")
    if time.time() - tel["received"] > STALE_TELEMETRY_SECS:
        raise HTTPException(409, "cart has not reported in recently")

    route = compute_route(tel["lat"], tel["lon"], d.lat, d.lon)
    if not route or len(route) < 2:
        raise HTTPException(422, "no route to that point on the campus map")

    mission_id = next(_mission_seq)
    with _state_lock:
        _missions[d.cart_id] = {
            "mission_id": mission_id, "route": route,
            "destination": [d.lat, d.lon], "created": time.time(),
        }
    length = route_length_m(route)
    logger.info("mission %s for %s: %d waypoints, %.0f m",
                mission_id, d.cart_id, len(route), length)
    return {"mission_id": mission_id, "waypoints": len(route),
            "distance_m": round(length), "route": route}


@app.post("/api/cancel", dependencies=[Depends(require_token)])
def cancel(cart_id: str = "cart-1"):
    with _state_lock:
        removed = _missions.pop(cart_id, None)
    logger.info("mission cancelled for %s", cart_id)
    return {"ok": True, "cancelled": bool(removed)}


@app.get("/api/status", dependencies=[Depends(require_token)])
def status(cart_id: str = "cart-1"):
    with _state_lock:
        tel = dict(_telemetry.get(cart_id) or {})
        mission = _missions.get(cart_id)
    age = time.time() - tel["received"] if tel.get("received") else None
    return JSONResponse({
        **tel,
        # a cart that stopped reporting is not a cart sitting where it was
        "online": age is not None and age < STALE_TELEMETRY_SECS,
        "age_secs": round(age, 1) if age is not None else None,
        "mission_id": mission["mission_id"] if mission else None,
        "destination": mission["destination"] if mission else None,
        "route": mission["route"] if mission else None,
    })


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="campus_graph.graphml",
                    help="from scripts/build_campus_graph.py")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if not TOKEN:
        logger.warning("CART_TOKEN is not set — this server will accept "
                       "commands from ANYONE who can reach it. Set it before "
                       "exposing this to the internet.")
    load_graph(args.graph)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
