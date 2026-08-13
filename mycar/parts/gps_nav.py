"""
GpsNav — GPS route follower and junction commander.

GPS NEVER steers (5 m error vs a 30 cm car). It does three things:
  1. reports position for the tracking app / tub logging,
  2. emits LEFT/STRAIGHT/RIGHT commands near route junctions (vision executes),
  3. enforces the geofence — FAIL CLOSED: no fix, stale fix, or outside the
     polygon all mean "not safe", and the arbiter stops the cart.

The campus graph is built ONCE on a laptop with scripts/build_campus_graph.py
(OSMnx) and copied to the Pi as graphml; here we only need networkx to read it.
Position source is gpsd (sudo apt install gpsd; NEO-M8N on USB-TTL).
"""
import logging
import math
import threading
import time

logger = logging.getLogger(__name__)

EARTH_R = 6371000.0


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def point_in_polygon(lat, lon, polygon):
    """Ray casting; polygon = [(lat, lon), ...]. Avoids a shapely dependency."""
    inside = False
    n = len(polygon)
    for i in range(n):
        la1, lo1 = polygon[i]
        la2, lo2 = polygon[(i + 1) % n]
        if (lo1 > lon) != (lo2 > lon):
            t = (lon - lo1) / (lo2 - lo1)
            if lat < la1 + t * (la2 - la1):
                inside = not inside
    return inside


class GpsNav:
    """
    Threaded DonkeyCar part.

    Outputs: gps/lat, gps/lon, nav/command (None|'LEFT'|'STRAIGHT'|'RIGHT'),
             nav/safe (bool — geofence AND fix freshness, fail closed),
             nav/arrived (bool)
    """

    def __init__(self, graphml_path=None, geofence=None,
                 fix_stale_secs=3.0, junction_radius_m=8.0,
                 turn_threshold_deg=30.0, arrive_radius_m=5.0):
        """
        :param graphml_path: campus graph from build_campus_graph.py (optional —
               without it there's no routing, but tracking + geofence still work)
        :param geofence: [(lat, lon), ...] polygon; None disables (nav/safe then
               only reflects fix freshness)
        """
        self.geofence = geofence
        self.fix_stale_secs = fix_stale_secs
        self.junction_radius_m = junction_radius_m
        self.turn_threshold_deg = turn_threshold_deg
        self.arrive_radius_m = arrive_radius_m

        self.graph = None
        if graphml_path:
            import networkx as nx
            self.graph = nx.read_graphml(graphml_path)
            logger.info("Loaded campus graph: %d nodes", self.graph.number_of_nodes())

        self.lat = self.lon = None
        self.fix_time = 0.0
        self.route = []            # [(lat, lon), ...] waypoints, set by set_destination
        self.route_index = 0
        self.arrived = False
        self._lock = threading.Lock()
        self.running = True

    # ---------- routing ----------

    def _node_latlon(self, node):
        d = self.graph.nodes[node]
        return float(d["y"]), float(d["x"])  # OSMnx convention: y=lat, x=lon

    def _nearest_node(self, lat, lon):
        return min(self.graph.nodes,
                   key=lambda n: haversine_m(lat, lon, *self._node_latlon(n)))

    def set_destination(self, dest_lat, dest_lon):
        """Compute a route from current position. Called by the app layer."""
        if self.graph is None or self.lat is None:
            logger.warning("cannot route: no graph or no fix")
            return False
        import networkx as nx
        src = self._nearest_node(self.lat, self.lon)
        dst = self._nearest_node(dest_lat, dest_lon)
        try:
            path = nx.shortest_path(self.graph, src, dst, weight="length")
        except nx.NetworkXNoPath:
            logger.warning("no path %s -> %s", src, dst)
            return False
        with self._lock:
            self.route = [self._node_latlon(n) for n in path]
            self.route_index = 0
            self.arrived = False
        logger.info("route set: %d waypoints", len(self.route))
        return True

    def _junction_command(self):
        """LEFT/STRAIGHT/RIGHT when close to the next waypoint, else None."""
        with self._lock:
            route, i = list(self.route), self.route_index
        if not route or self.lat is None:
            return None

        # advance past waypoints we've reached
        while i < len(route) and haversine_m(self.lat, self.lon, *route[i]) < self.arrive_radius_m:
            i += 1
        with self._lock:
            self.route_index = i
        if i >= len(route):
            self.arrived = True
            return None

        dist_next = haversine_m(self.lat, self.lon, *route[i])
        if dist_next > self.junction_radius_m or i + 1 >= len(route):
            return None

        # approaching a junction: compare inbound vs outbound bearing
        inbound = bearing_deg(self.lat, self.lon, *route[i])
        outbound = bearing_deg(*route[i], *route[i + 1])
        turn = (outbound - inbound + 540.0) % 360.0 - 180.0  # [-180, 180]
        if turn <= -self.turn_threshold_deg:
            return "LEFT"
        if turn >= self.turn_threshold_deg:
            return "RIGHT"
        return "STRAIGHT"

    # ---------- gpsd polling ----------

    def update(self):
        while self.running:
            try:
                self._poll_gpsd()
            except Exception:
                logger.exception("gpsd connection lost; retrying in 2 s")
                time.sleep(2.0)

    def _poll_gpsd(self):
        import json
        import socket
        s = socket.create_connection(("127.0.0.1", 2947), timeout=5)
        f = s.makefile("rw")
        f.write('?WATCH={"enable":true,"json":true}\n')
        f.flush()
        for line in f:
            if not self.running:
                break
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("class") == "TPV" and msg.get("mode", 0) >= 2:
                self.lat, self.lon = msg.get("lat"), msg.get("lon")
                self.fix_time = time.monotonic()
        s.close()

    # ---------- part interface ----------

    def run_threaded(self):
        fresh = (time.monotonic() - self.fix_time) < self.fix_stale_secs \
            and self.lat is not None
        safe = fresh  # no/stale fix -> unsafe, fail closed
        if safe and self.geofence:
            safe = point_in_polygon(self.lat, self.lon, self.geofence)
        command = self._junction_command() if fresh else None
        # 0.0 placeholders keep tub logging json-safe; nav/safe flags validity
        lat = self.lat if self.lat is not None else 0.0
        lon = self.lon if self.lon is not None else 0.0
        return lat, lon, command, safe, self.arrived

    def shutdown(self):
        self.running = False
