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
                 fix_stale_secs=4.0, junction_radius_m=12.0,
                 turn_threshold_deg=30.0, arrive_radius_m=8.0,
                 destination=None):
        """
        :param graphml_path: campus graph for LOCAL routing. Normally None:
               the EC2 mission server owns the graph and sends a finished
               waypoint list, so the Pi never loads networkx or a graphml at
               all. Set it only to test A->B without a server.
        :param geofence: [(lat, lon), ...] polygon; None disables (nav/safe then
               only reflects fix freshness)

        :param destination: (lat, lon) to drive to, or None to sit idle until
               something calls set_destination(). Routing needs a CURRENT
               position, and there is no fix for the first 30 s or more after
               boot, so a destination given here is held and resolved into a
               route as soon as the first fix arrives.

        Defaults are sized for a NEO-6M, which is GPS-only (no GLONASS or
        Galileo) and updates at 1 Hz out of the box — so expect roughly 3-7 m
        of error rather than the 2.5 m a multi-constellation NEO-M8N manages.
        The radii are therefore generous: a junction_radius smaller than your
        position error means the command fires at the wrong place, or not at
        all. Widen them further if your survey shows poor fixes; none of this
        affects steering, which is vision-only.
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

        # (lat, lon, monotonic_time) written as one tuple by the poll thread
        self._fix = None
        self.route = []            # [(lat, lon), ...] waypoints, set by set_destination
        self.route_index = 0
        self.arrived = False
        self._lock = threading.Lock()
        self.running = True

        # held until the first fix; see set_destination's position requirement
        self._pending_destination = destination
        self._route_key = None
        self._last_route_attempt = 0.0
        if destination:
            logger.info("destination pending first GPS fix: %s", destination)

    # ---------- routing ----------

    def _node_latlon(self, node):
        d = self.graph.nodes[node]
        return float(d["y"]), float(d["x"])  # OSMnx convention: y=lat, x=lon

    def _nearest_node(self, lat, lon):
        return min(self.graph.nodes,
                   key=lambda n: haversine_m(lat, lon, *self._node_latlon(n)))

    def set_destination(self, dest_lat, dest_lon):
        """
        Compute a route from the current position to (dest_lat, dest_lon).

        Requires a live fix, since the route starts wherever we are. If none is
        available yet the destination is queued and retried automatically once
        the receiver locks on.
        """
        fix = self._fix
        if self.graph is None:
            logger.error("cannot route: no campus graph loaded "
                         "(set CAMPUS_GRAPHML and copy the graphml to the Pi)")
            return False
        if fix is None:
            logger.info("no fix yet; holding destination %s until one arrives",
                        (dest_lat, dest_lon))
            self._pending_destination = (dest_lat, dest_lon)
            return False
        import networkx as nx
        src = self._nearest_node(fix[0], fix[1])
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
        self._pending_destination = None
        logger.info("route set: %d waypoints to %s",
                    len(self.route), (dest_lat, dest_lon))
        return True

    def _resolve_pending(self):
        """Turn a queued destination into a route once a fix exists."""
        if not self._pending_destination or self._fix is None:
            return
        now = time.monotonic()
        if now - self._last_route_attempt < 5.0:
            return  # routing walks every graph node; don't do it every tick
        self._last_route_attempt = now
        self.set_destination(*self._pending_destination)

    def _junction_command(self, lat, lon):
        """LEFT/STRAIGHT/RIGHT when close to the next waypoint, else None."""
        with self._lock:
            route, i = list(self.route), self.route_index
        if not route:
            return None

        # advance past waypoints we've reached
        while i < len(route) and haversine_m(lat, lon, *route[i]) < self.arrive_radius_m:
            i += 1
        with self._lock:
            self.route_index = i
        if i >= len(route):
            self.arrived = True
            return None

        dist_next = haversine_m(lat, lon, *route[i])
        if dist_next > self.junction_radius_m or i + 1 >= len(route):
            return None

        # approaching a junction: compare inbound vs outbound bearing
        inbound = bearing_deg(lat, lon, *route[i])
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
        # so readline wakes up periodically and shutdown() is actually honoured
        # instead of the thread blocking forever on a silent socket
        s.settimeout(2.0)
        try:
            f = s.makefile("rw")
            f.write('?WATCH={"enable":true,"json":true}\n')
            f.flush()
            while self.running:
                try:
                    line = f.readline()
                except socket.timeout:
                    continue  # no data this window; fix_time ages, safe goes False
                if not line:
                    logger.warning("gpsd closed the connection")
                    break
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if msg.get("class") == "TPV" and msg.get("mode", 0) >= 2:
                    lat, lon = msg.get("lat"), msg.get("lon")
                    if lat is None or lon is None:
                        continue
                    # single atomic store: a reader can never see a new lat
                    # paired with a stale lon
                    self._fix = (lat, lon, time.monotonic())
        finally:
            s.close()

    # ---------- part interface ----------

    def set_route(self, waypoints):
        """
        Adopt a route computed elsewhere (normally the EC2 mission server).

        This is the usual path: the server owns the campus graph and networkx,
        so the Pi never loads either. Local routing via set_destination stays
        available for testing without a server.
        """
        route = [(float(a), float(b)) for a, b in waypoints]
        if len(route) < 2:
            logger.error("refusing a route with %d waypoints", len(route))
            return False
        with self._lock:
            self.route = route
            self.route_index = 0
            self.arrived = False
        self._pending_destination = None
        logger.info("route adopted: %d waypoints, ending at %s",
                    len(route), route[-1])
        return True

    def run_threaded(self, route=None):
        """
        :param route: [[lat, lon], ...] from the mission server, or None. A new
               route replaces the current one; the same route repeated is
               ignored, so the client can republish it every tick. None leaves
               the current route running — a server that goes away must not
               strand a cart mid-delivery, since the route is already local.
        """
        if route:
            key = (route[0], route[-1], len(route))
            if key != self._route_key:
                self._route_key = key
                self.set_route(route)
        self._resolve_pending()
        fix = self._fix  # single read; the poll thread may replace it at any time
        if fix is None:
            return 0.0, 0.0, None, False, self.arrived

        lat, lon, fix_time = fix
        fresh = (time.monotonic() - fix_time) < self.fix_stale_secs
        safe = fresh  # no/stale fix -> unsafe, fail closed
        if safe and self.geofence:
            safe = point_in_polygon(lat, lon, self.geofence)
        command = self._junction_command(lat, lon) if fresh else None
        return lat, lon, command, safe, self.arrived

    def shutdown(self):
        self.running = False
