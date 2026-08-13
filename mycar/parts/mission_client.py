"""
MissionClient — the Pi's link to the EC2 mission server.

The server assigns missions; the Pi executes them. That split is deliberate and
the boundary matters more than the code:

  server decides  -> which route to drive (needs the campus graph, networkx,
                     and the user's pin). None of that has to live on a 2 GB Pi.
  Pi decides      -> everything with a deadline. Steering, throttle, obstacle
                     stops, the geofence. All local, all working with the
                     network unplugged.

NOTHING IN THE CONTROL LOOP CROSSES THE NETWORK. Campus WiFi to Mumbai and back
is 50-200 ms on a good day and unbounded on a bad one, which is fine for "go to
this pin" and catastrophic for "stop". So the cart pulls a whole waypoint list
in one go and then drives it alone; if the server disappears mid-delivery the
cart keeps going to the destination it already has, because the route is
already local. Losing a server should not strand a cart in a corridor.

The geofence deliberately stays on the Pi for the same reason: it is
fail-closed safety, and safety that depends on a reachable server is not
safety.

Connection direction: the Pi POLLS OUT. Campus NAT would block anything
inbound, and an outbound poll needs no port forwarding and no IT request.
"""
import json
import logging
import threading
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


class MissionClient:
    """
    Threaded DonkeyCar part.

    Inputs:  gps/lat, gps/lon, nav/safe, nav/arrived, seg/corridor_clear,
             sonar/stop, user/mode
    Outputs: mission/route — [[lat, lon], ...] or None, consumed by GpsNav

    Polls for a mission and pushes telemetry on the same cadence. Both are
    best-effort: a failure logs and retries, and never affects driving.
    """

    def __init__(self, server_url, cart_id="cart-1", token=None,
                 poll_secs=3.0, timeout_secs=5.0):
        """
        :param server_url: e.g. "https://ec2-x-x-x-x.compute.amazonaws.com"
        :param token: shared secret sent as a bearer header. The server can
               dispatch a physical vehicle, so it must not be open to the
               internet unauthenticated.
        :param poll_secs: mission poll interval. 3 s is plenty — a human
               dropping a pin does not need sub-second delivery, and a slower
               poll is kinder to a flaky campus link.
        :param timeout_secs: hard cap per request so a hung server can never
               wedge this thread.
        """
        self.server_url = server_url.rstrip("/")
        self.cart_id = cart_id
        self.token = token
        self.poll_secs = poll_secs
        self.timeout_secs = timeout_secs

        self.route = None            # the mission, held locally once fetched
        self.mission_id = None
        self.telemetry = {}
        self.online = False
        self.running = True
        self._last_offline_log = 0.0
        self._lock = threading.Lock()
        logger.info("mission client -> %s (cart %s)", self.server_url, cart_id)

    # ---------- transport ----------

    def _request(self, path, payload=None):
        req = urllib.request.Request(f"{self.server_url}{path}")
        req.add_header("Accept", "application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            req.add_header("Content-Type", "application/json")
            req.get_method = lambda: "POST"
        with urllib.request.urlopen(req, data=data,
                                    timeout=self.timeout_secs) as resp:
            return json.loads(resp.read().decode())

    # ---------- poll loop ----------

    def update(self):
        while self.running:
            try:
                self._push_telemetry()
                self._fetch_mission()
                self.online = True
            except (urllib.error.URLError, OSError, ValueError) as exc:
                self.online = False
                now = time.monotonic()
                if now - self._last_offline_log > 30.0:
                    # not an error for driving: the route is already local
                    logger.warning("mission server unreachable (%s). Continuing "
                                   "on the route already loaded.", exc)
                    self._last_offline_log = now
            except Exception:
                self.online = False
                logger.exception("mission client error")
            time.sleep(self.poll_secs)

    def _push_telemetry(self):
        with self._lock:
            body = dict(self.telemetry)
        body["cart_id"] = self.cart_id
        body["mission_id"] = self.mission_id
        self._request("/api/telemetry", body)

    def _fetch_mission(self):
        data = self._request(f"/api/mission?cart_id={self.cart_id}")
        mission_id = data.get("mission_id")

        if mission_id is None:
            if self.mission_id is not None:
                logger.info("mission cleared by server")
                with self._lock:
                    self.mission_id, self.route = None, None
            return

        if mission_id == self.mission_id:
            return  # already driving it

        route = data.get("route") or []
        if len(route) < 2:
            logger.error("server sent mission %s with %d waypoints — ignoring",
                         mission_id, len(route))
            return

        with self._lock:
            self.mission_id = mission_id
            self.route = [(float(p[0]), float(p[1])) for p in route]
        logger.info("mission %s accepted: %d waypoints", mission_id, len(route))

    # ---------- part interface ----------

    def run_threaded(self, lat=None, lon=None, safe=False, arrived=False,
                     corridor_clear=False, sonar_stop=False, mode="user"):
        with self._lock:
            self.telemetry = {
                # 0.0 is GpsNav's no-fix placeholder, not a real position
                "lat": lat or 0.0, "lon": lon or 0.0,
                "has_fix": bool(lat) and bool(lon),
                "safe": bool(safe), "arrived": bool(arrived),
                "corridor_clear": bool(corridor_clear),
                "sonar_stop": bool(sonar_stop), "mode": mode or "user",
                "ts": time.time(),
            }
            route = self.route
        return route

    def shutdown(self):
        self.running = False
