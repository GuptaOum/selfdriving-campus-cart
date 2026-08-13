"""
SafetyArbiter — merges all autonomy signals into pilot/angle + pilot/throttle.

Priority (high to low), from the approved plan:
  1. RC transmitter override      — handled UPSTREAM by DonkeyCar's DriveMode
                                    (CH5 'user' mode ignores pilot/* entirely)
  2. HC-SR04 hard stop (< 30 cm)
  3. Geofence violation / no GPS fix while on a mission -> stop (fail closed)
  4. YOLO person-in-corridor      -> stop (close) or half throttle (medium)
  5. Segmentation corridor blocked-> stop
  6. BREAKER mode                 -> straight + creep throttle
  7. Normal: segmentation steering (+ sonar caution bias), seg throttle

Deliberately a dumb, readable if-chain — this is the part a safety reviewer
(or you at 2 AM after a crash) must be able to audit at a glance.
"""
import logging

logger = logging.getLogger(__name__)


class SafetyArbiter:

    def __init__(self, creep_throttle=0.14, mission_requires_gps=False,
                 require_sonar=True, require_yolo=False):
        """
        :param mission_requires_gps: True once you run GPS A->B missions —
               nav/safe False then stops the cart. Keep False for track tests
               indoors where there is legitimately no fix.
        :param require_sonar: stop when the sonar array is not responding. Only
               set False if HAVE_ULTRASONIC is False, i.e. there is no array to
               be unhealthy.
        :param require_yolo: stop when the detector is erroring or stalled.
               Enable once HAVE_YOLO_GUARD is True and you rely on it.
        """
        self.creep_throttle = creep_throttle
        self.mission_requires_gps = mission_requires_gps
        self.require_sonar = require_sonar
        self.require_yolo = require_yolo
        self._last_reason = None

    def _log(self, reason):
        if reason != self._last_reason:
            logger.info("arbiter: %s", reason)
            self._last_reason = reason

    def run(self, seg_angle, seg_throttle, corridor_clear,
            sonar_stop, sonar_bias, sonar_healthy,
            yolo_stop, yolo_slow, yolo_healthy,
            breaker_active, nav_safe, nav_arrived):
        # tolerate not-yet-initialized threaded parts
        seg_angle = seg_angle or 0.0
        seg_throttle = seg_throttle or 0.0

        # Health first, so the log says "sensor broken" rather than "obstacle" —
        # the two demand completely different responses from you in the field.
        if self.require_sonar and not sonar_healthy:
            self._log("SONAR UNHEALTHY (not responding) — stop, check wiring")
            return 0.0, 0.0
        if self.require_yolo and not yolo_healthy:
            self._log("YOLO GUARD UNHEALTHY — stop, check model/CPU load")
            return 0.0, 0.0

        if sonar_stop:
            self._log("SONAR HARD STOP")
            return 0.0, 0.0
        if self.mission_requires_gps and not nav_safe:
            self._log("GPS UNSAFE (no fix / outside geofence) — fail closed")
            return 0.0, 0.0
        if nav_arrived:
            self._log("arrived at destination")
            return 0.0, 0.0
        if yolo_stop:
            self._log("person/obstacle close — stop")
            return 0.0, 0.0
        if not corridor_clear:
            self._log("no drivable corridor — stop")
            return 0.0, 0.0
        if breaker_active:
            self._log("BREAKER mode — straight + creep")
            return 0.0, self.creep_throttle

        throttle = seg_throttle * (0.5 if yolo_slow else 1.0)
        angle = max(-1.0, min(1.0, seg_angle + (sonar_bias or 0.0)))
        self._log("driving")
        return angle, throttle
