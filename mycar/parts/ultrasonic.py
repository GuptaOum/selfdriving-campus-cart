"""
UltrasonicArray — 3x HC-SR04 reflex safety layer. No ML, independent of vision.

Mount the sensors >= 10-12 cm above the ground, aimed level or slightly up:
a ~6 cm speed breaker then stays below the beam and never trips the hard
stop, while pedestrians/walls/big rocks still do.

Uses pigpio (microsecond edge timestamps in the daemon) — RPi.GPIO's
Python-side edge timing jitters by milliseconds, which is centimeters of
error. Run `sudo systemctl enable --now pigpiod` on the Pi.
"""
import logging
import statistics
import time

logger = logging.getLogger(__name__)

SOUND_CM_PER_US = 0.0343 / 2.0  # round trip


class _Sensor:
    """One HC-SR04 read via pigpio edge callbacks."""

    def __init__(self, pi, trig, echo, timeout_s=0.03):
        import pigpio
        self.pi, self.trig, self.echo, self.timeout_s = pi, trig, echo, timeout_s
        pi.set_mode(trig, pigpio.OUTPUT)
        pi.set_mode(echo, pigpio.INPUT)
        pi.write(trig, 0)
        self._rise_tick = None
        self._echo_us = None
        self._cb = pi.callback(echo, pigpio.EITHER_EDGE, self._edge)

    def _edge(self, gpio, level, tick):
        import pigpio
        if level == 1:
            self._rise_tick = tick
        elif level == 0 and self._rise_tick is not None:
            self._echo_us = pigpio.tickDiff(self._rise_tick, tick)

    def read_cm(self):
        """Trigger one ping; returns distance in cm or None on timeout."""
        self._echo_us = None
        self.pi.gpio_trigger(self.trig, 10, 1)  # 10 us pulse
        deadline = time.monotonic() + self.timeout_s
        while self._echo_us is None and time.monotonic() < deadline:
            time.sleep(0.001)
        if self._echo_us is None or self._echo_us > 25000:  # >4 m = junk
            return None
        return self._echo_us * SOUND_CM_PER_US

    def cancel(self):
        self._cb.cancel()


class UltrasonicArray:
    """
    Threaded DonkeyCar part polling left/center/right at ~15 Hz with
    median-of-5 filtering per sensor.

    Outputs: sonar/left, sonar/center, sonar/right (cm, None if no echo),
             sonar/stop (bool), sonar/bias (-1 steer left .. +1 steer right, 0 none)

    Tiers (from the plan):
      any sensor < stop_cm            -> sonar/stop            (hard throttle cut)
      center in [stop_cm, caution_cm) -> bias toward the side with more room
    Resume needs `clear_count` consecutive clear polls (no flicker restarts).
    """

    def __init__(self, pins, stop_cm=30.0, caution_cm=80.0, clear_count=5,
                 poll_hz=15.0, median_n=5):
        """:param pins: dict {'left': (trig, echo), 'center': ..., 'right': ...}"""
        import pigpio
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpiod not running — sudo systemctl start pigpiod")
        self.sensors = {name: _Sensor(self.pi, *te) for name, te in pins.items()}
        self.history = {name: [] for name in pins}
        self.median_n = median_n
        self.stop_cm, self.caution_cm = stop_cm, caution_cm
        self.clear_count, self._clear_streak = clear_count, 0
        self.poll_dt = 1.0 / poll_hz

        self.dist = {name: None for name in pins}
        self.stop = False
        self.bias = 0.0
        self.running = True
        logger.info("UltrasonicArray ready: %s", list(pins))

    def _filtered(self, name, raw):
        h = self.history[name]
        if raw is not None:
            h.append(raw)
            del h[:-self.median_n]
        # no echo can mean genuinely nothing in range OR a bad ping;
        # median over recent readings absorbs single dropouts
        return statistics.median(h) if h else None

    def update(self):
        while self.running:
            t0 = time.monotonic()
            for name, sensor in self.sensors.items():
                self.dist[name] = self._filtered(name, sensor.read_cm())
                time.sleep(0.005)  # stagger pings so sensors don't cross-talk

            vals = [d for d in self.dist.values() if d is not None]
            danger = any(d < self.stop_cm for d in vals)
            if danger:
                self.stop = True
                self._clear_streak = 0
            else:
                self._clear_streak += 1
                if self._clear_streak >= self.clear_count:
                    self.stop = False

            center = self.dist.get("center")
            if not self.stop and center is not None and center < self.caution_cm:
                left = self.dist.get("left") or 0.0
                right = self.dist.get("right") or 0.0
                # steer toward the side reporting more free space
                self.bias = 0.5 if right > left else -0.5
            else:
                self.bias = 0.0

            time.sleep(max(0.0, self.poll_dt - (time.monotonic() - t0)))

    def run_threaded(self):
        return (self.dist.get("left"), self.dist.get("center"),
                self.dist.get("right"), self.stop, self.bias)

    def shutdown(self):
        self.running = False
        for s in self.sensors.values():
            s.cancel()
        self.pi.stop()


if __name__ == "__main__":
    # standalone hardware test: python3 -m parts.ultrasonic
    import threading
    logging.basicConfig(level=logging.INFO)
    arr = UltrasonicArray(pins={"left": (5, 6), "center": (19, 26), "right": (20, 21)})
    threading.Thread(target=arr.update, daemon=True).start()
    try:
        while True:
            print(arr.run_threaded())
            time.sleep(0.2)
    except KeyboardInterrupt:
        arr.shutdown()
