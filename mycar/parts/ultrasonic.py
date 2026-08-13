"""
UltrasonicArray — 3x HC-SR04 reflex safety layer. No ML, independent of vision.

Mount the sensors >= 10-12 cm above the ground, aimed level or slightly up:
a ~6 cm speed breaker then stays below the beam and never trips the hard
stop, while pedestrians/walls/big rocks still do.

Uses pigpio (microsecond edge timestamps in the daemon) — RPi.GPIO's
Python-side edge timing jitters by milliseconds, which is centimeters of
error. Run `sudo systemctl enable --now pigpiod` on the Pi.

FAIL-CLOSED: a sensor that stops responding is treated as a reason to stop,
not as "path clear". See _Sensor.read_cm for how a live-but-nothing-in-range
sensor is distinguished from a dead one.
"""
import logging
import statistics
import time

logger = logging.getLogger(__name__)

SOUND_CM_PER_US = 0.0343 / 2.0  # round trip

# read_cm outcomes
NOTHING_IN_RANGE = float('inf')  # sensor answered, path is clear
NO_RESPONSE = None               # sensor did not answer at all — suspect wiring


class _Sensor:
    """One HC-SR04 read via pigpio edge callbacks."""

    def __init__(self, pi, trig, echo, timeout_s=0.06):
        """
        :param timeout_s: must exceed the ~38 ms 'no object' pulse the HC-SR04
               emits, otherwise every clear reading looks like a dead sensor.
        """
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
        """
        Trigger one ping. Returns distance in cm, NOTHING_IN_RANGE if the sensor
        responded but saw nothing, or NO_RESPONSE if it never drove ECHO high.

        That last distinction is what makes fail-closed possible: a working
        HC-SR04 with clear air ahead still raises ECHO (it emits a ~38 ms pulse),
        so silence on the rising edge means the sensor is unplugged or broken,
        not that the way is clear.
        """
        # clear BOTH, or a late echo from the previous ping is read as this one's
        self._rise_tick = None
        self._echo_us = None

        self.pi.gpio_trigger(self.trig, 10, 1)  # 10 us pulse
        deadline = time.monotonic() + self.timeout_s
        while self._echo_us is None and time.monotonic() < deadline:
            time.sleep(0.0005)

        if self._echo_us is None:
            # no falling edge within the timeout
            return NOTHING_IN_RANGE if self._rise_tick is not None else NO_RESPONSE
        if self._echo_us > 25000:  # > ~4 m, beyond usable range
            return NOTHING_IN_RANGE
        return self._echo_us * SOUND_CM_PER_US

    def cancel(self):
        self._cb.cancel()


class UltrasonicArray:
    """
    Threaded DonkeyCar part polling left/center/right with median-of-N
    filtering per sensor.

    Outputs: sonar/left, sonar/center, sonar/right (cm; None = not responding),
             sonar/stop (bool), sonar/bias (-1 left .. +1 right),
             sonar/healthy (bool — False means at least one sensor is silent)

    Tiers:
      any sensor < stop_cm            -> sonar/stop (hard throttle cut)
      center in [stop_cm, caution_cm) -> bias toward the side with more room
      any sensor not responding       -> sonar/stop, sonar/healthy False
    Resume needs `clear_count` consecutive clear polls, so a flickering
    reading can't restart the car.
    """

    def __init__(self, pins, stop_cm=30.0, caution_cm=80.0, clear_count=5,
                 poll_hz=15.0, median_n=5, max_misses=4):
        """
        :param pins: dict of name -> (trig, echo) or (trig, echo, angle_deg).

               The optional third element is the sensor's bearing: 0 is dead
               ahead, POSITIVE IS LEFT (matching the planner's ground frame).
               Supply angles and the array becomes a crude range scan the
               LocalPlanner can drop straight into its occupancy grid — a
               poor man's LiDAR, measured rather than inferred, for no CPU.
               Without angles you still get the stop/bias safety layer.

               A sensible fan for four sensors:
                   {"left":   (5, 6, 30), "cleft":  (19, 26, 10),
                    "cright": (20, 21, -10), "right": (16, 12, -30)}
               Roughly 20 deg apart matches the ~15 deg beam width, so the
               cones overlap slightly instead of leaving blind wedges.

        :param max_misses: consecutive non-responses before a sensor is declared
               dead. Small enough to catch a wire falling off mid-drive.

        NOTE ON SENSOR COUNT: pings must be staggered or the sensors hear each
        other, so a full sweep costs roughly (sensors x 65 ms). Three sweeps at
        ~5 Hz; eight at ~2 Hz. More sensors buy angular coverage and cost
        update rate — past about six the trade stops paying.
        """
        import pigpio
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpiod not running — sudo systemctl start pigpiod")
        self.sensors = {name: _Sensor(self.pi, spec[0], spec[1])
                        for name, spec in pins.items()}
        # bearing per sensor, degrees, positive left; absent -> straight ahead
        self.angles = {name: (float(spec[2]) if len(spec) > 2 else 0.0)
                       for name, spec in pins.items()}
        self.has_angles = any(len(spec) > 2 for spec in pins.values())
        self.history = {name: [] for name in pins}
        self.misses = {name: 0 for name in pins}
        self.median_n = median_n
        self.max_misses = max_misses
        self.stop_cm, self.caution_cm = stop_cm, caution_cm
        self.clear_count, self._clear_streak = clear_count, 0
        self.poll_dt = 1.0 / poll_hz

        self.dist = {name: None for name in pins}
        self.stop = True          # start stopped until sensors prove themselves
        self.healthy = False
        self.bias = 0.0
        self.running = True
        self._last_unhealthy_log = 0.0
        logger.info("UltrasonicArray ready: %s", list(pins))

    def _filtered(self, name, raw):
        """Median-filter a reading; returns None once a sensor is declared dead."""
        if raw is NO_RESPONSE:
            self.misses[name] += 1
            if self.misses[name] >= self.max_misses:
                # stop reporting stale distances for a sensor that has gone quiet
                self.history[name].clear()
                return None
            # tolerate a couple of dropped pings using recent history
            h = self.history[name]
            return statistics.median(h) if h else None

        self.misses[name] = 0
        if raw is NOTHING_IN_RANGE:
            self.history[name].clear()
            return NOTHING_IN_RANGE

        h = self.history[name]
        h.append(raw)
        del h[:-self.median_n]
        return statistics.median(h)

    def update(self):
        while self.running:
            t0 = time.monotonic()
            try:
                for name, sensor in self.sensors.items():
                    self.dist[name] = self._filtered(name, sensor.read_cm())
                    time.sleep(0.005)  # stagger pings so sensors don't cross-talk
                self._evaluate()
            except Exception:
                logger.exception("ultrasonic poll failed — holding stop")
                self.stop, self.healthy = True, False
                time.sleep(0.2)  # back off instead of spinning hot on a hard fault

            time.sleep(max(0.0, self.poll_dt - (time.monotonic() - t0)))

    def _evaluate(self):
        dead = [n for n, d in self.dist.items() if d is None]
        self.healthy = not dead

        if dead:
            # a proximity sensor we cannot hear from is not permission to drive
            now = time.monotonic()
            if now - self._last_unhealthy_log > 5.0:
                logger.error("sonar not responding: %s — forcing stop "
                             "(check wiring/pigpiod)", dead)
                self._last_unhealthy_log = now
            self.stop = True
            self._clear_streak = 0
            self.bias = 0.0
            return

        vals = [d for d in self.dist.values() if d is not NOTHING_IN_RANGE]
        if any(d < self.stop_cm for d in vals):
            self.stop = True
            self._clear_streak = 0
        else:
            self._clear_streak += 1
            if self._clear_streak >= self.clear_count:
                self.stop = False

        center = self.dist.get("center")
        if (not self.stop and center is not None
                and center is not NOTHING_IN_RANGE and center < self.caution_cm):
            # inf compares greater than any float, so an out-of-range side
            # correctly reads as "more room that way"
            left = self.dist.get("left") or 0.0
            right = self.dist.get("right") or 0.0
            self.bias = 0.5 if right > left else -0.5
        else:
            self.bias = 0.0

    def scan(self):
        """
        [(bearing_deg, distance_m), ...] for every sensor that saw something.

        Only real returns are included: a sensor reporting NOTHING_IN_RANGE is
        omitted rather than reported as free space. Sound glances off angled
        surfaces without coming back, so silence is not evidence of clear
        ground — the planner may add obstacles from this, never remove them.
        """
        out = []
        for name, d in self.dist.items():
            if d is None or d is NOTHING_IN_RANGE:
                continue
            out.append((self.angles.get(name, 0.0), d / 100.0))
        return out

    def run_threaded(self):
        def out(v):
            return None if v is NOTHING_IN_RANGE else v
        return (out(self.dist.get("left")), out(self.dist.get("center")),
                out(self.dist.get("right")), self.stop, self.bias,
                self.healthy, self.scan())

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
            left, center, right, stop, bias, healthy = arr.run_threaded()
            def fmt(v):
                return " ---" if v is None else f"{v:5.1f}"
            print(f"L{fmt(left)}  C{fmt(center)}  R{fmt(right)}  "
                  f"stop={stop!s:5}  bias={bias:+.1f}  healthy={healthy}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        arr.shutdown()
