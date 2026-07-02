import serial
import struct
import threading


class IBusReceiver:
    """
    Reads FlySky iBUS protocol over UART.
    Packet format: 0x20, 0x40, 14 channels x 2 bytes (little-endian), 2 byte checksum
    Channel values: 1000-2000 microseconds, 1500 = center/neutral
    """
    PACKET_LEN = 32
    DEADZONE = 50  # ignore small stick movement near center

    def __init__(self,
                 serial_port='/dev/serial0',
                 baud=115200,
                 steering_channel=0,   # CH1
                 throttle_channel=1,   # CH2
                 mode_channel=4):      # CH5 - for manual/auto switch
        self.port = serial_port
        self.baud = baud
        self.steering_ch = steering_channel
        self.throttle_ch = throttle_channel
        self.mode_ch = mode_channel

        self.channels = [1500] * 14
        self.running = False
        self._ser = None
        self._thread = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # DonkeyCar part interface                                             #
    # ------------------------------------------------------------------ #

    def update(self):
        """Background thread: continuously read iBUS packets."""
        buf = bytearray()
        while self.running:
            try:
                data = self._ser.read(self.PACKET_LEN)
            except Exception:
                break
            if not data:
                continue
            buf.extend(data)
            while len(buf) >= self.PACKET_LEN:
                if buf[0] == 0x20 and buf[1] == 0x40:
                    self._parse(buf[:self.PACKET_LEN])
                    buf = buf[self.PACKET_LEN:]
                else:
                    del buf[0]  # re-sync

    def run_threaded(self):
        """Called by DonkeyCar main loop. Returns (steering, throttle, mode_raw)."""
        with self._lock:
            steering = self._scale(self.channels[self.steering_ch])
            throttle = self._scale(self.channels[self.throttle_ch])
            mode_raw = self.channels[self.mode_ch]  # 1000=manual, 2000=auto
        return steering, throttle, mode_raw

    def shutdown(self):
        self.running = False
        if self._ser:
            self._ser.close()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def start(self):
        self._ser = serial.Serial(self.port, self.baud, timeout=1)
        self.running = True
        self._thread = threading.Thread(target=self.update, daemon=True)
        self._thread.start()
        return self

    def _parse(self, packet):
        # Validate checksum
        checksum = 0xFFFF
        for b in packet[:30]:
            checksum -= b
        rx_checksum = struct.unpack_from('<H', packet, 30)[0]
        if checksum != rx_checksum:
            return  # bad packet, skip
        with self._lock:
            for i in range(14):
                self.channels[i] = struct.unpack_from('<H', packet, 2 + i * 2)[0]

    def _scale(self, value):
        """Convert iBUS value (1000-2000) to DonkeyCar range (-1.0 to 1.0)."""
        mid = 1500
        if abs(value - mid) < self.DEADZONE:
            return 0.0
        if value > mid:
            return min(1.0, (value - mid) / (2000 - mid))
        else:
            return max(-1.0, (value - mid) / (mid - 1000))
