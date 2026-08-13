import serial
import struct
import threading


class IBusReceiver:
    """
    DonkeyCar controller part that reads FlySky iBUS protocol over UART.
    Packet format: 0x20, 0x40, 14 channels x 2 bytes (little-endian), 2 byte checksum
    Channel values: 1000-2000 microseconds, 1500 = center/neutral

    Wired into the Vehicle loop as a threaded part: the framework spawns
    update() in a background thread to keep reading serial packets, and
    calls run_threaded() every drive loop tick to hand back the latest
    steering/throttle/mode/recording values without blocking.
    """
    PACKET_LEN = 32
    DEADZONE = 50  # ignore small stick movement near center (in raw 1000-2000 units)
    RECORD_THROTTLE_DEADZONE = 0.05  # scaled throttle magnitude that counts as "driving" for auto-record

    def __init__(self,
                 serial_port='/dev/serial0',
                 baud=115200,
                 steering_channel=0,   # CH1
                 throttle_channel=1,   # CH2
                 mode_channel=4,       # CH5 - for manual/auto switch
                 mode_user_max=1300,   # raw value below this => 'user'
                 mode_local_angle_max=1700,  # raw value below this => 'local_angle', else 'local'
                 auto_record_on_throttle=True):
        self.port = serial_port
        self.baud = baud
        self.steering_ch = steering_channel
        self.throttle_ch = throttle_channel
        self.mode_ch = mode_channel
        self.mode_user_max = mode_user_max
        self.mode_local_angle_max = mode_local_angle_max
        self.auto_record_on_throttle = auto_record_on_throttle

        self.channels = [1500] * 14
        self.running = True
        self._lock = threading.Lock()

        self._ser = serial.Serial(self.port, self.baud, timeout=1)
        self._thread = None

    # ------------------------------------------------------------------ #
    # DonkeyCar part interface                                             #
    # ------------------------------------------------------------------ #

    def update(self):
        """Background thread (spawned by the Vehicle framework since this
        part is added with threaded=True): continuously read iBUS packets."""
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

    def run_threaded(self, mode=None, recording=None):
        """
        Called every drive loop tick by the Vehicle framework.
        :param mode: previous 'user/mode' value (unused unless auto-record is off)
        :param recording: previous 'recording' value (passed through when auto-record is off)
        :return: (user/steering, user/throttle, user/mode, recording)
        """
        with self._lock:
            steering = self._scale(self.channels[self.steering_ch])
            throttle = self._scale(self.channels[self.throttle_ch])
            mode_raw = self.channels[self.mode_ch]

        new_mode = self._mode_from_raw(mode_raw)

        if self.auto_record_on_throttle:
            is_recording = new_mode == 'user' and abs(throttle) > self.RECORD_THROTTLE_DEADZONE
        else:
            is_recording = bool(recording)

        return steering, throttle, new_mode, is_recording

    def shutdown(self):
        self.running = False
        if self._ser:
            self._ser.close()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _mode_from_raw(self, value):
        """Map a 3-position switch channel (1000/1500/2000) to a DonkeyCar mode string."""
        if value < self.mode_user_max:
            return 'user'
        elif value < self.mode_local_angle_max:
            return 'local_angle'
        else:
            return 'local'

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
