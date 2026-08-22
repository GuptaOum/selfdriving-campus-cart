"""
KerasSensorFusion — PilotNet/DAVE-2 behavioural cloning with a proprioceptive
side channel.

The camera branch is stock DonkeyCar `core_cnn_layers` (the NVIDIA DAVE-2 conv
stack, unchanged). Alongside it runs a small MLP over a sensor vector — sonar
distances, and optionally IMU — whose output is concatenated with the image
features before the final dense layers. The architecture mirrors DonkeyCar's
own `default_imu`, so nothing here is exotic: it is the framework's existing
multi-input pattern with a configurable vector.

WHY THIS MIGHT DO NOTHING
    Gradient descent uses whatever predicts the label. If the camera already
    resolves every obstacle you steered around, the sensor branch weights decay
    toward zero and this model is `linear` with extra parameters. That is not a
    bug, it is the network correctly finding the channel redundant. The vector
    earns its place only on data where it is the *sole* signal: obstacles the
    camera handles poorly (low kerbs, dark objects, glare, poor light).

    Train `linear` and `fusion` on the same tubs and compare on the held-out
    track. Either result is a finding.

TRAIN/INFERENCE SKEW IS THE FAILURE MODE
    The tub stores RAW sensor readings (cm, m/s^2, deg/s) so they stay
    inspectable. Normalisation happens in exactly one place — `normalize()` —
    called by `x_transform` during training and by `SensorVectorizer` at drive
    time. Do not normalise anywhere else. A scale mismatch between the two
    paths shifts the input distribution and the car drives badly for reasons
    that look like a model problem.
"""
import logging
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input
from tensorflow.keras.models import Model

from donkeycar.parts.keras import KerasPilot, core_cnn_layers
from donkeycar.parts.interpreter import Interpreter, KerasInterpreter
from donkeycar.pipeline.types import TubRecord

logger = logging.getLogger(__name__)

# Tub keys, in the order they enter the vector. Order is part of the model
# contract: change it and every previously trained model is silently wrong.
IMU_KEYS = [f'imu/{f}_{x}' for f in ('acl', 'gyr') for x in 'xyz']

# Defaults, overridable from myconfig.py.
SONAR_MAX_CM = 400.0     # HC-SR04 usable ceiling; also the "nothing there" value
IMU_ACCEL_SCALE = 20.0   # m/s^2 at +-2g full scale
IMU_GYRO_SCALE = 250.0   # deg/s at the MPU6050 default full scale


def sonar_keys(cfg) -> List[str]:
    """Sonar tub keys in the order given by ULTRASONIC_PINS."""
    pins = getattr(cfg, 'ULTRASONIC_PINS', {}) or {}
    return [f'sonar/{name}' for name in pins]


def sensor_keys(cfg) -> List[str]:
    """
    The full sensor vector for this config, sonar first then IMU.

    Driven entirely by config, so adding or removing the IMU is a flag change
    rather than a code change. The count must match the trained model — a model
    trained with 4 sonar cannot be loaded against a 10-input config.
    """
    keys = []
    if getattr(cfg, 'HAVE_ULTRASONIC', False):
        keys += sonar_keys(cfg)
    if getattr(cfg, 'HAVE_IMU', False):
        keys += IMU_KEYS
    return keys


def normalize(cfg, values: List[Union[float, None]]) -> np.ndarray:
    """
    Raw readings -> float32 vector in roughly [0, 1] (sonar) and [-1, 1] (IMU).

    THE SINGLE SOURCE OF TRUTH for scaling. Training and inference both call
    this; nothing else may scale these numbers.

    `None` means the sonar reported nothing in range OR the sensor is silent —
    `UltrasonicArray.run_threaded` collapses both to None. Both map to 1.0
    ("far"), because a dead sensor is not the model's problem to solve: the
    array already fails closed and forces a stop via `sonar/stop`, and the
    safety arbiter sits above the pilot. Do not try to encode sensor health
    here; keep that on its own channel.
    """
    max_cm = float(getattr(cfg, 'SONAR_MAX_CM', SONAR_MAX_CM))
    accel = float(getattr(cfg, 'IMU_ACCEL_SCALE', IMU_ACCEL_SCALE))
    gyro = float(getattr(cfg, 'IMU_GYRO_SCALE', IMU_GYRO_SCALE))

    n_sonar = len(sonar_keys(cfg)) if getattr(cfg, 'HAVE_ULTRASONIC', False) else 0
    out = np.zeros(len(values), dtype=np.float32)

    for i, v in enumerate(values):
        if i < n_sonar:
            if v is None or not np.isfinite(v):
                out[i] = 1.0                      # nothing in range == far
            else:
                out[i] = min(float(v), max_cm) / max_cm
        else:
            if v is None or not np.isfinite(v):
                out[i] = 0.0                      # IMU silent == no motion
            else:
                # first three IMU slots are accelerometer, next three gyro
                scale = accel if (i - n_sonar) < 3 else gyro
                out[i] = np.clip(float(v) / scale, -1.0, 1.0)
    return out


def default_sensor_fusion(num_outputs, num_sensor_inputs, input_shape):
    """
    Two-branch network, structurally identical to DonkeyCar's `default_imu`.

    The sensor branch gets three Dense(14) layers of its own BEFORE the merge.
    That matters: concatenating four raw scalars straight onto 100 image
    features would let them be swamped. Giving the branch its own capacity lets
    it build a representation worth merging.

    Input names sort as img_in < sensor_in, which keeps TensorRT's alphabetical
    input ordering aligned with the order used here.
    """
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    sensor_in = Input(shape=(num_sensor_inputs,), name='sensor_in')

    x = core_cnn_layers(img_in, drop)
    x = Dense(100, activation='relu', name='dense_img')(x)
    x = Dropout(.1)(x)

    y = Dense(14, activation='relu', name='dense_sensor_1')(sensor_in)
    y = Dense(14, activation='relu', name='dense_sensor_2')(y)
    y = Dense(14, activation='relu', name='dense_sensor_3')(y)

    z = Concatenate(name='fused')([x, y])
    z = Dense(50, activation='relu', name='dense_fused_1')(z)
    z = Dropout(.1)(z)
    z = Dense(50, activation='relu', name='dense_fused_2')(z)
    z = Dropout(.1)(z)

    outputs = [Dense(1, activation='linear', name=f'out_{i}')(z)
               for i in range(num_outputs)]

    return Model(inputs=[img_in, sensor_in], outputs=outputs,
                 name='sensor_fusion')


class KerasSensorFusion(KerasPilot):
    """
    Image + sensor vector in, steering and throttle out.

    `keys` is stored on the instance so training and inference read the same
    tub fields in the same order.
    """

    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 2,
                 keys: List[str] = None,
                 cfg=None):
        self.num_outputs = num_outputs
        self.cfg = cfg
        self.keys = keys or []
        if not self.keys:
            raise ValueError(
                "KerasSensorFusion needs a non-empty sensor key list. "
                "Set HAVE_ULTRASONIC = True (and/or HAVE_IMU) in myconfig.py.")
        self.num_sensor_inputs = len(self.keys)
        logger.info("KerasSensorFusion: %d sensor inputs %s",
                    self.num_sensor_inputs, self.keys)
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return default_sensor_fusion(num_outputs=self.num_outputs,
                                     num_sensor_inputs=self.num_sensor_inputs,
                                     input_shape=self.input_shape)

    def compile(self):
        self.interpreter.compile(optimizer=self.optimizer, loss='mse')

    def interpreter_to_output(self, interpreter_out) \
            -> Tuple[Union[float, np.ndarray], ...]:
        steering = interpreter_out[0]
        throttle = interpreter_out[1]
        return steering[0], throttle[0]

    def x_transform(
            self,
            record: Union[TubRecord, List[TubRecord]],
            img_processor: Callable[[np.ndarray], np.ndarray]) \
            -> Dict[str, Union[float, np.ndarray]]:
        assert isinstance(record, TubRecord), 'TubRecord expected'
        img_arr = record.image(processor=img_processor)
        raw = [record.underlying.get(k) for k in self.keys]
        return {'img_in': img_arr, 'sensor_in': normalize(self.cfg, raw)}

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        assert isinstance(record, TubRecord), 'TubRecord expected'
        return {'out_0': record.underlying['user/angle'],
                'out_1': record.underlying['user/throttle']}

    def output_shapes(self):
        img_shape = self.get_input_shape('img_in')[1:]
        return ({'img_in': tf.TensorShape(img_shape),
                 'sensor_in': tf.TensorShape([self.num_sensor_inputs])},
                {'out_0': tf.TensorShape([]),
                 'out_1': tf.TensorShape([])})


class SonarDistances:
    """
    Publishes one distance per sensor, in ULTRASONIC_PINS order, by reading an
    UltrasonicArray's state.

    Why an adapter instead of an extra output on the array: `ultrasonic.py`
    belongs to the dormant Phase 2 stack and is left untouched on purpose. Its
    `run_threaded` reports a fixed left/center/right triple, which cannot
    express a four-sensor fan — so Phase 1 reads `array.dist` directly rather
    than changing that contract.

    Reads only; the array's own thread does the writing. Plain dict lookups of
    floats, so no lock is needed.

    BOTH "nothing in range" and "sensor is silent" become SONAR_MAX_CM. That
    keeps the tub free of nulls and infinities (neither survives a JSON round
    trip cleanly) and matches how `normalize` treats them. Sensor health is not
    lost — it stays on `sonar/healthy`, and the array still fails closed via
    `sonar/stop`. The pilot is not the layer that should react to a dead sensor.
    """

    def __init__(self, array, cfg):
        self.array = array
        self.cfg = cfg
        self.names = list(getattr(cfg, 'ULTRASONIC_PINS', {}) or {})
        self.max_cm = float(getattr(cfg, 'SONAR_MAX_CM', SONAR_MAX_CM))

    def run(self):
        out = []
        for name in self.names:
            d = self.array.dist.get(name)
            out.append(self.max_cm if d is None or not np.isfinite(d)
                       else min(float(d), self.max_cm))
        return tuple(out)

    def shutdown(self):
        pass


class SensorVectorizer:
    """
    Drive-time counterpart to `x_transform`: assembles the raw readings the
    vehicle loop provides into the same normalised vector, in the same order.

    Added to the vehicle with `inputs=sensor_keys(cfg)`, so the memory-bus keys
    and the tub keys are literally the same list.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.keys = sensor_keys(cfg)

    def run(self, *values):
        return normalize(self.cfg, list(values))

    def shutdown(self):
        pass


def register_model_type(cfg):
    """
    Teach DonkeyCar's model factory about 'fusion' without editing the package.

    `donkeycar.pipeline.training` does `from donkeycar.utils import
    get_model_by_type`, binding the function into its own module namespace at
    import time — so patching `donkeycar.utils` alone would not affect
    training. Both namespaces are patched here.

    Call once, early, from manage.py and train.py.
    """
    import donkeycar.utils as dk_utils

    original = getattr(dk_utils, '_fusion_original_get_model_by_type', None)
    if original is None:
        original = dk_utils.get_model_by_type
        dk_utils._fusion_original_get_model_by_type = original

    def get_model_by_type(model_type, config):
        if model_type is None:
            model_type = config.DEFAULT_MODEL_TYPE
        base = model_type
        for prefix in ('tflite_', 'tensorrt_', 'fastai_'):
            base = base.replace(prefix, '')

        if base != 'fusion':
            return original(model_type, config)

        from donkeycar.parts.interpreter import KerasInterpreter, TfLite, TensorRT
        interpreter = (TfLite() if 'tflite_' in model_type
                       else TensorRT() if 'tensorrt_' in model_type
                       else KerasInterpreter())
        input_shape = (config.IMAGE_H, config.IMAGE_W, config.IMAGE_DEPTH)
        return KerasSensorFusion(interpreter=interpreter,
                                 input_shape=input_shape,
                                 keys=sensor_keys(config),
                                 cfg=config)

    dk_utils.get_model_by_type = get_model_by_type
    try:
        import donkeycar.pipeline.training as dk_training
        dk_training.get_model_by_type = get_model_by_type
    except ImportError:
        # training deps (tensorflow pipeline) absent on the Pi — drive-only
        pass
