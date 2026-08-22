#!/usr/bin/env python3
"""
Scripts to train a keras model using tensorflow.
Basic usage should feel familiar: train.py --tubs data/ --model models/mypilot.h5

Usage:
    train.py [--tubs=tubs] (--model=<model>)
    [--type=(linear|categorical|fusion|inferred|tensorrt_linear|tflite_linear)]
    [--comment=<comment>]

Options:
    -h --help              Show this screen.

`fusion` trains the camera + sensor-vector model (see parts/sensor_pilot.py).
It reads the sonar/IMU columns already stored in the tubs, so the same tubs
train both `linear` and `fusion` — which is what makes the ablation honest.
"""

from docopt import docopt
import donkeycar as dk
from donkeycar.pipeline.training import train
from parts.sensor_pilot import register_model_type


def main():
    args = docopt(__doc__)
    cfg = dk.load_config()
    # must happen before train() resolves the model type
    register_model_type(cfg)
    tubs = args['--tubs']
    model = args['--model']
    model_type = args['--type']
    comment = args['--comment']
    train(cfg, tubs, model, model_type, comment)


if __name__ == "__main__":
    main()
