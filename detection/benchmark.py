import time
import argparse
from pathlib import Path
import numpy as np
import cv2

from detection.model.registry import load_model
from detection.dataset.dataset_generator import Dataset


def create_model(config, weights):
    if Path(config).exists() and Path(weights).exists():
        init_time = time.time()
        model = load_model(config_path=config, weights_path=weights)
        print("{0:<20}: {1:.3f}s".format("Model init", time.time() - init_time))
    else:
        raise TypeError("Weights or config path wrong.")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iterations', type=int, default=100, required=False)
    parser.add_argument('-w', '--warmup', type=int, default=10, required=False)
    parser.add_argument('-e', '--evaluate', type=bool, default=False, required=False)
    parser.add_argument('-b', '--batch', type=int, default=1, required=False)
    parser.add_argument('weights')
    parser.add_argument('config')

    args = parser.parse_args()

    model = create_model(args.config, args.weights)

    images = []
    for image_path in Path("/media/t9s9/UBUNTU 20_0/test_model/test_images").iterdir():
        images.append(cv2.imread(image_path))

    model.predict_timed(images, batch_size=args.batch)
