import argparse
import time
from picamera import PiCamera

parser = argparse.ArgumentParser()
parser.add_argument('duration')
args = parser.parse_args()

camera = PiCamera(resolution=("720p"))
camera.iso = 800


camera.start_preview()
time.sleep(int(args.duration))
camera.stop_preview()

