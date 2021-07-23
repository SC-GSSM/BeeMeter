import argparse
import time
from picamera import PiCamera
import datetime

now = datetime.datetime.now()
now_str = now.strftime("%d-%m-%H-%M-%S")
filename = "vid_{}.h264".format(now_str)

parser = argparse.ArgumentParser()
parser.add_argument('duration')
parser.add_argument('-fps', '--framerate', action='store', type=int, default=24, 
            help="video framerate")
args = parser.parse_args()

if args.framerate:
    fps = args.framerate
else:
    fps = 24

print("Start recording for {0}s with {1} fps.".format(args.duration, fps))
camera = PiCamera(resolution=(1280,720), framerate=fps)
camera.exposure_mode = 'sports'
camera.iso = 800
print(camera.shutter_speed, camera.exposure_mode, camera.iso)

camera.start_preview()
time.sleep(2)
camera.start_recording(filename)
camera.wait_recording(int(args.duration))
camera.stop_recording()
camera.stop_preview()
print("Finshed recording into: {}".format(filename))
