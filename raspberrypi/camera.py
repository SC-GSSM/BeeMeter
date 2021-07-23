import time
from picamera import PiCamera
import threading
import os
import logging
import numpy as np
import datetime

logger = logging.getLogger("Logger")


class Camera(threading.Thread):
    def __init__(self, mode='series', path='images', resolution=(1280, 720), framerate=24, **kwargs):
        super().__init__()
        if mode in ['series', 'video']:
            self.mode = mode
        self.camera = PiCamera()
        self.camera.framerate = framerate
        self.camera.resolution = resolution    
        self.camera.exposure_mode = 'sports'
        self.camera.iso = 800
        
        now = datetime.datetime.now()
        now_str = now.strftime("%d-%m-%H-%M-%S")

        self.running = True
        self.path = path + '-' + now_str
        self.kwargs = kwargs

        if not os.path.isdir(self.path):
            try:
                os.mkdir(self.path)
            except OSError:
                print("Failed to create new dir")
                self.path = path
        
        self.camera.start_preview()
        time.sleep(2)

    def __del__(self):
        self.camera.close()

    def __str__(self):
        return "Framerate: {0}FPS\nResolution: {1}\nISO: {2}\nExposure Mode: {3}\nCamera Mode: {4}".format(self.camera.framerate, self.camera.resolution, self.camera.iso, self.camera.exposure_mode, self.mode)

    def stop(self):
        self.running = False
        if self.mode == 'video':
            self.camera.stop_recording()
        self.join()

    def series(self, delay=1, video_port=True, season=''):
        filename = os.path.join(self.path, 'image{}'.format(season)) + '{counter:02d}.jpg'
        image_iter = self.camera.capture_continuous(filename, use_video_port=video_port)
        logger.info("Running series for photo."
            t1 = time.time()
            next(image_iter)
            t2 = time.time()
            time.sleep(max(0.0, delay - (t2 - t1)))

    def video(self, extension='h264', season=str(int(time.time()))):
        self.camera.start_recording('video{0}.{1}'.format(season, extension), format=extension)

    def run(self):
        if self.mode == 'series':
            self.series(**self.kwargs)
        elif self.mode == 'video':
            self.video(**self.kwargs)


if 1:
    c = Camera(mode='series', delay=3)
    print(c)
    c.start()
    time.sleep(1)
    c.stop()
else:
    c = Camera(mode='video')
    c.start()
    time.sleep(10)
    c.stop()
