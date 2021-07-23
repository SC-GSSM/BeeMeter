import threading
import cv2
import numpy as np
import time


class Watcher(threading.Thread):
    def __init__(self, q, fps=24):
        super(Watcher, self).__init__()
        self.running = True
        self.fps = fps
        self.q = q

    def run(self):
        while self.running:
            if not self.q.empty():
                image = self.q.get()
                image = np.frombuffer(image, dtype=np.uint8)
                decimg = cv2.imdecode(image, flags=1)
                cv2.imshow('Stream', decimg)
                cv2.waitKey(24)
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False
        self.join()
