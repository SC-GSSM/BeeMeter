import socket
import sys
import struct
import time
import io
import numpy as np
import cv2
from queue import Queue
import threading


class Client(threading.Thread):
    def __init__(self, identifier, ip='192.168.178.150'):
        super().__init__()
        self.running = True
        self.client_socket = socket.socket()
        self.client_socket.connect((ip, 8000))

        if not self.identify(identifier):
            self.client_socket.close()
            raise ConnectionError("Connection cannot be identified.")
        print("Success!")

    def identify(self, identifier):
        print("Start Identification")
        self.client_socket.send(str(identifier).encode('utf8'))
        return self.client_socket.recv(32).decode('utf8') == 'OK'

    def stop(self):
        # send signal for shutting down
        self.client_socket.send(struct.pack('<L', 0))
        self.running = False
        self.client_socket.close()
        self.join()


class DataClient(Client):
    def __init__(self, data_queue, identifier=100, ip='192.168.178.147'):
        super().__init__(identifier=identifier, ip=ip)
        self.q = data_queue

    def run(self):
        while self.running:
            if not self.q.empty():
                data = self.q.get().encode('utf8')
                data_size = len(data)
                # first send an unsigned int for the data size
                self.client_socket.send(struct.pack('<L', data_size))
                # then send the data
                self.client_socket.send(data)
                # waiting for an answer before keep going


class CameraClient(Client):
    def __init__(self, fps=24, resolution=(1280, 720), identifier=300, ip='192.168.178.150'):
        super().__init__(identifier=identifier, ip=ip)
        self.camera = CameraStream(framerate=fps, resolution=resolution).start()
        self.send_time = []
        self.last_read = 0
        self.spf = 1.0 / fps

    def run(self):
        avg_datasize = []
        count = 0
        start = time.time()
        while self.running:
            diff = time.time() - self.last_read
            if diff <= self.spf:
                time.sleep(self.spf - diff)
            image = self.camera.read()

            self.last_read = time.time()
            data_str = image.tobytes()
            data_size = len(data_str)
            avg_datasize.append(data_size)
            self.client_socket.send(struct.pack('<L', data_size) + data_str)
            self.send_time.append(time.time() - self.last_read)
            count += 1
        end = time.time()
        t = end - start
        print("TIME: {0:.2f}, FRAMES: {1}, FPS: {2:.2f}".format(t, count, count / t))
        print("Average send time:", self.avg_send_time(0))
        print("Average data size:", np.mean(avg_datasize))
        self.client_socket.close()

    def avg_send_time(self, num=24):
        return sum(self.send_time[-num:]) / len(self.send_time[-num:])

    def stop(self):
        self.camera.stop()
        self.running = False
        self.join()


class CameraStream:
    def __init__(self, framerate=24, resolution=(640, 480), src=0):  # (1280, 470)
        self.cap = cv2.VideoCapture(src)  # activate PiCam with sudo modprobe bcm2835-v4l2
        # camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, framerate)

        self.fps = framerate
        self.running = True
        self.current_fps = None
        self.encode_opt = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        print('Initialize camera...')
        time.sleep(2)

        if self.cap.isOpened():
            _, self.frame = self.cap.read()
        else:
            raise ValueError("Cannot open Stream")

    def start(self):
        threading.Thread(target=self.update).start()
        return self

    def read(self):
        return self.frame

    def update(self):
        current_fps_timer = time.time()
        current_frames = 0
        while self.running:
            ret, image = self.cap.read()
            if ret:
                image = image[250:720, 0:1280]
                image = cv2.resize(image, dsize=(420, 240))
                cv2.imwrite('debug.jpg', image)
                ret, self.frame = cv2.imencode('.jpg', image, self.encode_opt)
                if time.time() - current_fps_timer < 1:
                    current_frames += 1
                else:
                    self.current_fps = current_frames
                    if current_frames < self.fps:
                        print("WARNING: Camera cannot hold FPS, current:", current_frames)
                    current_frames = 0
                    current_fps_timer = time.time()
            else:
                print("ERROR while trying to get frame from the stream.")

    def stop(self):
        self.running = False
        self.cap.release()


class PiCamClient(Client):
    def __init__(self, resolution=(1280, 720), framerate=24, identifier=200, ip='192.168.178.147'):
        super().__init__(identifier=identifier, ip=ip)
        try:
            from picamera import PiCamera
        except ImportError:
            raise ImportError("Cannot detect picamera module. Please install with pip.")
        self.camera = PiCamera()
        self.camera.framerate = framerate
        self.camera.resolution = resolution
        self.camera.exposure_mode = 'sports'
        self.camera.iso = 800

        self.camera.start_preview()
        time.sleep(2)
        print("RDY")

    def run(self):
        connection = self.client_socket.makefile('wb')
        output = FrameBuffer(connection)
        start = time.time()
        print("start recording")
        self.camera.start_recording(output, format='mjpeg')
        print("middle")
        self.camera.wait_recording(10)
        print("Waiting for duration over, stopping...")
        self.camera.stop_recording()
        finish = time.time()
        print("Camera finished, stopping process.")
        connection.close()
        print('Sent %d images in %d seconds at %.2ffps' % (
            output.count, finish - start, output.count / (finish - start)))
        self.stop()


class FrameBuffer:
    def __init__(self, connection):
        self.connection = connection
        self.stream = io.BytesIO()
        self.count = 0

    def write(self, buf):
        print("write")
        # magic number of jpeg
        if buf.startswith(b'\xff\xd8'):
            size = self.stream.tell()
            if size > 0:
                # first write the size of one frame
                self.connection.write(struct.pack('<L', size))
                self.connection.flush()
                self.stream.seek(0)
                # then write the frame
                self.connection.write(self.stream.read(size))
                self.count += 1
                self.stream.seek(0)
        self.stream.write(buf)


if __name__ == '__main__':
    q = Queue(maxsize=512)
    q2 = Queue(maxsize=128)
    q.put("['table', 'test', 1]")
    q.put("['table', 100.0, 3.1]")
    q.put("['table', 'das ist ein test', 1455371]")
    q2.put("das ist das")
    q2.put("haus")
    q2.put("vom guten alten")
    q2.put("Nikolaus")

    import time

    p1 = DataClient(q)
    p2 = DataClient(q2)
    p1.start()
    p2.start()
    time.sleep(1)
    p1.stop()
    p2.stop()

    # camera = CameraClient()
    # # camera = CameraStream()
    # camera.start()
    # time.sleep(10)
    # camera.stop()

    # camera_stream = CameraClientStream()
    # camera_stream.start()
    # time.sleep(10)
    # camera_stream.stop()
