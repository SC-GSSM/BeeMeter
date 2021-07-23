import socket
# from multiprocessing import Process, Event, Queue, Value
import threading
from queue import Queue
import struct
import io
import numpy as np
import cv2
import logging

logger = logging.getLogger("Logger")
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('debug.log', mode='w')
stream = logging.StreamHandler()
form = logging.Formatter(fmt='%(asctime)s.%(msecs)03d - %(levelname)s - %(threadName)s - %(message)s',
                         datefmt='%H:%M:%S')

fh.setFormatter(form)
stream.setFormatter(form)
logger.addHandler(stream)
logger.addHandler(fh)


class ConnectionHandler(threading.Thread):
    def __init__(self, connection, address, q):
        super().__init__()
        logger.info("Found connection from {}".format(address))

        self.running = True
        self.connection = connection
        self.q = q

    def classify_connection(self):
        connection_type = self.connection.recv(16).decode('utf8')
        if connection_type == '100':
            self.receive = self.run_data
            self.connection.send("OK".encode('utf8'))
            logger.info("Found DataClient")
        elif connection_type == '200':
            self.receive = self.run_camera
            self.connection.send("OK".encode('utf8'))
            logger.info("Found Camera Client")
        elif connection_type == '300':
            self.receive = self.run_camera_cv2
            self.connection.send("OK".encode('utf8'))
            logger.info("Found Camera Client with openCV")
        elif connection_type == 'end':
            logger.info("Found END command.. shutting down")
        else:
            self.connection.close()
            logger.error("The connection could not be assigned. Unknown connection type {}. "
                         "Please try to establish a connection again.".format(connection_type))

    def run(self):
        self.classify_connection()
        self.receive()

    def receive(self):
        pass

    def run_data(self):
        while self.running:
            size = self.recvall(struct.calcsize('<L'))
            if size:
                conv_size = struct.unpack('<L', size)[0]
                if conv_size == 0:
                    break
                data = self.recvall(conv_size).decode('utf8')
                if self.q.full():
                    logging.warning('Queue is full!')
                logger.info(data)
                self.q.put(data)
        self.connection.close()

    def run_camera(self):
        self.connection = self.connection.makefile('rb')
        while self.running:
            image_size = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
            if not image_size:
                print('Steam ends')
                break
            image_stream = io.BytesIO()
            image_stream.write(self.connection.read(image_size))
            image_stream.seek(0)
            print("Found image: Q size:", self.q.qsize())
            self.q.put(image_stream)
        print("Finished stream")
        self.connection.close()

    def run_camera_cv2(self):
        count = 0
        damaged = 0
        while self.running:
            raw_size = self.recvall(struct.calcsize('<L'))
            if raw_size is None:
                logger.info("RECV Images: {0}".format(count))
                logger.info("DMG: {0}".format(damaged))
                return None
            size = struct.unpack('<L', raw_size)[0]
            data = self.recvall(size)
            if not data.startswith(b'\xff\xd8'):
                damaged += 1
                logger.warning("Data could be damaged.")
            else:
                self.q.put(data)
            count += 1
        self.connection.close()

    def recvall(self, data_len):
        data = []
        recved_data = 0
        while recved_data < data_len:
            packet = self.connection.recv(data_len - recved_data)
            if not packet:
                return None
            data.append(packet)
            recved_data += len(packet)
        return b''.join(data)

    def stop(self):
        self.running = False
        self.join()


class Server(threading.Thread):
    def __init__(self, q, listen=5, ip='192.168.178.150'):
        super().__init__()
        self.server_socket = socket.socket()
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((ip, 8000))
        self.server_socket.listen(listen)

        self.running = True
        self.name = "Server"
        self.connections = []
        self.connection_count = 0
        self.q = q

    def run(self):
        while self.running:
            logger.info('Waiting for connection... Active connections {}'.format(self.connection_count))
            connection, address = self.server_socket.accept()
            c = ConnectionHandler(connection, address, self.q)
            self.connection_count += 1
            self.connections.append(c)
            c.start()

    def stop(self):
        self.running = False
        [t.stop() for t in self.connections]
        # sending an end socket to stop waiting for incoming connections
        end_socket = socket.socket()
        end_socket.connect(('192.168.178.150', 8000))
        end_socket.send('end'.encode('utf8'))
        # end all ConnectionHandler
        self.server_socket.close()
        self.join()


if __name__ == '__main__':
    import time
    from watching import Watcher

    q = Queue()
    w = Watcher(q)
    s = Server(q)
    s.start()
    w.start()
    time.sleep(30)
    s.stop()
    w.stop()
