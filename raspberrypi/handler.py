import time
import threading
from database import DBConnector
from device import Device, MCP3008, BMP180, DHT11
from queue import Queue
import logging
from plotter import Plotter
from camera import Camera

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


class SensorHandler:
    def __init__(self, *args, qsize=128):
        self.running = True
        self.sensors = []
        self.q = Queue(qsize)
        self.write_worker = threading.Thread(target=self.write_db)

        for sensor in args:
            if isinstance(sensor, Device):
                sensor.set_q(self.q)
                self.sensors.append(sensor)
                logger.info("Initialised sensor: {}.".format(sensor))
            elif isinstance(sensor, (Plotter, Camera)):
                self.sensors.append(sensor)
                logger.info("Initialised sensor: {}.".format(sensor))
            else:
                logger.warning("Unknown device of type {}.".format(sensor))

    def start(self):
        self.write_worker.start()
        for sensor in self.sensors:
            sensor.start()

    def stop(self):
        for sensor in self.sensors:
            sensor.stop()
        self.running = False
        self.write_worker.join()

    def write_db(self):
        db = DBConnector("sensor_data.db", same_thread=False, reset=True)
        while self.running or not self.q.empty():
            if self.q.full():
                logger.warning("Queue is full.")
            if not self.q.empty():
                item = self.q.get()
                if item[0] == 0:
                    db.insert_many('counter', [item[1:]])
                    logger.debug("Inserted into counter: {}".format(item))
                elif item[0] == 1:
                    db.insert_many('dht11', [item[1:]])
                    logger.debug("Inserted into DHT11: {}".format(item))
                elif item[0] == 2:
                    db.insert_many('bmp180', [item[1:]])
                    logger.debug("Inserted into BMP180: {}".format(item))
        # delay?


if __name__ == '__main__':
    # instance = SensorHandler(DHT11(delay=5), BMP180(delay=5), Plotter(delay=5, rows=3, cols=1, keep=1000))
    instance = SensorHandler(MCP3008(0, 0, connected_sensors=[(0, 1)], buffer=10, delay=0.1))
    instance.start()
