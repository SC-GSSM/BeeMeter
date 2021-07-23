import logging
import threading
import time

import adafruit_dht
import board
import numpy as np
from spidev import SpiDev

import adafruit_bmp
from database import DBConnector

logger = logging.getLogger("Logger")


class Device(threading.Thread):
    def __init__(self, delay=1):
        super().__init__()
        self.q = None
        self.delay = delay
        self.running = True

    def set_q(self, q):
        self.q = q

    def stop(self):
        self.running = False
        self.join()


class MCP3008(Device):
    def __init__(self, bus, device, connected_sensors=None, delay=1, buffer=1, calibration_db=None):
        super().__init__(delay=delay)

        self.bus, self.device = bus, device
        self.spi = SpiDev()
        self.spi.open(self.bus, self.device)
        self.spi.max_speed_hz = 1350000

        self.sensors = [(x - 1, x) for x in range(1, 8, 2)] if not connected_sensors else connected_sensors

        if calibration_db:
            db = DBConnector(calibration_db)
            calibration_data = {x: (y, z) for x, y, z in db.get_calibration_data()}
        else:
            calibration_data = None

        self.gates = [Gate(i, o, self.spi, device, buffer, calibration_data=calibration_data) for i, o in self.sensors]

    def run(self):
        while self.running:
            for gate in self.gates:
                result = gate.scan()
                if result is not None:
                    self.q.put(result)
            time.sleep(self.delay - len(self.gates) * 2 * 0.005)

    def stop(self):
        self.running = False
        self.join()
        for gate in self.gates:  # clear buffer
            if gate.buffer_count != 0:  # buffer is not in queue
                self.q.put(gate.create_output())

    def __del__(self):
        self.spi.close()

    def export_calibration(self, database):
        db = DBConnector(database)
        stor = []
        for gate in self.gates:
            stor.append((gate.gate_id, gate.calibration_in, gate.calibration_out))
        db.insert_many("calibration", stor)


class Gate:
    def __init__(self, sensor_in, sensor_out, spi, device, buffer, detection_error=10, walk_time=3,
                 calibration_data=None):
        self.spi = spi
        self.device = device

        self.sensor_in = sensor_in
        self.sensor_out = sensor_out
        self.gate_id = str(self.device) + str(self.sensor_in) + str(self.sensor_out)

        if calibration_data:
            logger.info("Found calibration data for {}.".format(self.gate_id))
            self.calibration_in = calibration_data[self.gate_id][0]
            self.calibration_out = calibration_data[self.gate_id][0]
        else:
            logger.info("Start new calibration for Gate {}.".format(self.gate_id))
            self.calibration_in = self.calibrate(self.sensor_in) - detection_error
            self.calibration_out = self.calibrate(self.sensor_out) - detection_error
            logger.info(
                "Calibration finished with: IN: {0}, OUT: {1}".format(self.calibration_in, self.calibration_out))

        self.count_out = 0
        self.count_in = 0
        self.buffer_max = buffer
        self.buffer_count = 0
        self.walk_time = walk_time

        self.prev_in = 0
        self.prev_out = 0

        self.t_last_high_in = 0
        self.t_last_high_out = 0
        self.t_last_low_in = 0
        self.t_last_low_out = 0

    def scan(self):
        time_for_gab_between_sensors = 0.2  # seconds
        now = time.time()
        in_analog = self.read(self.sensor_in)
        out_analog = self.read(self.sensor_out)
        in_digital = in_analog < self.calibration_in
        out_digital = out_analog < self.calibration_out

        if not self.prev_in == in_digital:
            self.prev_in = in_digital
            if in_digital:  # bee entered IN sensor range
                self.t_last_high_in = now
                logger.debug("IN 0 -> 1")
            else:  # bee leaves IN sensor range
                self.t_last_low_in = now
                logger.debug("IN 1 -> 0")
                logger.debug("IN - LastHighIN: {0}, LastHighOUT: {1}, LastLowOut: {2}".format(
                    round(now - self.t_last_high_in, 2), round(now - self.t_last_high_out, 2),
                    round(now - self.t_last_low_out, 2)))
                if (now - self.t_last_high_out) < self.walk_time and \
                        (now - self.t_last_high_in) < self.walk_time and \
                        (now - self.t_last_low_out) < time_for_gab_between_sensors:
                    self.count_in += 1

        if not self.prev_out == out_digital:
            self.prev_out = out_digital
            if out_digital:
                logger.debug("OUT 0 -> 1")
                self.t_last_high_out = now
            else:
                self.t_last_low_out = now
                logger.debug("OUT 1 -> 0")
                logger.debug("OUT - LastHighOUT: {0}, LastHighIN: {1}, LastLowIN: {2}".format(
                    round(now - self.t_last_high_out, 2), round(now - self.t_last_high_in, 2),
                    round(now - self.t_last_low_in, 2)))
                if (now - self.t_last_high_in) < self.walk_time and \
                        (now - self.t_last_high_out) < self.walk_time and \
                        (now - self.t_last_low_in) < time_for_gab_between_sensors:
                    self.count_out += 1

        self.buffer_count += 1
        if self.buffer_count < self.buffer_max:
            return None
        else:
            self.buffer_count = 0
            output = self.create_output()
            self.count_out, self.count_in = 0, 0
            return output

    def create_output(self):
        return [0, self.gate_id, self.count_in, self.count_out, int(
            time.time())]

    def read(self, channel):
        adc = self.spi.xfer2([1, (8 + channel) << 4, 0])
        data = ((adc[1] & 3) << 8) + adc[2]
        time.sleep(0.005)
        return data

    def calibrate(self, channel, test=200):
        results = []
        for i in range(test):
            results.append(self.read(channel))
        return np.mean(results) - 2 * np.std(results)


class BMP180(Device):
    """
    Sensor device to measure air pressure and temperature.
    Measurement Range and Accuracy:
        Pressure: 300-1100 hPa +- 2 hPa (at 0-65°C) +- 4.5 hPa (at -20-0°C), Resolution: 0.01
        Temperature: 0-65°C +- 2°C, Resolution: 0.1
    For more details see data sheet:
        https://ae-bst.resource.bosch.com/media/_tech/media/product_flyer/BST-BMP180-FL000.pdf
    """

    def __init__(self, delay=1):
        super().__init__(delay=delay)
        self.device = adafruit_bmp.BMP180()

    def run(self):
        while self.running:
            temperature = self.device.read_temperature()
            pressure = self.device.read_pressure()
            self.q.put([2, int(time.time()), temperature, pressure])
            time.sleep(self.delay)


class DHT11(Device):
    """
    Sensor device to measure humidity and temperature.
    Measurement Range and Accuracy:
        Humidity: 20-90% +- 5% , Resolution: 1%
        Temperature: 0-50°C +- 2°C , Resolution: 1°C
    """

    def __init__(self, pin=board.D4, delay=1):
        super().__init__(delay=delay)
        self.device = adafruit_dht.DHT11(pin)
        if delay < 1:
            raise ValueError("DHT11: Sampling period at intervals should be no less than 1 second.")

    def run(self):
        while self.running:
            try:
                self.device.measure()
                temperature = self.device._temperature
                humidity = self.device._humidity
                if temperature and humidity:
                    self.q.put([1, int(time.time()), temperature, humidity])
            except RuntimeError:
                pass
            time.sleep(self.delay)
