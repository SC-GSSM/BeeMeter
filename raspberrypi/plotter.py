import matplotlib.pyplot as plt
from matplotlib.dates import date2num, DateFormatter
import matplotlib.animation as animation
from database import DBConnector
from datetime import datetime
import logging

logger = logging.getLogger("Logger")


class Plotter:
    def __init__(self, delay=1.0, rows=3, cols=1, keep=0):
        self.delay = delay
        self.fig, self.ax = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 12))
        self.db = DBConnector("sensor_data.db", reset=False, same_thread=False)
        self.ani = None
        self.k = keep
        self.lines = []
        self.config()
        for graph in self.ax.reshape(-1):
            line, = graph.plot([], [])
            self.lines.append(line)

    def config(self):
        date_formatter = DateFormatter('%H:%M:%S')
        for graph in self.ax.reshape(-1):
            graph.xaxis.set_major_formatter(date_formatter)
            graph.xaxis_date()
            graph.tick_params(labelrotation=45)
        self.ax[0].set_title('Temperature measure')
        self.ax[0].set_ylabel('Temperature (°C)')
        self.ax[1].set_title('Pressure measure')
        self.ax[1].set_ylabel('Pressure (mPas)')
        self.ax[2].set_title('Humidity measure')
        self.ax[2].set_ylabel('Humidity (%)')
        self.ax[2].set_xlabel('Time (s)')
        plt.tight_layout(pad=3.0)

    def refresh(self, i):
        try:
            time, temp, pressure = list(zip(*self.db.execute_query('SELECT time, temperature, pressure FROM bmp180')))
        except ValueError:
            time, temp, pressure = [], [], []
        try:
            time2, humidity = list(zip(*self.db.execute_query('SELECT time, humidity from dht11')))
        except ValueError:
            time2, humidity = [], []
        time, temp, pressure = time[-self.k:], temp[-self.k:], pressure[-self.k:]
        time2, humidity = time2[-self.k:], humidity[-self.k:]
        logging.info("Plotting data.")
        time = [datetime.fromtimestamp(ts) for ts in time]
        time_nums = date2num(time)
        time2 = [datetime.fromtimestamp(ts) for ts in time2]
        time_nums2 = date2num(time2)

        self.lines[0].set_data(time_nums, temp)
        self.lines[1].set_data(time_nums, pressure)
        self.lines[2].set_data(time_nums2, humidity)

        for graph in self.ax.reshape(-1):
            graph.relim()
            graph.autoscale_view(tight=True, scalex=True, scaley=True)

        return self.lines

    def start(self):
        self.ani = animation.FuncAnimation(self.fig, self.refresh, interval=self.delay * 1000, blit=False)
        plt.show()

    def stop(self):
        self.ani.event_source.stop()


if __name__ == '__main__':
    i = Plotter(delay=1, rows=3, cols=1, keep=500)