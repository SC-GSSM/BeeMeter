import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class Visualizer:
    def __init__(self, path):
        self.data = np.genfromtxt(path, delimiter=',')
        print(self.data)

    def show(self, cumsum=True):
        x = self.data[:, 0]
        inside = self.data[:, 1]
        outside = self.data[:, 2]

        if cumsum:
            inside = np.cumsum(inside)
            outside = np.cumsum(outside)

        ax = plt.figure().gca()
        ax.plot(x, inside, label="In")
        ax.plot(x, outside, label="Out")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        plt.show()

    def show_balance(self):
        self.data = self.data[:50000]
        x = self.data[:, 0]
        y = np.cumsum(self.data[:, 1] - self.data[:, 2])

        ax = plt.figure(figsize=(20, 18)).gca()

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(x, y)
        plt.show()


if __name__ == '__main__':
    Visualizer(path="/home/t9s9/PycharmProjects/BeeMeter/tracking/data/BeeCountShadow.csv").show_balance()
