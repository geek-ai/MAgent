"""dynamic plot class"""
import matplotlib.pyplot as plt


class DynamicPlot:
    def __init__(self, n):
        self.x_data  = []
        self.y_datas = []
        self.lines   = []

        plt.show()
        axes = plt.gca()

        for i in range(n):
            self.y_datas.append([])
            line, = axes.plot(self.x_data, self.y_datas[-1])
            self.lines.append(line)

        self.axes = axes

    def add_point(self, x, ys):
        self.x_data.append(x)

        for i in range(len(ys)):
            self.y_datas[i].append(ys[i])

    def redraw(self):
        for i in range(len(self.lines)):
            self.lines[i].set_xdata(self.x_data)
            self.lines[i].set_ydata(self.y_datas[i])

        self.axes.autoscale(True)
        self.axes.relim()
        x_left, x_right = self.axes.get_xlim()
        y_left, y_right = self.axes.get_ylim()
        self.axes.set_xlim(x_left, (int(x_right) / 100 + 1) * 100)
        self.axes.set_ylim(0, y_right * 1.2)
        plt.draw()
        plt.pause(1e-15)

    def save(self, filename):
        plt.savefig(filename)

