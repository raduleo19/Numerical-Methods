import matplotlib.pyplot as plt
import numpy as np
import csv


def read_csv(filename):
    with open(filename, 'rt') as csv_file:
        points = []
        buffer = csv.reader(csv_file)
        for row in buffer:
            points.append([float(x) for x in row])
    return points


def neville_interpolation(points, x):
    n = points[0].size
    y = np.zeros((n, n - 1))
    y = np.concatenate((points[1][:, None], y), axis=1)

    for i in range(1, n):
        for j in range(1, i + 1):
            y[i, j] = ((x - points[0][i - j]) * y[i, j - 1] -
                       (x - points[0][i]) * y[i - 1, j - 1]) \
                / (points[0][i] - points[0][i - j])

    return y[n - 1, n - 1]


if(__name__ == "__main__"):
    points = np.array(read_csv("points.csv"))
    interval = np.linspace(min(points[0]), max(points[0]), 5000)
    interpolation = [neville_interpolation(points, x) for x in interval]
    plt.plot(interval, interpolation)
    plt.plot(points[0], points[1], 'o')
    plt.show()
