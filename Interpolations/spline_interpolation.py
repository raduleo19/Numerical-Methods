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


def linear_interpolation(points, x):
    for i in range(points[0].size - 1):
        if points[0][i] <= x and x < points[0][i + 1]:
            return points[1][i] + (points[1][i + 1] - points[1][i]) * (x - points[0][i]) / (points[0][i + 1] - points[0][i])

if(__name__ == "__main__"):
    points = np.array(read_csv("points.csv"))
    interval = np.linspace(min(points[0]), max(points[0]), 5000)
    interpolation = [linear_interpolation(points, x) for x in interval]
    plt.plot(interval, interpolation)
    plt.plot(points[0], points[1], 'o')
    plt.show()
