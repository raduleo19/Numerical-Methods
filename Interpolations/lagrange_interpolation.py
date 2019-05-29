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


def lagrange_multiplicator(points, x, k):
    numerator = 1
    denominator = 1
    for i in range(points[0].size):
        if i != k:
            numerator *= (x - points[0][i])
            denominator *= (points[0][k] - points[0][i])
    return numerator / denominator


def lagrange_poly(points, x):
    y = 0
    for i in range(points[0].size):
        y += lagrange_multiplicator(points, x, i)*points[1][i]
    return y

if(__name__ == "__main__"):
    points = np.array(read_csv("points.csv"))
    interval = np.linspace(min(points[0]), max(points[0]), 5000)
    interpolation = [lagrange_poly(points, x) for x in interval]
    plt.plot(interval, interpolation)
    plt.plot(points[0], points[1], 'o')
    plt.show()
