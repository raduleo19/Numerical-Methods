import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import csv


def read_csv(filename):
    with open(filename, 'rt') as csv_file:
        points = []
        buffer = csv.reader(csv_file)
        for row in buffer:
            points.append([float(x) for x in row])
    return points


def compute_polynoms(points):
    a = points[1]
    n = a.size

    h = np.zeros(n - 1)
    for i in range(n - 1):
        h[i] = points[0][i + 1] - points[0][i]

    u = np.zeros((n, n))
    u[0][0] = 1
    u[n - 1][n - 1] = 1
    for i in range(1, n - 1):
        u[i][i - 1] = h[i - 1]
        u[i][i] = 2 * (h[i - 1] + h[i])
        u[i][i + 1] = h[i]

    v = np.zeros(n)
    for i in range(1, n - 1):
        v[i] = (3 * (a[i + 1] - a[i]) / h[i]) - \
            (3 * (a[i] - a[i - 1]) / h[i - 1])

    c = inv(u).dot(v)

    d = np.zeros(n - 1)
    b = np.zeros(n - 1)

    for i in range(n - 1):
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
        b[i] = (a[i + 1] - a[i]) / h[i] - ((2 * c[i] + c[i + 1]) * h[i]) / 3

    return a, b, c, d


def spline_interpolation(points, x):
    a, b, c, d = compute_polynoms(points)
    k = -1
    for i in range(points[0].size - 1):
        if points[0][i] <= x and x <= points[0][i + 1]:
            k = i

    return a[k] + (b[k] * (x - points[0][k])) + (c[k] * (x - points[0][k]) ** 2) + (d[k] * (x - points[0][k]) ** 3)


if(__name__ == "__main__"):
    points = np.array(read_csv("points.csv"))
    interval = np.linspace(min(points[0]), max(points[0]), 5000)
    interpolation = [spline_interpolation(points, x) for x in interval]
    plt.plot(interval, interpolation)
    plt.plot(points[0], points[1], 'o')
    plt.show()
