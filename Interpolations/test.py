import matplotlib.pyplot as plt
import numpy as np
import csv
import time


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

    c = np.linalg.inv(u).dot(v)

    d = np.zeros(n - 1)
    b = np.zeros(n - 1)

    for i in range(n - 1):
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
        b[i] = (a[i + 1] - a[i]) / h[i] - ((2 * c[i] + c[i + 1]) * h[i]) / 3

    return a, b, c, d


def spline_interpolation(points, x, a, b, c, d):
    k = -1
    for i in range(points[0].size - 1):
        if points[0][i] <= x and x <= points[0][i + 1]:
            k = i

    return a[k] + (b[k] * (x - points[0][k])) + (c[k] * (x - points[0][k]) ** 2) + (d[k] * (x - points[0][k]) ** 3)


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
            return points[1][i] + (points[1][i + 1] - points[1][i]) \
                * (x - points[0][i]) / (points[0][i + 1] - points[0][i])


def lagrange_multiplicator(points, x, k):
    numerator = 1
    denominator = 1
    for i in range(points[0].size):
        if i != k:
            numerator *= (x - points[0][i])
            denominator *= (points[0][k] - points[0][i])
    return numerator / denominator


def lagrange_interpolation(points, x):
    y = 0
    for i in range(points[0].size):
        y += lagrange_multiplicator(points, x, i)*points[1][i]
    return y


if(__name__ == "__main__"):
    points = np.array(read_csv("points.csv"))
    interval = np.linspace(min(points[0]), max(points[0]), 5000)
    lin_time = time.time()
    linear_interpolation = [linear_interpolation(points, x) for x in interval]
    print("Linear Interpolation Time: ", time.time() - lin_time)

    spline_time = time.time()
    a, b, c, d = compute_polynoms(points)
    spline_interpolation = [spline_interpolation(points, x, a, b, c, d) for x in interval]
    print("Spline Interpolation Time: ", time.time() - spline_time)

    poly_time = time.time()
    poly_interpolation = [lagrange_interpolation(points, x) for x in interval]
    print("Lagrange Poly Interpolation Time: ", time.time() - poly_time)

    plt.plot(interval, linear_interpolation, label='Linear')
    plt.plot(interval, spline_interpolation, label='Spline')
    plt.plot(interval, poly_interpolation, label='Poly')
    plt.plot(points[0], points[1], 'o', label='Points')
    plt.legend()
    plt.show()
