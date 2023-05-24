import numpy as np
import matplotlib.pyplot as plt


def dy_dx(x, y):
    # y = x^3
    return 3 * x**2

    # y = -e^{-x}
    # return -y


def d2y_dx2(x, y):
    # y = x^3
    return 6 * x

    # y = -e^{-x}
    # return y


def euler(dy_dx, d2y_dx2, x0, y0, x_end, step):
    x = np.arange(x0, x_end + step, step)
    y = np.zeros(len(x))
    y[0] = y0

    for i in range(1, len(x)):
        y[i] = y[i - 1] + dy_dx(x[i - 1], y[i - 1]) * step

    return x, y


def modified_euler(dy_dx, d2y_dx2, x0, y0, x_end, step):
    x = np.arange(x0, x_end + step, step)
    y = np.zeros(len(x))
    y[0] = y0

    for i in range(1, len(x)):
        pred = y[i - 1] + dy_dx(x[i - 1], y[i - 1]) * step
        y[i] = y[i - 1] + 0.5 * step * (dy_dx(x[i - 1], y[i - 1]) + dy_dx(x[i], pred))

    return x, y


def modified_euler_2(dy_dx, d2y_dx2, x0, y0, x_end, step):
    x = np.arange(x0, x_end + step, step)
    y = np.zeros(len(x))
    y[0] = y0

    v = dy_dx(x0, y0)

    for i in range(1, len(x)):
        # pred = v + d2y_dx2(x[i - 1], y[i - 1]) * step
        # v = v + 0.5 * step * (d2y_dx2(x[i - 1], y[i - 1]) + d2y_dx2(x[i], pred))
        # y[i] = y[i - 1] + v * step

        # employ modified euler both on v and y
        pred = v + step * d2y_dx2(x[i - 1], y[i - 1])
        y[i] = y[i - 1] + step * 0.5 * (v + pred)
        v = v + 0.5 * step * (d2y_dx2(x[i - 1], y[i - 1]) + d2y_dx2(x[i], pred))
        # ?
        # v = pred

    return x, y


def leapfrog(dy_dx, d2y_dx2, x0, y0, x_end, step):
    x = np.arange(x0, x_end + step, step)
    y = np.zeros(len(x))
    y[0] = y0

    v = dy_dx(x0, y0)
    a = d2y_dx2(x0, y0)

    for i in range(1, len(x)):
        v += a * step / 2
        y[i] = y[i - 1] + v * step
        a = d2y_dx2(x[i], y[i])
        v += a * step / 2

    return x, y


if __name__ == '__main__':
    # for the calculation of mean relative error, x0 could not be 0
    x0, y0 = 1, 1
    # x0, y0 = 1, -np.exp(-1)
    x_end = 11
    step = 2
    step_truth = 0.01

    rslt = []
    rslt.append(euler(dy_dx, d2y_dx2, x0, y0, x_end, step))
    rslt.append(modified_euler(dy_dx, d2y_dx2, x0, y0, x_end, step))
    # rslt.append(leapfrog(dy_dx, d2y_dx2, x0, y0, x_end, step))
    # rslt.append(modified_euler_2(dy_dx, d2y_dx2, x0, y0, x_end, step))

    plt.figure(dpi=300)
    plt.plot(rslt[0][0], rslt[0][1], 'bo--', label='Euler')
    plt.plot(rslt[1][0], rslt[1][1], 'go--', label='Modified Euler')
    # plt.plot(rslt[2][0], rslt[2][1], 'yo--', label='Leapfrog')
    # plt.plot(rslt[3][0], rslt[3][1], 'co--', label='Modified Euler 2')

    x_truth = np.arange(x0, x_end + step_truth, step_truth)
    plt.plot(x_truth, x_truth ** 3, 'r', label='Exact')
    # plt.plot(x_truth, -np.exp(-x_truth), 'r', label='Exact')
    plt.title('Compare different methods and exact solution for simple ODE\n(y=x^3)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend(loc='best')
    plt.savefig('data/curve.png')

    name = ['Euler', 'Modified Euler', 'Leapfrog', 'Modified Euler 2']

    for i in range(2):
        mean_relative_error = np.mean(np.abs((rslt[i][0] ** 3 - rslt[i][1]) / rslt[i][0] ** 3))
        print(f'Mean relative error of {name[i]}: {mean_relative_error}')
