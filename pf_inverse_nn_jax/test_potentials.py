import os
import numpy as onp
import matplotlib.pyplot as plt

from jax_interpolate import InterpolatedUnivariateSpline
import jax.numpy as jnp

from configuration import crt_file_path

# hyperparams for potential 1
hyperparams_true = (-0.90, 0.95, 1 / 0.025 ** 2)


def regular_solution_model(u):
    u1, u2, a = hyperparams_true
    return a*((u-u1)*(u-u2))**2/4


def regular_solution_model_diff(u):
    return jnp.sum(regular_solution_model(u)) # sum: workaround to differentiate jax.batch_array object


def read_data(file_name):
    file_path = os.path.join(crt_file_path, file_name)
    data = onp.genfromtxt(file_path, delimiter=", ")

    return data


def fit_spline_from_data(file_name):
    data = read_data(file_name)

    # rescale potential suitably to our problem
    old_max = onp.max(data[:, 1])
    new_max = 200
    data[:, 0] = 2 * data[:, 0] - 1
    data[:, 1] = new_max / old_max * data[:, 1]

    if file_name == "data1.txt": # add extra points to enforce minima at boundaries
        extra_left_points = onp.array([(-1.05, new_max), (-1.01, data[0, 1])])
        extra_right_points = onp.array([(1.01, data[-1, 1]), (1.05, new_max)])
        data = onp.vstack((extra_left_points, data, extra_right_points))

    cubic_spline = InterpolatedUnivariateSpline(data[:, 0], data[: ,1])
    diff_cubic_spline = cubic_spline.derivative
    return cubic_spline, diff_cubic_spline


if __name__ == "__main__":
    # plot test potentials

    def plot(file, label):
        f, df = fit_spline_from_data(file)

        if file == "data2.txt":
            u_min, u_max = -1.0, 1.0
        else:
            u_min, u_max = -1.05, 1.05
        x = onp.linspace(u_min, u_max, 600)
        y = f(x)
        plt.plot(x, y, label=label)


    plt.figure()
    u_min, u_max = -1.0, 1.0
    x = onp.linspace(u_min, u_max, 600)
    y = regular_solution_model(x)
    plt.plot(x, y, label="Potential 1")
    plot("data2.txt", "Potential 2")
    plot("data1.txt", "Potential 3")

    plt.legend()
    plt.xlabel("u")
    plt.ylabel("Energy")
    plt.title("True potential comparison")
    plt.show()
