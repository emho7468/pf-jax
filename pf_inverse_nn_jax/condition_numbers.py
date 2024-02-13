import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def construct_matrix_A(n):
    A = 2/3 * np.eye(n)
    A[0,0] -= 1/3
    A[-1,-1] -= 1/3

    sub_diag = 1/6*np.eye(n-1)
    sub_diag = np.vstack((sub_diag, np.zeros(n-1)))
    sub_diag = np.hstack((np.zeros((n,1)), sub_diag))

    A += sub_diag + sub_diag.T
    A /= n

    return A


def construct_matrix_B(n):
    B = 2 * np.eye(n)
    B[0,0] -= 1
    B[-1,-1] -= 1

    sub_diag = -np.eye(n-1)
    sub_diag = np.vstack((sub_diag, np.zeros(n-1)))
    sub_diag = np.hstack((np.zeros((n,1)), sub_diag))

    B += sub_diag + sub_diag.T
    B *= n

    return B


def get_M_for_AC(n, dt, a):
    A = construct_matrix_A(n)
    B = construct_matrix_B(n)
    return (1/dt - 2*a) * A + B


def get_M_for_CH(n, dt, a):
    A = construct_matrix_A(n)
    B = construct_matrix_B(n)
    C1 = np.hstack((1/dt*A, B))
    C2 = np.hstack((2*a*A - B, A))
    return np.vstack((C1,C2))


def heat_plot(dt_lst, n_lst, get_coeff_mat):
    result = np.zeros((len(dt_lst), len(n_lst)))

    for i, dt in enumerate(dt_lst):
        print(f"Now doing dt={dt}")

        for j, n in enumerate(n_lst):
            print(f"Now doing n={n}")
            M = get_coeff_mat(n, dt, a)
            result[i,j] = np.log10(np.linalg.cond(M))

    plt.figure()

    yticks = ["{0:.0e}".format(dt) for dt in dt_lst]

    sns.heatmap(result, vmin=0, vmax=max(5.0, np.max(result)), annot=True, cmap='Blues', xticklabels=n_lst, yticklabels=yticks)
    plt.xlabel("n")
    plt.ylabel("dt")
    plt.title("Condition number log(10)")
    plt.show()


a = 1/0.025**2
dt_lst = [10**(-i) for i in range(3,9)]
n_lst = [10**i for i in range(1,5)]
heat_plot(dt_lst, n_lst, get_M_for_CH)
