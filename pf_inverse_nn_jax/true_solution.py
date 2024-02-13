import glob
import os
import matplotlib.pyplot as plt
import numpy as onp

from jax_am.fem.solver import solver
from jax_am.fem.utils import save_sol

from fem_setup import get_initial_solution, get_true_problem
from configuration import inverse_dir, vtk_dir
from initialize import f, u_min_plot, u_max_plot


def save(saved_arrays, save_results_indices, u, i):
    if i in save_results_indices:
        saved_arrays.append(u)


def solve_and_write(problem_u, u_old, n_dt, save_results_indices):
    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    vtk_path = os.path.join(vtk_dir, f"u_{0:05d}.vtu")
    save_sol(problem_u, u_old, vtk_path) # write solution to file

    saved_arrays = []
    save(saved_arrays, save_results_indices, u_old, 0)

    for i in range(n_dt):
        print(f"\nStep {i + 1}, total step = {n_dt}")

        # Set parameter and solve for T
        problem_u.set_params(u_old)
        u_new = solver(problem_u, use_petsc=False)  # The flag use_petsc=False will use JAX solver - Good for GPU

        vtk_path = os.path.join(vtk_dir, f"u_{i + 1:05d}.vtu")
        save_sol(problem_u, u_new, vtk_path) # write solution to file
        save(saved_arrays, save_results_indices, u_new, i+1)

        u_old = u_new

    return saved_arrays


def solve_true_problem(n_dt, save_results_indices):
    u_initial = get_initial_solution()
    problem_true = get_true_problem()
    return solve_and_write(problem_true, u_initial, n_dt, save_results_indices)


if __name__ == "__main__":
    u_lst = onp.linspace(u_min_plot, u_max_plot, 600)
    f_lst = [f(u) for u in u_lst]

    plt.figure()
    plt.plot(u_lst, f_lst)
    plt.xlabel("u")
    plt.ylabel("Energy")
    plt.title("True energy")
    plt.tight_layout()
    plt.savefig(inverse_dir + "/true.png")
    plt.close()


def find_true_phases():
    u_lst = onp.linspace(-1,1,200)
    f_lst = [f(u) for u in u_lst]

    phase1 = -1
    phase2 = 1
    f_phase1, f_phase2 = 1e6, 1e6
    for u, f_u in zip(u_lst, f_lst):
        if u < 0: # minimum on interval [-1, 0]
            if f_u < f_phase1:
                phase1, f_phase1 = u, f_u
        else: # minimum on interval [0, 1]
            if f_u < f_phase2:
                phase2, f_phase2 = u, f_u
    return phase1, phase2


def plot_true_potential(u_lst):
    f_lst = [f(u) for u in u_lst]
    plt.plot(u_lst, f_lst, label="true potential")
