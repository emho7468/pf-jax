import time

import jax.numpy as jnp
import jax
from jax_am.fem.solver import ad_wrapper

from initialize import optimizer
from true_solution import solve_true_problem, save
from fem_setup import get_initial_solution, get_problem
from save_results import plot_energy, plot_results, write_specs


def array_loss(u1, u2):
    loss = jnp.mean((u1 - u2) ** 2)
    return loss


def end_to_end_loss(u, hyperparams, n_dt, save_results_indices, solution_true, savedLosses):

    saved_arrays = []
    save(saved_arrays, save_results_indices, u, 0)

    for i in range(n_dt):
        print(f"\nStep {i + 1}, total step = {n_dt}")

        u_new = fwd_pred((u, hyperparams))
        save(saved_arrays, save_results_indices, u_new, i+1)
        u = u_new

    losses = jnp.array(list(map(array_loss, saved_arrays, solution_true))) # apply array_loss for each index in 'saved_indices'
    loss = jnp.mean(losses)
    savedLosses.append(float(loss.primal)) # cast to plottable object
    return loss


def train_nb_epochs(nb_epochs, trained_epochs, arguments, savedLosses):

    for i in range(nb_epochs):
        print("\n")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print(f"Epoch: {trained_epochs + i}")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("\n")

        grad = loss_grad(*arguments, savedLosses)
        optimizer.step(grad)

    return nb_epochs


def train(n_train_rounds, n_epochs, args, trained_epochs, savedLosses, nn_params, m, v):
    time_trained = 0

    for i in range(n_train_rounds):
        time_start = time.time()
        trained_epochs += train_nb_epochs(n_epochs, trained_epochs, args, savedLosses)
        time_trained += time.time() - time_start
        plot_results(trained_epochs, savedLosses, nn_params)
        write_specs(trained_epochs, time_trained, savedLosses, nn_params, m, v)


def get_saved_results_indices(n_dt):
    return [int(n_dt // 2), n_dt - 1]


# initiate problem
u_initial = get_initial_solution()
problem = get_problem()
fwd_pred = ad_wrapper(problem)

loss_grad = jax.grad(end_to_end_loss, argnums=1)

# Training
n_dt = 20
save_results_indices = [n_dt // 2, n_dt]
solution_true = solve_true_problem(n_dt, save_results_indices)
args = (u_initial, optimizer.nn_params, n_dt, save_results_indices, solution_true)
n_epochs = 10
n_train_rounds = 20
trained_epochs = 0
savedLosses = []

plot_energy(trained_epochs, optimizer.nn_params)
train(n_train_rounds, n_epochs, args, trained_epochs, savedLosses, optimizer.nn_params, optimizer.m, optimizer.v)
