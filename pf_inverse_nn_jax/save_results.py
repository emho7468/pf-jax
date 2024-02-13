import matplotlib.pyplot as plt

import jax.numpy as np

import nn
from true_solution import plot_true_potential, find_true_phases
from configuration import inverse_dir, nn_amp
from initialize import f, u_min_plot, u_max_plot

n_points_plot = 600
phase1, phase2 = find_true_phases()


def shift_potential(potential_array, true_potential_array):
    return potential_array + np.mean(true_potential_array - potential_array)


def plot_loss(trained_nb_epochs, saved_losses):

    plt.figure()
    plt.plot(range(trained_nb_epochs), saved_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.savefig(inverse_dir + "/loss.png")
    plt.close()

    # plot second half of training seperately
    plot_from = trained_nb_epochs // 2
    loss_from = saved_losses[plot_from:]

    plt.figure()
    plt.plot(range(plot_from, plot_from + len(loss_from)), loss_from)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.savefig(inverse_dir + "/second_half_loss.png")
    plt.close()


def plot_energy(trained_nb_epochs, nn_params):

    # Plot NN model
    u_lst = np.linspace(u_min_plot, u_max_plot, n_points_plot)
    u_lst = u_lst[..., np.newaxis]

    final_nn_params_dict = nn.params_list_to_dict(nn_params)
    nn_lst = nn.model.apply(final_nn_params_dict, u_lst)
    nn_lst = nn_amp * nn_lst

    plt.figure()
    plt.plot(u_lst, nn_lst)
    plt.axvline(x=phase1, color='grey', linewidth=0.5)
    plt.axvline(x=phase2, color='grey', linewidth=0.5)
    plt.xlabel("u")
    plt.ylabel("Energy")
    plt.title("Learned potential")
    plt.savefig(inverse_dir + "/energy.png")
    plt.close()

    # Plot comparison with true potential
    f_lst = [f(u) for u in u_lst]
    f_lst = np.array(f_lst)

    plt.figure()
    plt.plot(u_lst, shift_potential(nn_lst, f_lst), label="learned model")
    plot_true_potential(u_lst)
    plt.axvline(x=phase1, color='grey',linewidth=0.5)
    plt.axvline(x=phase2, color='grey',linewidth=0.5)
    plt.legend(fontsize=20)
    plt.xlabel("u", fontsize=20)
    plt.ylabel("Energy", fontsize=20)
    plt.title(f"Potential comparison, epoch: {trained_nb_epochs}", fontsize=20)
    plt.xticks(ticks=[-1, -0.5, 0, 0.5, 1], labels=["-1.0", "-0.5", "0", "0.5", "1.0"], fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(inverse_dir + f"/energy_comp_{trained_nb_epochs}.png")
    plt.close()


def plot_results(trained_nb_epochs, saved_losses, nn_params):
    plot_loss(trained_nb_epochs, saved_losses)
    plot_energy(trained_nb_epochs, nn_params)


def write_specs(trained_nb_epochs, time_trained, saved_losses, nn_params, m, v):
    f = open(inverse_dir + "/specs.txt", "a")
    f.write(f"Epoch {trained_nb_epochs}\n")
    f.write("Time per epoch: {0:.2f} s\n".format(time_trained / trained_nb_epochs))
    f.write(f"Final loss: {saved_losses[-1]}\n")
    f.write(f"Loss evolution: {saved_losses}\n")
    f.write(f"NN params: {nn_params}\n")
    f.write(f"Adam m: {m}\n")
    f.write(f"Adam v: {v}\n")
    f.close()
