import jax

from configuration import algorithm, true_potential, available_true_energy_potentials
from optimizer import Optimizer
from pretrained_params import pretrained_params
from test_potentials import fit_spline_from_data, regular_solution_model_diff

# Optimizer
optimizer = Optimizer(algorithm, pretrained_params)

# True energy potential
if true_potential == available_true_energy_potentials[0]:
    f = regular_solution_model_diff
    df = jax.grad(f)
elif true_potential == available_true_energy_potentials[1]:
    f, df = fit_spline_from_data("data1.txt")
elif true_potential == available_true_energy_potentials[2]:
    f, df = fit_spline_from_data("data2.txt")
else:
    raise Exception("Potential " + true_potential + " not implemented!")

if true_potential == available_true_energy_potentials[2]:
    u_min_plot = -1.0
    u_max_plot = 1.0
else:
    u_min_plot = -1.05
    u_max_plot = 1.05
