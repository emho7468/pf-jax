import os
from pretrained_params import pretrained_norm

# Set problem
available_problems = ["Allen-Cahn", "Cahn-Hilliard"]
available_true_energy_potentials = ["potential 1",
                                    "potential 2",
                                    "potential 3"]

problem = available_problems[1]
true_potential = available_true_energy_potentials[1]

# nn model
n_nodes = 8
init_mu = 0.1 # in case of random initiation
init_std = 1 # in case of random initiation
nn_amp = pretrained_norm * 1.3 # "normalize data" by ballpark figure

# nn training
available_algorithms = ["adam", "grad descent"]
algorithm = available_algorithms[0]
learning_rate = 0.05

# Discretization
Nx, Ny = 40, 40
dt = 1e-9

# create directories
problem_dic = {available_problems[0]: "ac", available_problems[1]: "ch"}
true_potential_dic = {available_true_energy_potentials[0]: "pot 1",
                      available_true_energy_potentials[1]: "pot 2",
                      available_true_energy_potentials[2]: "pot 3"}

crt_file_path = os.path.dirname(__file__)
data_dir = os.path.join(crt_file_path, 'data')
inverse_dir = os.path.join(data_dir, f"{problem_dic[problem]}, {true_potential_dic[true_potential]}, Nx*Ny={Nx}*{Ny}")
vtk_dir = os.path.join(inverse_dir, 'vtk')
os.makedirs(vtk_dir, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "2" # GPU
