import typing
import matplotlib.pyplot as plt

import jax
import jax.numpy as np

from flax import linen as nn

from configuration import init_mu, init_std, n_nodes, inverse_dir, nn_amp

random_key = 42 # in case of randomly initializing nn

class SimpleNetwork(nn.Module):
    hidden_sizes : typing.Sequence
    kernel_init : typing.Callable = nn.linear.default_kernel_init

    @nn.compact
    def __call__(self, x, return_activations=False):
        for hd in self.hidden_sizes:
            x = nn.Dense(hd, kernel_init=self.kernel_init, bias_init=self.kernel_init)(x)
            x = nn.sigmoid(x)

        x = nn.Dense(1, kernel_init=self.kernel_init,bias_init=self.kernel_init)(x)
        return x


def init_func(key, shape, dtype):
    return init_mu + init_std*jax.random.normal(key, shape, dtype)


def params_dict_to_list(nn_params_dict):
    nn_params = []

    for layer in nn_params_dict["params"].values():
        nn_params.append([layer["kernel"], layer["bias"]])

    return nn_params


def params_list_to_dict(nn_params_list):
    nn_params_dic = {"params": {}}

    for i, param_set in enumerate(nn_params_list):
        layer_name = f"Dense_{i}"
        layer_dic = {"kernel" : param_set[0], "bias" : param_set[1]}
        nn_params_dic["params"][layer_name] = layer_dic

    return nn_params_dic


# Initialize NN model and its parameters
model = SimpleNetwork(hidden_sizes = (n_nodes,n_nodes), kernel_init = init_func)
params_dict = model.init(jax.random.PRNGKey(random_key), [1])
nn_params = params_dict_to_list(params_dict)
nn_params_dict = params_list_to_dict(nn_params)


def apply(model_params_dict, u):
    return np.sum(model.apply(model_params_dict, u))


# derivative of neural network
dfdu = jax.grad(apply,argnums=1)


if __name__ == "__main__":
    # plot randomly initialized nn
    u_lst = np.linspace(-40, 40, 50)
    u_lst = u_lst[..., np.newaxis]

    nn_lst = model.apply(params_dict, u_lst)
    nn_lst = nn_amp * nn_lst

    plt.figure()
    plt.plot(u_lst, nn_lst)
    plt.xlabel("u")
    plt.ylabel("Energy")
    plt.title("Initial model")
    plt.savefig(inverse_dir + "/initial.png")
    plt.close()
