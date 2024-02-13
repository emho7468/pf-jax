import jax.numpy as jnp

from configuration import learning_rate, inverse_dir


class Optimizer:

    def __init__(self, algorithm, nn_params):
        self.nn_params = nn_params

        if algorithm == "grad descent":
            self.step = self.step_grad_descent
        elif algorithm == "adam":
            self.m, self.v = initiate_m_v(nn_params)
            self.i = 1
            self.step = self.get_adam_step()
        else:
            raise Exception(f"Unrecognized optimizer: {algorithm}")
        write_optimizer_to_file(algorithm)

    def get_adam_step(self):

        def step(grad):
            step_adam(self.nn_params, grad, self.m, self.v, self.i)
            self.i += 1

        return step

    def step_grad_descent(self, grad):
        for i, layer in enumerate(self.nn_params):
            grad_layer = grad[i]
            layer[0] = layer[0] - learning_rate * grad_layer[0]
            layer[1] = layer[1] - learning_rate * grad_layer[1]


# Adam parameters
beta1 = 0.9
beta2 = 0.999
eps = 1e-8


def initiate_m_v(nn_params):
    m_init = []
    v_init = []
    for layer in nn_params:
        m_layer_init = []
        v_layer_init = []
        for j in range(2):
            m_layer_init.append(jnp.zeros(layer[j].shape))
            v_layer_init.append(jnp.zeros(layer[j].shape))
        m_init.append(m_layer_init)
        v_init.append(v_layer_init)

    return m_init, v_init


def step_adam(nn_params, grad, m, v, i):
    for param_layer, grad_layer, m_layer, v_layer in zip(nn_params, grad, m, v):
        step_adam_layer(param_layer, grad_layer, m_layer, v_layer, i)


def step_adam_layer(param_layer, grad_layer, m_layer, v_layer, i):
    for j in range(2): # kernel and bias components
        param_layer[j], m_layer[j], v_layer[j] = step_adam_comp(param_layer[j], grad_layer[j], m_layer[j], v_layer[j], i)


def step_adam_comp(param_comp, grad_comp, m_comp, v_comp, i):
    m_new = beta1 * m_comp + (1 - beta1) * grad_comp
    v_new = beta2 * v_comp + (1 - beta2) * jnp.power(grad_comp, 2)
    m_hat = m_new / (1 - beta1 ** i)
    v_hat = v_new / (1 - beta2 ** i)
    param_new = param_comp - learning_rate * m_hat/(jnp.sqrt(v_hat)+eps)
    return param_new, m_new, v_new


def write_optimizer_to_file(algorithm):
    f = open(inverse_dir + "/specs.txt", "a")
    f.write("Optimizer: " + algorithm + "\n")
    f.close()
