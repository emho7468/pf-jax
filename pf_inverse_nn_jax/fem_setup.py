import jax.numpy as np
from jax_am.fem.core import FEM
from jax_am.common import rectangle_mesh
from jax_am.fem.generate_mesh import Mesh, get_meshio_cell_type

from configuration import Nx, Ny, dt, nn_amp, problem
from initialize import df
import nn

# Mesh
ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly = 1, 1  # domain size
meshio_mesh = rectangle_mesh(Nx, Ny, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type], ele_type=ele_type)


def boundaries(point):
    x_min = np.isclose(point[0], 0., atol=1e-5)
    x_max = np.isclose(point[0], Lx, atol=1e-5)
    y_min = np.isclose(point[1], 0., atol=1e-5)
    y_max = np.isclose(point[1], Ly, atol=1e-5)
    return x_min | x_max | y_min | y_max


def neumann_bc(point):
    return np.array([0])


def ac_initiate_u(point):
    x = point[0]
    y = point[1]
    return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)


def ch_initiate_v(u):
    return df(u) - 8 * np.pi**2 * u


def ch_initiate_u(point):
    x = point[0]
    y = point[1]
    u = np.cos(2*np.pi*x) * np.cos(2*np.pi*y)
    return [u, ch_initiate_v(u)]


def get_initial_solution():

    if problem == "Allen-Cahn":
        u_initial = [ac_initiate_u(p) for p in mesh.points]
        u_initial = np.array(u_initial)
        u_initial = np.reshape(u_initial, (len(u_initial), 1))
    elif problem == "Cahn-Hilliard":
        u_initial = [ch_initiate_u(p) for p in mesh.points]
        u_initial = np.array(u_initial)
    else:
        raise Exception(f"Problem {problem} not implemented")

    return u_initial


class TrueAllenCahnProblem(FEM):

    def get_mass_map(self):
        """Override base class method.
        """

        def mass_map(u, u_old):
            return (u - u_old) / dt + df(u)

        return mass_map

    def get_tensor_map(self):
        """Override base class method.
        """

        def tensor_map(u_grad):
            return u_grad

        return tensor_map

    def set_params(self, u_old):
        """Override base class method.
        Note that 'neumann' and 'mass' are reserved keywords.
        """
        self.internal_vars['mass'] = [self.convert_from_dof_to_quad(u_old)]


class TrueCahnHilliardProblem(FEM):

    def get_mass_map(self):
        """Override base class method.
        """
        def mass_map(U, U_old):
            return np.array((1/dt*U[0],U[1])) + np.array((-U_old[0] / dt, -df(U[0])))

        return mass_map

    def get_tensor_map(self):
        """Override base class method.
        """

        def tensor_map(U_grad):
            return np.array((U_grad[1], -U_grad[0]))

        return tensor_map

    def set_params(self, params):
        """Override base class method.
        Note that 'neumann' and 'mass' are reserved keywords.
        """
        sol_u_old = params
        self.internal_vars['mass'] = [self.convert_from_dof_to_quad(sol_u_old)]


def get_true_problem():

    if problem == "Allen-Cahn":
        neumann_bc_info_u = [[boundaries], [neumann_bc]]
        fem_problem = TrueAllenCahnProblem(mesh, vec=1, dim=2, ele_type=ele_type, neumann_bc_info=neumann_bc_info_u)
    elif problem == "Cahn-Hilliard":
        neumann_bc_info_u = [[boundaries] * 2, [neumann_bc] * 2]
        fem_problem = TrueCahnHilliardProblem(mesh, vec=2, dim=2, ele_type=ele_type, neumann_bc_info=neumann_bc_info_u)
    else:
        raise Exception(f"Problem {problem} not implemented")

    return fem_problem


class AllenCahnProblem(FEM):

    def get_mass_map(self):
        """Override base class method.
        """

        def mass_map(u, u_old):
            return (u - u_old) / dt + nn_amp * nn.dfdu(self.nn_params_dict, u)

        return mass_map

    def get_tensor_map(self):
        """Override base class method.
        """

        def tensor_map(u_grad):
            return u_grad

        return tensor_map

    def set_params(self, params):
        """Override base class method.
        Note that 'neumann' and 'mass' are reserved keywords.
        """
        sol_u_old, nn_params_list = params
        self.internal_vars['mass'] = [self.convert_from_dof_to_quad(sol_u_old)]
        self.nn_params_dict = nn.params_list_to_dict(nn_params_list)


class CahnHilliardProblem(FEM):

    def get_mass_map(self):
        """Override base class method.
        """
        def mass_map(U, U_old):
            # cast to right dimension
            x = U[0]
            x = np.array(x)
            x = x[..., np.newaxis]
            Z = nn.dfdu(self.nn_params_dict, x)
            return np.array((1/dt*U[0],U[1])) + np.array((-U_old[0] / dt, -nn_amp * Z[0]))

        return mass_map

    def get_tensor_map(self):
        """Override base class method.
        """

        def tensor_map(U_grad):
            return np.array((U_grad[1], -U_grad[0]))

        return tensor_map

    def set_params(self, params):
        """Override base class method.
        Note that 'neumann' and 'mass' are reserved keywords.
        """
        sol_u_old, nn_params_list = params
        self.internal_vars['mass'] = [self.convert_from_dof_to_quad(sol_u_old)]
        self.nn_params_dict = nn.params_list_to_dict(nn_params_list)


def get_problem():

    if problem == "Allen-Cahn":
        neumann_bc_info_u = [[boundaries], [neumann_bc]]
        fem_problem = AllenCahnProblem(mesh, vec=1, dim=2, ele_type=ele_type, neumann_bc_info=neumann_bc_info_u)
    elif problem == "Cahn-Hilliard":
        neumann_bc_info_u = [[boundaries] * 2, [neumann_bc] * 2]
        fem_problem = CahnHilliardProblem(mesh, vec=2, dim=2, ele_type=ele_type, neumann_bc_info=neumann_bc_info_u)
    else:
        raise Exception(f"Problem {problem} not implemented")

    return fem_problem
