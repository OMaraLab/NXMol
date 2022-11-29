##### multifitted ESP field fitting machinery is located here
# TODO: conversation about moving the indivudual fittin here as well should be had

import numpy as np
from numpy.linalg import lstsq

from chemistry_data_structure.objects.molecular_entity import Molecule3D


# will probably set the fitter up as an object itself

class MoleculeFieldFitter:
    def __init__(self,
                 molecules: list[Molecule3D]):
        """
        The MoleculeFieldFitter is designed to act as an interactive class for both setting up the system
        and querying the results of the solution
        :param molecules: a list of the chemical data structure molecules
        """
        # maintains a list of the current molecules the system
        self._molecules = molecules
        # statuses
        # unsolved = 0
        # solved = 1
        # error = -1
        self._status = 0
        self._constraint_matrix: np.array = None
        self._coeff_matrix: np.array = None
        self._target_vector: np.array = None
        self._transformed_target: np.array = None
        self._transformed_coeff_matrix: np.array = None

    @property
    def status(self):
        return self._status

    @property
    def molecules(self):
        return self._molecules

    def add_molecule(self, molecule: Molecule3D):
        if not isinstance(molecule, Molecule3D):
            raise TypeError('molecule must be of type Molecule3D')
        # TODO: can currently add the same molecule more than once, need to check if this is intended behaviour
        self._molecules.append(molecule)

    def generate_matrices(self):

        # separate out the individual esp and coefficient matrix data
        single_coeffs, single_esps = zip(*[molecule.lsqComponents() for molecule in self._molecules])

        # start generating the system coefficient matrix
        self._coeff_matrix = np.zeros(
            (sum(a.shape[0] for a in single_coeffs), sum(a.shape[1] for a in single_coeffs))
        )
        self._target_vector = np.zeros(
            sum(a.shape[0] for a in single_coeffs)
        )

        # iteratively populate the diagonals of the coefficient matrix with the individual matrices
        # and iteratively fill the full target esp values, since numpy doesn't natively support ragged shapes
        row_count = 0
        col_count = 0
        for mol_index in range(len(self.molecules)):
            shape = single_coeffs[mol_index].shape

            self._coeff_matrix[row_count:row_count + shape[0],
                                col_count:col_count + shape[1]
                              ] = single_coeffs[mol_index]
            self._target_vector[row_count:row_count + shape[0]] = single_esps[mol_index].T[0]
            row_count += shape[0]
            col_count += shape[1]

    def load_constraints(self, constraints: dict):
        """
        Constraints should be of the form:
        {'group': {molecule_obj}: [internal atom indexes]}
        :param constraints:
        :return:
        """
        raise NotImplementedError

    def fit(self):
        """
        Function to generate the assigned partial charges
        :return:
        """
        raise NotImplementedError



