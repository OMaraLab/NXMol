##### multifitted ESP field fitting machinery is located here
# TODO: conversation about moving the individual fittin here as well should be had

import numpy as np
from numpy.linalg import lstsq
# https://stackoverflow.com/questions/44508561/algorithm-that-numpy-is-using-for-numpy-linalg-lstsq
import pulp  # todo might set this as optional dependency
from scipy.spatial import distance_matrix
from typing import Optional, Any, Union

from chemistry_data_structure.objects.molecular_entity import Molecule3D
from chemistry_data_structure.objects.atom_bond import Atom3D

# unions allow the constraints to be named or not when creating them
SYMMETRY_TYPING = Union[list[dict[Any, list[Any]]], dict[Any, dict[Any, list[Any]]]]
SUM_TYPING = Any  # todo this is a bit of a mess, will work out later
FLAT_SYMMETRY = dict[str, list[tuple[Molecule3D, Atom3D]]]


# todo: maybe use the actual atom objects as keys for the dictionary rather than the internal atom indexes

def _lsq_components(molecule: Molecule3D):
    # these setups are used for both solving methods

    # default load units are bohrs

    # currently unknown units of the surface
    # want rows to correspond to every atom for each grid point

    # assumes a.u.
    distance_pairs = distance_matrix(molecule._esp_grid_coords, molecule.atom_coord_matrix)
    # atom coords read in as BOHR, might just convert this to Metres
    A = 1 / distance_pairs  # don't need constant in a.u.
    b = molecule._esp_grid_charge.reshape(-1, 1)  # turn 1d array into n arrays with 1 element each
    return A, b


def _generate_lsq_matrices(molecules: list[Molecule3D]):
    """
    Method for generating the internal coefficient matrices, and the index lookups capable of tracking the rows/columns
    also works for a single molecule
    """
    # separate out the individual esp and coefficient matrix data
    single_coeffs, single_esps = zip(*[molecule.lsqComponents() for molecule in molecules])
    index_lookup = {}
    index_backlookup = {}

    # start generating the system coefficient matrix
    coeff_matrix = np.zeros(
        (sum(a.shape[0] for a in single_coeffs), sum(a.shape[1] for a in single_coeffs))
    )
    esp_vector = np.zeros(
        (sum(a.shape[0] for a in single_coeffs), 1)
    )

    # iteratively populate the diagonals of the coefficient matrix with the individual matrices
    # and iteratively fill the full target esp values, since numpy doesn't natively support ragged shapes
    row_count = 0
    col_count = 0
    for mol_index in range(len(molecules)):
        shape = single_coeffs[mol_index].shape

        coeff_matrix[row_count:row_count + shape[0],
        col_count:col_count + shape[1]
        ] = single_coeffs[mol_index]
        esp_vector[row_count:row_count + shape[0]] = single_esps[mol_index]

        # at this generation point also want to generate the index lookup and backlookup dictionaries
        # these will be linked to the internal indexes of the atom orders as they are self consistent

        # iterate over the atom indexes (which are not always super consistent)
        # currently use the atom objects as indexes to remove any ambiguity
        # todo need to write a sanity check in the unit tests that confirms this
        for molecule_col_iter, atom_obj in enumerate(molecules[mol_index].atom_objects):
            index_lookup[
                molecules[mol_index], atom_obj
            ] = col_count + molecule_col_iter

            index_backlookup[col_count + molecule_col_iter] = (molecules[mol_index], atom_obj)

        row_count += shape[0]
        col_count += shape[1]

    return index_lookup, index_backlookup, coeff_matrix, esp_vector


def _generate_constraint_matrices(index_lookup: dict,
                                  flat_symmetry_constraints: FLAT_SYMMETRY = None,
                                  flat_sum_constraints: list[tuple[tuple[Molecule3D, int], float]] = None
                                  ) -> Union[np.array, np.array]:
    """
    Internal method for setting up the constraint matrices
    # todo might merge with _generate_lsq_matrices of they see no separate use
    """

    # number of atoms involved in symmetry constraints
    num_symmetry_atoms = sum(
        sum(len(atom_list) for atom_list in group_dict.values())
        for group_dict in symmetry_constraints.values()
    )

    num_constraints = len(sum_constraints) + num_symmetry_atoms - len(symmetry_constraints)

    # need one row for every constraint and one row for every atom
    constraint_matrix = np.zeros((num_constraints, num_atoms))
    constraint_target = np.zeros((self._num_constraints, 1))

    constraint_iter = 0
    for group_name, flattened_group in flat_symmetry_constraints.items():

        # iterate through these pairs to utilise the column location lookup
        # assign the appropriate coefficients using the daisy chain approach
        for mol_atom_pair_1, mol_atom_pair_2 in zip(flattened_group[:-1], flattened_group[1:]):
            constraint_matrix[constraint_iter,
                              index_lookup[mol_atom_pair_1]] = 1.0
            constraint_matrix[constraint_iter,
                              index_lookup[mol_atom_pair_2]] = -1.0

            constraint_iter += 1

    # iterate over the sum constraints, using similar methods for looking up indices
    for group_name, flattened_group in flat_sum_constraints.items():
        # SYMMETRY GROUPS HAVE DIFFERENT STRUCTURE

        for mol_atom_pair in flattened_group['pairs']:
            self._constraint_matrix[constraint_iter, self._index_lookup[mol_atom_pair]] = 1.0
        constraint_target[constraint_iter] = flattened_group['charge']
        constraint_iter += 1

    return constraint_matrix, constraint_target


def _lsq_charge_fit(molecules: list[Molecule3D],
                    flat_symmetry_constraints: list[tuple[Molecule3D, Atom3D]] = None,
                    flat_sum_constraints: list[tuple[tuple[Molecule3D, int], float]] = None) -> np.array:
    """
    Internal function for performing the fitting process and setting up the constraint matrices based off the molecules
    And the preprocessed constraint information
    """
    index_lookup, index_backlookup, coeff_matrix, esp_vector = _generate_lsq_matrices(molecules)

    # q = inv(A.T @ A) @ A.T @ b
    # q = inv(A_star) @ b_star
    q = lstsq(A_star, b_star, rcond=None)

    return {
        'A_star': A_star,
        'b_star': b_star,
        'q_star': q  # this vector also has the Lagrangian's
    }


def flatten_symmetry(molecules: list[Molecule3D],
                     symmetry_groups: SYMMETRY_TYPING,
                     index_type: str = 'name',
                     molecule_map: Optional[dict[Any, Molecule3D]] = None
                     ) -> FLAT_SYMMETRY:
    """
    Function for flattening symmetry constraints of a typical format into a flattened format that uses object references
    Can convert from most predictable numbering schemes/id schemes, as long as they are attached to the molecule objects
    """

    # assume the objects are being used as keys
    if molecule_map is None:
        molecule_map = {molecule: molecule for molecule in molecules}

    flat_symmetry_constraints = {}

    if type(symmetry_groups) == dict:
        pass
    else:
        # if the groups aren't named, generate internal names
        symmetry_groups = {
            f'sym_{jj}': group for jj, group in enumerate(symmetry_groups)
        }

    # iterate through the groups, and using the lookup dictionaries to map the molecule atom index
    # to the filters internal matrix columns

    for group_name, group_dict in symmetry_groups.items():
        # flatten out the groups into key,val pair
        flattened_group = [
            (molecule_map[molecule_id],
             molecule_map[molecule_id].get_atom(atom_index, index_type=index_type))
            for molecule_id, atom_list in group_dict.items()
            for atom_index in atom_list]

        flat_symmetry_constraints[group_name] = flattened_group
    return flat_symmetry_constraints


def flatten_sum(molecules: list[Molecule3D],
                sum_groups: SUM_TYPING,
                index_type: str = 'name',
                molecule_map: Optional[dict[Any, Molecule3D]] = None
                ) -> dict[Any, dict[str, Union[tuple[tuple[Molecule3D, Any], ...], Any]]]:
    """
    Function to flatten sum constraints of any typical format into named lists of molecule, atom pairs
    Useful for checking information later
    """

    # assume the objects are being used as keys
    if molecule_map is None:
        molecule_map = {molecule: molecule for molecule in molecules}

    flat_sum_constraints = {}

    if type(flat_sum_constraints) == dict:
        pass
    else:
        # if the groups aren't named, generate internal names
        sum_groups = {
            f'sum_{jj}': group for jj, group in enumerate(sum_groups)
        }

    # pair the lookup with the value of interest
    for group_name, group_dict in sum_groups.items():
        flattened_group = {'pairs':
                               tuple((molecule_map[molecule_id],
                                      molecule_map[molecule_id].get_atom(atom_index, index_type=index_type))
                                     for molecule_id, atom_list in group_dict[0].items()
                                     for atom_index in atom_list),
                           'charge': group_dict[1]
                           }
        flat_sum_constraints[group_name] = flattened_group

    return flat_sum_constraints


def lsq_partial_charge_fit(molecule: Molecule3D,
                           symmetry_constraints=None,
                           sum_constraints=None,
                           total_charge_constraint=True,
                           ):
    """
    Fitting partial charges to single molecule
    Force the flattening as the functions exist to do it now
    """

    return


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
        self._num_constraints = 0
        self._constraint_matrix: np.array = np.array([])
        self._coeff_matrix: np.array = None
        self._target_vector: np.array = None
        self._constraint_target: np.array = None
        self._transformed_target: np.array = None
        self._transformed_coeff_matrix: np.array = None
        self._solution: np.array = None
        self._solution_round: np.array = None
        self._residuals: np.array = None

        # constraints
        self._flat_symmetry_constraints: dict = None
        self._flat_sum_constraints: list[tuple[tuple[Molecule3D, list[int]], float]] = None

        self._index_lookup: dict = {}
        self._index_backlookup: dict = {}

    @property
    def status(self):
        return self._status

    @property
    def molecules(self):
        return self._molecules

    @property
    def num_atoms(self):
        return self._coeff_matrix.shape[1]

    @property
    def sum_constraints(self):
        return self._flat_sum_constraints

    @property
    def symmetry_constraints(self):
        return self._flat_symmetry_constraints

    def add_molecule(self, molecule: Molecule3D):
        if not isinstance(molecule, Molecule3D):
            raise TypeError('molecule must be of type Molecule3D')
        # TODO: can currently add the same molecule more than once, need to check if this is intended behaviour
        self._molecules.append(molecule)

    def load_constraints(self,
                         symmetry_constraints=None,
                         sum_constraints=None,
                         # index_type: str,
                         ):
        """
        Constraints should be of the form:
        {'group': {molecule_obj}: [internal atom indexes]}
        :param sum_constraints:
        :param symmetry_constraints:
        :return:
        """
        # size of the constraint matrix is given by M+N-G
        # M Molecules (assuming the total charge constraints), N atoms (constrained), G groups

        # https://stackoverflow.com/questions/4391697/find-the-index-of-a-dict-within-a-list-by-matching-the-dicts-value
        # todo Might be lenient with the atom indexing, since the molecules can lookup based on a number of methods
        # todo currently the groups are assumed to be merged, it is known not a trivial thing to implement

        num_constrained_atoms = sum(
            sum(len(atom_list) for atom_list in group_dict.values())
            for group_dict in symmetry_constraints.values()
        )

        self._num_constraints = len(sum_constraints) + num_constrained_atoms - len(symmetry_constraints)

        # need one row for every constraint and one row for every atom
        self._constraint_matrix = np.zeros((self._num_constraints,
                                            self.num_atoms))
        self._constraint_target = np.zeros((self._num_constraints, 1))

        self._flat_sum_constraints = []
        self._flat_symmetry_constraints = {}

        # iterate through the groups, and using the lookup dictionaries to map the molecule atom index
        # to the filters internal matrix columns
        constraint_iter = 0
        for group_name, group_dict in symmetry_constraints.items():
            # flatten out the groups into key,val pair
            flattened_group = [(molecule, atom_index) for molecule, atom_list in group_dict.items()
                               for atom_index in atom_list]

            # iterate through these pairs to utilise the column location lookup
            # assign the appropriate coefficients using the daisy chain approach
            for mol_atom_pair_1, mol_atom_pair_2 in zip(flattened_group[:-1], flattened_group[1:]):
                self._constraint_matrix[constraint_iter,
                                        self._index_lookup[mol_atom_pair_1]] = 1.0
                self._constraint_matrix[constraint_iter,
                                        self._index_lookup[mol_atom_pair_2]] = -1.0

                constraint_iter += 1

            self._flat_symmetry_constraints[group_name] = flattened_group

        # iterate over the sum constraints, using similar methods for looking up indices
        for sum_dict in sum_constraints:
            flattened_group = [(molecule, atom_index) for molecule, atom_list in sum_dict['atoms'].items()
                               for atom_index in atom_list]
            for mol_atom_pair in flattened_group:
                self._constraint_matrix[constraint_iter, self._index_lookup[mol_atom_pair]] = 1.0
            self._constraint_target[constraint_iter] = sum_dict['value']
            constraint_iter += 1

            self._flat_sum_constraints.append((flattened_group, sum_dict['value']))

    def fit(self):
        """
        Function to generate the assigned partial charges
        :return:
        """

        # generate the new matrices used for the fitting procedure (with constraints)

        self._transformed_coeff_matrix = np.zeros(
            (self._num_constraints + self.num_atoms, self._num_constraints + self.num_atoms)
        )

        if self._num_constraints > 0:
            self._transformed_coeff_matrix[:self.num_atoms, :self.num_atoms] = self._coeff_matrix.T @ self._coeff_matrix
            self._transformed_coeff_matrix[self.num_atoms:, :self.num_atoms] = self._constraint_matrix
            self._transformed_coeff_matrix[:self.num_atoms, self.num_atoms:] = self._constraint_matrix.T

            self._transformed_target = np.zeros((self.num_atoms + self._num_constraints, 1))
            self._transformed_target[:self.num_atoms] = self._coeff_matrix.T @ self._target_vector
            self._transformed_target[self.num_atoms:] = self._constraint_target

        else:
            self._transformed_coeff_matrix = self._coeff_matrix
            self._transformed_target = self._target_vector

        self._solution, self._residuals, rank, singular_values = lstsq(self._transformed_coeff_matrix,
                                                                       self._transformed_target,
                                                                       rcond=None)
        self._status = 1
        return self._solution

    def round_post_hoc(self,
                       round_places: int = 3,
                       timeout: int = 10 * 60):
        """
        Rounds the assigned partial charges to an arbitrary decimal place
        Performs a minmax of the residuals between the rounded values and the original
        :return: <np.array> containing the assigned charges ordered by internal atom index
        """
        self._solution_round = np.zeros((self.num_atoms, 1))

        roundProblem = pulp.LpProblem('SystemRounding', pulp.LpMinimize)

        ### Variables
        atoms_vars = pulp.LpVariable.dicts('', range(self.num_atoms), lowBound=-(10 ** round_places - 1),
                                           upBound=(10 ** round_places - 1), cat='Integer')
        abs_vars = pulp.LpVariable.dicts('abs', range(self.num_atoms), lowBound=0)
        max_residual = pulp.LpVariable('max_charge', lowBound=0)

        ### Constraints

        for atom_index_internal in range(self.num_atoms):
            # constructing the absolute value variables of each charge
            # need powers of 10 as setup rounded values variables as integers with units of 10^-(round_places)
            roundProblem += abs_vars[atom_index_internal] >= (
                    atoms_vars[atom_index_internal] - self._solution[atom_index_internal][0] * 10 ** round_places)
            roundProblem += abs_vars[atom_index_internal] >= -(
                    atoms_vars[atom_index_internal] - self._solution[atom_index_internal][0] * 10 ** round_places)

            # set up constraint to define the maximum residual
            roundProblem += max_residual >= abs_vars[atom_index_internal]

        # enforcing the existing constraints on the system

        # daisy chaining the equivalence (symmetry) groups
        for group_name, pairs in self._flat_symmetry_constraints.items():
            for p_1, p_2 in zip(pairs[:-1], pairs[1:]):
                roundProblem += atoms_vars[self._index_lookup[p_1]] == atoms_vars[self._index_lookup[p_2]]

        for pairs, charge in self._flat_sum_constraints:
            roundProblem += sum(atoms_vars[self._index_lookup[p]] for p in pairs) == charge

        # Set the objective as minimising the maximum residual
        roundProblem += max_residual

        solver = pulp.apis.PULP_CBC_CMD(
            maxSeconds=timeout,
            threads=4,
            timeMode="cpu")

        status = roundProblem.solve(solver=solver)

        if roundProblem.status != 1:
            print('ILP rounding failure')
            raise Exception  # todo make own exception

        # slice notation means this line will throw an error if vectors are incorrectly sized
        # as it will attempt to broadcast the values rather than overwrite the variable
        self._solution_round[:] = np.array([pulp.value(a) for a in atoms_vars.values()]).reshape(-1,
                                                                                                 1) * 10 ** -round_places

    def transfer_partial_charges(self):
        """
        Function to transfer the calculated partial charges into the Molecule3D partial charge property
        """
        if self._status != 1:
            print('system not yet fitted')
            raise Exception  # todo make a new exception

        for molecule in self.molecules:
            for atom_internal_index in range(molecule.num_atoms):
                molecule.atom_objects[atom_internal_index].partial_charge = \
                    self._solution[self._index_lookup[molecule, atom_internal_index]][0]
