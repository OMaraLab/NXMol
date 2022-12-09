##### multifitted ESP field fitting machinery is located here
import numpy as np
from numpy.linalg import lstsq
# https://stackoverflow.com/questions/44508561/algorithm-that-numpy-is-using-for-numpy-linalg-lstsq

from scipy.spatial import distance_matrix
from typing import Optional, Any, Union, Sequence
import sys
import warnings

try:
    import gurobipy as gp
    from gurobipy import GRB
except ModuleNotFoundError:
    pass

try:
    import pulp
except ModuleNotFoundError:
    pass

from chemistry_data_structure.objects.molecular_entity import Molecule3D
from chemistry_data_structure.objects.atom_bond import Atom3D

# unions allow the constraints to be named or not when creating them
SYMMETRY_TYPING = Union[list[dict[Any, list[Any]]], dict[Any, dict[Any, list[Any]]]]
SUM_TYPING = Any  # todo this is a bit of a mess, will work out later
FLAT_SYMMETRY = dict[Any,
                     tuple[tuple[Molecule3D, Any], ...]
]
FLAT_SUM = dict[str,
                dict[str,
                     Union[tuple[tuple[Molecule3D, Any], ...], float]
                ]
]

# todo need to write failsafe check for full coverage of inferred molecules


class ConstraintNotSatisfied(Exception):
    pass


def _lsq_components(molecule: Molecule3D) -> tuple[np.array, np.array]:
    """
    function to generate linear formulation of the field fit problem from molecule3D objects
    default load units are bohr

    :param molecule: Single Molecule3D object, must have esp field information populated
    :return: tuple of *index_lookup*

    """

    # assumes a.u.
    distance_pairs = distance_matrix(molecule._esp_grid_coords, molecule.atom_coord_matrix)
    # atom coords read in as BOHR, might just convert this to Metres
    A = 1 / distance_pairs  # don't need constant in a.u.
    b = molecule._esp_grid_charge.reshape(-1, 1)  # turn 1d array into n arrays with 1 element each
    return A, b


def _lsq_components_infer(molecule: Molecule3D) -> tuple[np.array, np.array]:
    """
    Method for generating matrices that allow the incorporation of molecules with no esp grid
    :param molecule:
    :return: placeholder arrays that will be compatible with the current matrix setups, and allow for the inclusion of
    constraints on the included molcules
    """
    # todo this could be incorporated into the above but I want to force users/myself to treat them separately
    return np.zeros((0, molecule.num_atoms)), np.zeros((0, 1))


def _generate_lsq_matrices(molecules: Sequence[Molecule3D],
                           molecules_infer: Optional[Sequence[Molecule3D]] = None
                           ) -> tuple[dict, dict, np.array, np.array]:
    """
    generic function for generating the internal coefficient matrices for a molecule system, and the index lookups capable of
     tracking the rows/columns, also works for a single molecule
    :param molecules: List of Molecule3D objects, must have esp field information populated
    :return: tuple of,

    """
    # separate out the individual esp and coefficient matrix data

    if molecules_infer is None:
        molecules_infer = []

    single_coeffs, single_esps = zip(*(
            [_lsq_components(molecule) for molecule in molecules] +
            [_lsq_components_infer(molecule) for molecule in molecules_infer]
    ))
    molecules = molecules

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
    # todo: don't like this double iteration
    row_count = 0
    col_count = 0

    mol_seq = [*molecules, *molecules_infer]
    for mol_index in range(len(mol_seq)):
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
        for molecule_col_iter, atom_obj in enumerate(mol_seq[mol_index].atom_objects):
            index_lookup[
                mol_seq[mol_index], atom_obj
            ] = col_count + molecule_col_iter

            index_backlookup[col_count + molecule_col_iter] = (mol_seq[mol_index], atom_obj)

        row_count += shape[0]
        col_count += shape[1]

    return index_lookup, index_backlookup, coeff_matrix, esp_vector


def _generate_constraint_matrices(index_lookup: dict,
                                  flat_symmetry_constraints: FLAT_SYMMETRY = None,
                                  flat_sum_constraints: FLAT_SUM = None
                                  ) -> Union[np.array, np.array]:
    """
    Internal method for setting up the constraint matrices
    # todo might merge with _generate_lsq_matrices if they see no separate use
    """

    if flat_symmetry_constraints is None:
        flat_symmetry_constraints = {}
    if flat_sum_constraints is None:
        flat_sum_constraints = {}

    # number of atoms involved in symmetry constraints
    num_symmetry_atoms = sum(
        len(pairs) for pairs in flat_symmetry_constraints.values()
    )

    num_atoms = len(index_lookup)

    num_constraints = len(flat_sum_constraints) + num_symmetry_atoms - len(flat_symmetry_constraints)

    # need one row for every constraint and one row for every atom
    constraint_matrix = np.zeros((num_constraints, num_atoms))
    constraint_target = np.zeros((num_constraints, 1))

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
            constraint_matrix[constraint_iter, index_lookup[mol_atom_pair]] = 1.0
        constraint_target[constraint_iter] = flattened_group['charge']
        constraint_iter += 1

    return constraint_matrix, constraint_target


def _generate_transformed_lsq(coeff_matrix: np.array,
                              esp_vector: np.array,
                              constraint_matrix: np.array,
                              constraint_target: np.array
                              ) -> Union[np.array, np.array]:
    """
    Combines the constraint matrices and standard lsq components into a single constrained lsq problem
    """

    num_constraints = constraint_target.shape[0]  # each row corresponds to a constraint
    num_atoms = coeff_matrix.shape[1]  # each column corresponds to an atom

    # setting up the new coefficient matrix
    transformed_coeff_matrix = np.zeros(
        (num_constraints + num_atoms, num_constraints + num_atoms)
    )
    transformed_coeff_matrix[:num_atoms, :num_atoms] = coeff_matrix.T @ coeff_matrix
    transformed_coeff_matrix[num_atoms:, :num_atoms] = constraint_matrix
    transformed_coeff_matrix[:num_atoms, num_atoms:] = constraint_matrix.T

    # setting up the new target vector
    transformed_target = np.zeros((num_atoms + num_constraints, 1))
    transformed_target[:num_atoms] = coeff_matrix.T @ esp_vector
    transformed_target[num_atoms:] = constraint_target

    return transformed_coeff_matrix, transformed_target


def _lsq_charge_fit(molecules: Sequence[Molecule3D],
                    flat_symmetry_constraints: FLAT_SYMMETRY = None,
                    flat_sum_constraints: FLAT_SUM = None,
                    molecules_infer: Optional[Sequence[Molecule3D]] = None
                    ) -> np.array:
    """
    Internal function for performing the fitting process and setting up the constraint matrices based off the molecules
    And the preprocessed constraint information
    """
    # todo, need to decide if want to switch between assigning partial charges directly, or having the option to
    #  output as a dictionary lookup
    # todo: currently implemented as forcing full coverage of molecules as the current method will assign them something
    #  or may result in failing/non full rank matrices

    if flat_symmetry_constraints is None:
        flat_symmetry_constraints = {}
    if flat_sum_constraints is None:
        flat_sum_constraints = {}
    if molecules_infer is None:
        molecules_infer = tuple()

    index_lookup, index_backlookup, coeff_matrix, esp_vector = _generate_lsq_matrices(molecules=molecules,
                                                                                      molecules_infer=molecules_infer)
    constraint_matrix, constraint_target = _generate_constraint_matrices(index_lookup,
                                                                         flat_symmetry_constraints,
                                                                         flat_sum_constraints)

    transformed_coeff_matrix, transformed_target = _generate_transformed_lsq(coeff_matrix,
                                                                             esp_vector,
                                                                             constraint_matrix,
                                                                             constraint_target)

    assert np.linalg.matrix_rank(transformed_coeff_matrix) ==\
           transformed_target.shape[0], 'matrix is not full rank. If using inferred charge generation this can ' \
                                         'result from not fully covering the inferred molecules'

    solution, residuals, rank, singular_values = lstsq(transformed_coeff_matrix,
                                                       transformed_target,
                                                       rcond=None)

    # assign the partial charges directly to the atoms
    # the solution also contains the Lagrangian's here, so need to only iterate through the first part containing
    # ch
    for atom_iter, charge in enumerate(solution[:len(index_lookup)].T[0]):
        index_backlookup[atom_iter][1].partial_charge = charge

    return solution, residuals, rank, singular_values


def flatten_symmetry(molecules: Sequence[Molecule3D],
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
        flattened_group = tuple(
            (molecule_map[molecule_id],
             molecule_map[molecule_id].get_atom(atom_index, index_type=index_type))
            for molecule_id, atom_list in group_dict.items()
            for atom_index in atom_list)

        flat_symmetry_constraints[group_name] = flattened_group
    return flat_symmetry_constraints


def flatten_sum(molecules: Sequence[Molecule3D],
                sum_groups: SUM_TYPING,
                index_type: str = 'name',
                molecule_map: Optional[dict[Any, Molecule3D]] = None
                ) -> FLAT_SUM:
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


def _gurobi_attach_constraints(model: 'gp.Model',
                               atom_vars: dict,
                               sum_target_scale: float = 1.0,
                               flat_symmetry_constraints: list[tuple[Molecule3D, Atom3D]] = None,
                               flat_sum_constraints: FLAT_SUM = None
                               ) -> tuple[dict, dict]:
    """
    Internal method for attaching the symmetry and sum constraints into a gurobi model for charge fitting
    """
    if 'gurobipy' not in sys.modules:
        raise ModuleNotFoundError('gurobipy not imported')

    if flat_symmetry_constraints is None:
        flat_symmetry_constraints = {}
    if flat_sum_constraints is None:
        flat_sum_constraints = {}

    gb_symmetry_constraints = {}

    # iterate through these pairs to utilise the column location lookup
    # assign the appropriate coefficients using the daisy chain approach
    for group_name, flattened_group in flat_symmetry_constraints.items():
        gb_symmetry_constraints[group_name] = tuple(
            model.addConstr(atom_vars[mol_atom_pair_1] == atom_vars[mol_atom_pair_2])
            for mol_atom_pair_1, mol_atom_pair_2
            in zip(flattened_group[:-1], flattened_group[1:])
        )

    gb_sum_constraints = {}
    # iterate over the sum constraints, using similar methods for looking up indices
    for group_name, flattened_group in flat_sum_constraints.items():
        # SYMMETRY GROUPS HAVE DIFFERENT STRUCTURE
        gb_sum_constraints[group_name] = model.addConstr(
            gp.quicksum(atom_vars[mol_atom_pair]
                        for mol_atom_pair in flattened_group['pairs'])
            == flattened_group['charge'] * sum_target_scale
        )

    # return the lookups for the gurobi constraints based off the constraint names
    return gb_symmetry_constraints, gb_sum_constraints


def _gurobi_init_variables(molecules: Sequence[Molecule3D],
                           verbose: bool = False,
                           round_places: Optional[int] = None
                           ):
    """
    Internal method for using gurobi to assign atomic partial charges
    """
    if 'gurobipy' not in sys.modules:
        raise ModuleNotFoundError('gurobipy not imported')

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", verbose)
    env.start()
    model = gp.Model(env=env)

    index_lookup, index_backlookup, coeff_matrix, esp_vector = _generate_lsq_matrices(molecules)
    # todo don't like this being called twice

    if round_places is not None:
        # effectively sets upper and lower bounds of the charges as 1,-1
        atoms_vars = model.addVars(
            index_lookup.keys(),
            vtype=GRB.INTEGER,
            lb=-(10 ** round_places - 1),
            ub=(10 ** round_places - 1)
        )

        atoms_vars_array = np.array(list(atoms_vars.values())).reshape(-1, 1) / 10 ** float(round_places)

    else:

        atoms_vars = model.addVars(
            index_lookup.keys(),
            vtype=GRB.CONTINUOUS,
            lb=-1.0,
            ub=1.0)
        # todo need to confirm the bounds on these values

        atoms_vars_array = np.array(list(atoms_vars.values())).reshape(-1, 1)

    return model, atoms_vars, atoms_vars_array


def _gurobi_charge_fit(molecules: Sequence[Molecule3D],
                       flat_symmetry_constraints: list[tuple[Molecule3D, Atom3D]] = None,
                       flat_sum_constraints: FLAT_SUM = None,
                       verbose: bool = False,
                       round_places: Optional[int] = None
                       ):
    """
    Internal method for using gurobi to assign atomic partial charges
    """

    if 'gurobipy' not in sys.modules:
        raise ModuleNotFoundError('gurobipy not imported')

    model, atoms_vars, atoms_vars_array = _gurobi_init_variables(molecules,
                                                                 verbose,
                                                                 round_places
                                                                 )

    index_lookup, index_backlookup, coeff_matrix, esp_vector = _generate_lsq_matrices(molecules)

    gb_symmetry_constraints, gb_sum_constraints = _gurobi_attach_constraints(
        model,
        atoms_vars,
        10 ** float(round_places) if round_places is not None else 1.0,
        flat_symmetry_constraints,
        flat_sum_constraints
    )
    Q = coeff_matrix.T @ coeff_matrix
    c = -2 * esp_vector.T @ coeff_matrix
    obj = atoms_vars_array.T @ Q @ atoms_vars_array + c @ atoms_vars_array + esp_vector.T @ esp_vector

    # gurobi can solve quadratic expressions
    model.setObjective(obj.sum())
    # this gives exact same result as lsq

    model.optimize()
    if round_places is not None:
        for key in index_lookup.keys():
            key[1].partial_charge = atoms_vars[key].x * 10 ** - round_places
    else:
        for key in index_lookup.keys():
            key[1].partial_charge = atoms_vars[key].x

    return


def partial_charge_fit(molecules: Sequence[Molecule3D],
                       total_charge_constraint: Optional[dict[Molecule3D, Union[int, None]]],
                       symmetry_constraints: FLAT_SYMMETRY = None,
                       sum_constraints: FLAT_SUM = None,
                       verbose: bool = False,
                       method: str = 'lstsq',
                       round_places: Optional[int] = None,
                       molecules_infer: Optional[Sequence[Molecule3D]] = None
                       ) -> None:
    """


    Top level function for assigning atomic partial charges to Molecule3D objects. Partial charges are assigned to
    the given molecule objects not returned.

    *Methods*
    Methods are passed as string to `method`
    * *lstq* - Least squares method
    * *grb* - Gurobi Quadratic solver method
    * *grb-round* Gurobi Quadratic solver method with simultaneous rounded charge optimisation, requires round_places
    argument

        :param molecules: Sequence of molecule objects to be fitted
        :param total_charge_constraint: Dictionary of total charge targets for each molecule, symmetry constraints are
        then generated and added to `sum_constraints`
        :param symmetry_constraints: flattened symmetry constraint configurations
        :param sum_constraints: flattened sum constraint configuration
        :param verbose: Fitting process flag for ILP methods
        :param method: method for assigning the partial charges, see above
        :param round_places: decimal places to round charges to using method 'grb-round'
        :param molecules_infer: Sequence of molecule objects to be fitted, via coverage of constraints
        :rtype: None
        :return: None


    """
    if round_places is not None and method != 'grb-round':
        raise ValueError(f"round_places={round_places} given but method is not compatible with method '{method}'")

    if molecules_infer is not None and method != 'lstsq':
        raise NotImplementedError('Inference only available for least squares assignment')

    methods = ('lstsq', 'grb', 'grb-round')
    if method not in methods:
        raise ValueError(f"Unknown method '{method}', options are {methods}")

    if total_charge_constraint is not None:
        if sum_constraints is None:
            sum_constraints = {}
        for mol_obj, charge_target in enumerate(total_charge_constraint.items()):
            sum_constraints[('total_charge', mol_obj)] = {'pairs':
                tuple(
                    (molecule, atom_obj) for atom_obj in molecule.atom_objects
                ),
                'charge': charge_target
            }

    if method == 'lstsq':

        solution, residuals, rank, singular_values = _lsq_charge_fit(molecules=molecules,
                                                                     flat_symmetry_constraints=symmetry_constraints,
                                                                     flat_sum_constraints=sum_constraints,
                                                                     molecules_infer=molecules_infer)
    elif method == 'grb':
        _gurobi_charge_fit(molecules=molecules,
                           flat_symmetry_constraints=symmetry_constraints,
                           flat_sum_constraints=sum_constraints,
                           verbose=verbose,
                           round_places=None
                           )
    elif method == 'grb-round':
        _gurobi_charge_fit(molecules=molecules,
                           flat_symmetry_constraints=symmetry_constraints,
                           flat_sum_constraints=sum_constraints,
                           verbose=verbose,
                           round_places=round_places
                           )

    # return solution
    return


def _gurobi_post_hoc_round(molecules: Sequence[Molecule3D],
                           flat_symmetry_constraints: FLAT_SYMMETRY = None,
                           flat_sum_constraints: FLAT_SUM = None,
                           verbose: bool = False,
                           round_places: int = 3
                           ) -> None:
    """

    Rounds the partial charges to specified number of decimal places, by minimising the maximum residual between the
    unrounded and rounded charges.

    Method uses an ILP minmax implementation in Gurobi

    :param molecules:
    :param flat_symmetry_constraints:
    :param flat_sum_constraints:
    :param verbose:
    :param round_places:
    :return:
    """

    if 'gurobipy' not in sys.modules:
        raise ModuleNotFoundError('gurobipy not imported')

    # set up the scaling for the sum constraints
    sum_target_scale = 10 ** float(round_places)

    # all gurobi models are initiated with the same variables for partial charge assignment or rounding
    model, atoms_vars, atoms_vars_array = _gurobi_init_variables(molecules=molecules,
                                                                 verbose=verbose,
                                                                 round_places=round_places
                                                                 )
    # generate the indexing information for the molecule set
    index_lookup, index_backlookup, coeff_matrix, esp_vector = _generate_lsq_matrices(molecules)

    # implement the given constraints in the model
    gb_symmetry_constraints, gb_sum_constraints = _gurobi_attach_constraints(
        model=model,
        atom_vars=toms_vars,
        sum_target_scale=sum_target_scale,
        flat_symmetry_constraints=flat_symmetry_constraints,
        flat_sum_constraints=flat_sum_constraints
    )

    # setup the absolute value variables specific to the minmax problem
    abs_vars = model.addVars(
        index_lookup.keys(),
        vtype=GRB.CONTINUOUS,
        lb=0,
    )
    max_abs = model.addVar(vtype=GRB.CONTINUOUS)

    # enforcing the constraints to initialise the absolute value variables
    for molecule_atom_pair in index_lookup.keys():
        # this method requires the partial charges to already be assigned to the atoms, then the values
        # are accessed directly to calculate the residuals
        model.addConstr(abs_vars[molecule_atom_pair] >= (
                atoms_vars[molecule_atom_pair] - molecule_atom_pair[1].partial_charge * sum_target_scale))
        model.addConstr(abs_vars[molecule_atom_pair] >= -(
                atoms_vars[molecule_atom_pair] - molecule_atom_pair[1].partial_charge * sum_target_scale))
        model.addConstr(max_abs >= abs_vars[molecule_atom_pair])

    # minimise the maximum absolute value deviation from the initial charges
    model.setObjective(max_abs)
    model.optimize()

    # if round_places is not None:
    #     q = np.vectorize(lambda var: var.getValue())(atoms_vars_array)
    # else:
    #     q = np.vectorize(lambda var: var.x)(atoms_vars_array)

    # assign the partial charges to the atoms
    for key in index_lookup.keys():
        key[1].partial_charge = atoms_vars[key].x * 10 ** - round_places

    return


def _pulp_post_hoc_round(molecules: Sequence[Molecule3D],
                         flat_symmetry_constraints: list[tuple[Molecule3D, Atom3D]] = None,
                         flat_sum_constraints: FLAT_SUM = None,
                         verbose: bool = False,
                         round_places: int = 3,
                         timeout: int = 60 * 5
                         ) -> None:
    """

    Rounds the partial charges to specified number of decimal places, by minimising the maximum residual between the
    unrounded and rounded charges.

    Method uses an ILP minmax implementation in pulp

    :param molecules:
    :param flat_symmetry_constraints:
    :param flat_sum_constraints:
    :param verbose:
    :param round_places:
    :param timeout:
    :return:
    """

    if flat_symmetry_constraints is None:
        flat_symmetry_constraints = {}
    if flat_sum_constraints is None:
        flat_sum_constraints = {}

    roundProblem = pulp.LpProblem('RoundingProblem', pulp.LpMinimize)

    index_lookup, index_backlookup, coeff_matrix, esp_vector = _generate_lsq_matrices(molecules)

    atoms_vars = pulp.LpVariable.dicts('',
                                       index_lookup.keys(),
                                       lowBound=-(10 ** round_places - 1),
                                       upBound=(10 ** round_places - 1),
                                       cat='Integer')
    abs_vars = pulp.LpVariable.dicts('abs', index_lookup.keys(), lowBound=0)

    max_abs = pulp.LpVariable('max_abs', lowBound=0)

    # set up variables to represent the absolute value of the difference
    for molecule_atom_pair in index_lookup.keys():
        # absolute values for the objective function
        roundProblem += abs_vars[molecule_atom_pair] >= (
                atoms_vars[molecule_atom_pair] - molecule_atom_pair[1].partial_charge * 10 ** round_places)
        roundProblem += abs_vars[molecule_atom_pair] >= -(
                atoms_vars[molecule_atom_pair] - molecule_atom_pair[1].partial_charge * 10 ** round_places)

        roundProblem += max_abs >= abs_vars[molecule_atom_pair]

    # set up constraints that are loaded with the system
    # currently not implemented a way of exporting the constraints to dict

    # iterate through these pairs to utilise the column location lookup
    # assign the appropriate coefficients using the daisy chain approach
    for group_name, flattened_group in flat_symmetry_constraints.items():
        for mol_atom_pair_1, mol_atom_pair_2 in zip(flattened_group[:-1], flattened_group[1:]):
            roundProblem += atom_vars[mol_atom_pair_1] == atoms_vars[mol_atom_pair_2]

    # iterate over the sum constraints, using similar methods for looking up indices
    for group_name, flattened_group in flat_sum_constraints.items():
        # SYMMETRY GROUPS HAVE DIFFERENT STRUCTURE
        # need to multiply the total charge value to match the new units of the rounded variables
        roundProblem += (sum(atoms_vars[mol_atom_pair] for mol_atom_pair in flattened_group['pairs'])
                         == flattened_group['charge'] * 10 ** float(round_places))

    # for Objective, minimise the maximum deviation
    roundProblem += max_abs

    # print(roundProblem)
    if timeout:
        status = roundProblem.solve(solver=pulp.apis.PULP_CBC_CMD(
            timeLimit=timeout,
            # threads=4,
            timeMode="cpu",
            msg=verbose))
    else:
        status = roundProblem.solve(
            solver=pulp.apis.PULP_CBC_CMD(
                # threads=4,
                timeMode="cpu",
                msg=verbose)
        )  # Solver
    # return np.array([pulp.value(x) for x in atoms_vars.values()]).reshape(-1, 1) / float(10 ** round_places)
    for key in index_lookup.keys():
        key[1].partial_charge = pulp.value(atoms_vars[key]) * 10 ** - round_places
    return


def post_hoc_charge_round(molecules: Sequence[Molecule3D],
                          round_places: int = 3,
                          flat_symmetry_constraints: list[tuple[Molecule3D, Atom3D]] = None,
                          flat_sum_constraints: FLAT_SUM = None,
                          verbose: bool = False,
                          timeout: int = 60 * 5,
                          engine='pulp_cbc'
                          ):
    if engine == 'pulp_cbc':
        _pulp_post_hoc_round(molecules=molecules,
                             flat_symmetry_constraints=flat_symmetry_constraints,
                             flat_sum_constraints=flat_sum_constraints,
                             verbose=verbose,
                             round_places=round_places,
                             timeout=timeout
                             )
    elif engine == 'gurobi':
        _gurobi_post_hoc_round(molecules=molecules,
                               flat_symmetry_constraints=flat_symmetry_constraints,
                               flat_sum_constraints=flat_sum_constraints,
                               verbose=verbose,
                               round_places=round_places
                               )
    else:
        raise ValueError(f'Unknown engine {engine}, options are "gurobi" or "pulp"')
    return


def _symmetry_constraint_charge(flat_symmetry_constraints=FLAT_SYMMETRY,
                                assert_enforced: bool = False) -> dict[Any, float]:
    if assert_enforced:
        for group_name, pairs in flat_symmetry_constraints.items():
            if not np.allclose([p[1].partial_charge for p in pairs], pairs[0][1].partial_charge):
                raise ConstraintNotSatisfied(f'Symmetry group {group_name} not equivalent within tolerance.')

    return {
        group_name: pairs[0][1].partial_charge for group_name, pairs in flat_symmetry_constraints.items()
    }


# class MoleculeFieldFitter:
#     def __init__(self,
#                  molecules: Sequence[Molecule3D]):
#         """
#         The MoleculeFieldFitter is designed to act as an interactive class for both setting up the system
#         and querying the results of the solution
#         :param molecules: a list of the chemical data structure molecules
#         """
#         # maintains a list of the current molecules the system
#         self._molecules = molecules
#         # statuses
#         # unsolved = 0
#         # solved = 1
#         # error = -1
#         self._status = 0
#         self._num_constraints = 0
#         self._constraint_matrix: np.array = np.array([])
#         self._coeff_matrix: np.array = None
#         self._target_vector: np.array = None
#         self._constraint_target: np.array = None
#         self._transformed_target: np.array = None
#         self._transformed_coeff_matrix: np.array = None
#         self._solution: np.array = None
#         self._solution_round: np.array = None
#         self._residuals: np.array = None
#
#         # constraints
#         self._flat_symmetry_constraints: FLAT_SYMMETRY = {}
#         self._flat_sum_constraints: FLAT_SUM = {}
#
#         self._index_lookup: dict[tuple[Molecule3D, Atom3D], int] = {}
#         self._index_backlookup: dict[int, tuple[Molecule3D, Atom3D]] = {}
#
#     @property
#     def status(self):
#         return self._status
#
#     @property
#     def molecules(self):
#         return self._molecules
#
#     @property
#     def num_atoms(self):
#         return self._coeff_matrix.shape[1]
#
#     @property
#     def sum_constraints(self):
#         return self._flat_sum_constraints
#
#     @property
#     def symmetry_constraints(self):
#         return self._flat_symmetry_constraints
#
#     def add_molecule(self, molecule: Molecule3D):
#         """
#
#         :param molecule:
#         """
#         if not isinstance(molecule, Molecule3D):
#             raise TypeError('molecule must be of type Molecule3D')
#         # TODO: can currently add the same molecule more than once, need to check if this is intended behaviour
#         self._molecules.append(molecule)
#
#     def load_constraints(self,
#                          symmetry_constraints=None,
#                          sum_constraints=None,
#                          # index_type: str,
#                          ):
#         """
#         Constraints should be of the form:
#         {'group': {molecule_obj}: [internal atom indexes]}
#         :param sum_constraints:
#         :param symmetry_constraints:
#         :return:
#         """
#         # size of the constraint matrix is given by M+N-G
#         # M Molecules (assuming the total charge constraints), N atoms (constrained), G groups
#
#         # https://stackoverflow.com/questions/4391697/find-the-index-of-a-dict-within-a-list-by-matching-the-dicts-value
#         # todo Might be lenient with the atom indexing, since the molecules can lookup based on a number of methods
#         # todo currently the groups are assumed to be merged, it is known not a trivial thing to implement
#
#         num_constrained_atoms = sum(
#             sum(len(atom_list) for atom_list in group_dict.values())
#             for group_dict in symmetry_constraints.values()
#         )
#
#         self._num_constraints = len(sum_constraints) + num_constrained_atoms - len(symmetry_constraints)
#
#         # need one row for every constraint and one row for every atom
#         self._constraint_matrix = np.zeros((self._num_constraints,
#                                             self.num_atoms))
#         self._constraint_target = np.zeros((self._num_constraints, 1))
#
#         self._flat_sum_constraints = []
#         self._flat_symmetry_constraints = {}
#
#         # iterate through the groups, and using the lookup dictionaries to map the molecule atom index
#         # to the filters internal matrix columns
#         constraint_iter = 0
#         for group_name, group_dict in symmetry_constraints.items():
#             # flatten out the groups into key,val pair
#             flattened_group = [(molecule, atom_index) for molecule, atom_list in group_dict.items()
#                                for atom_index in atom_list]
#
#             # iterate through these pairs to utilise the column location lookup
#             # assign the appropriate coefficients using the daisy chain approach
#             for mol_atom_pair_1, mol_atom_pair_2 in zip(flattened_group[:-1], flattened_group[1:]):
#                 self._constraint_matrix[constraint_iter,
#                                         self._index_lookup[mol_atom_pair_1]] = 1.0
#                 self._constraint_matrix[constraint_iter,
#                                         self._index_lookup[mol_atom_pair_2]] = -1.0
#
#                 constraint_iter += 1
#
#             self._flat_symmetry_constraints[group_name] = flattened_group
#
#         # iterate over the sum constraints, using similar methods for looking up indices
#         for sum_dict in sum_constraints:
#             flattened_group = [(molecule, atom_index) for molecule, atom_list in sum_dict['atoms'].items()
#                                for atom_index in atom_list]
#             for mol_atom_pair in flattened_group:
#                 self._constraint_matrix[constraint_iter, self._index_lookup[mol_atom_pair]] = 1.0
#             self._constraint_target[constraint_iter] = sum_dict['value']
#             constraint_iter += 1
#
#             self._flat_sum_constraints.append((flattened_group, sum_dict['value']))
#
#     def fit(self):
#         """
#         Function to generate the assigned partial charges
#         :return:
#         """
#
#         # generate the new matrices used for the fitting procedure (with constraints)
#
#         self._transformed_coeff_matrix = np.zeros(
#             (self._num_constraints + self.num_atoms, self._num_constraints + self.num_atoms)
#         )
#
#         if self._num_constraints > 0:
#             self._transformed_coeff_matrix[:self.num_atoms, :self.num_atoms] = self._coeff_matrix.T @ self._coeff_matrix
#             self._transformed_coeff_matrix[self.num_atoms:, :self.num_atoms] = self._constraint_matrix
#             self._transformed_coeff_matrix[:self.num_atoms, self.num_atoms:] = self._constraint_matrix.T
#
#             self._transformed_target = np.zeros((self.num_atoms + self._num_constraints, 1))
#             self._transformed_target[:self.num_atoms] = self._coeff_matrix.T @ self._target_vector
#             self._transformed_target[self.num_atoms:] = self._constraint_target
#
#         else:
#             self._transformed_coeff_matrix = self._coeff_matrix
#             self._transformed_target = self._target_vector
#
#         self._solution, self._residuals, rank, singular_values = lstsq(self._transformed_coeff_matrix,
#                                                                        self._transformed_target,
#                                                                        rcond=None)
#         self._status = 1
#         return self._solution
#
#     def round_post_hoc(self,
#                        round_places: int = 3,
#                        timeout: int = 10 * 60):
#         """
#         Rounds the assigned partial charges to an arbitrary decimal place
#         Performs a minmax of the residuals between the rounded values and the original
#         :return: <np.array> containing the assigned charges ordered by internal atom index
#         """
#         self._solution_round = np.zeros((self.num_atoms, 1))
#
#         roundProblem = pulp.LpProblem('SystemRounding', pulp.LpMinimize)
#
#         ### Variables
#         atoms_vars = pulp.LpVariable.dicts('', range(self.num_atoms), lowBound=-(10 ** round_places - 1),
#                                            upBound=(10 ** round_places - 1), cat='Integer')
#         abs_vars = pulp.LpVariable.dicts('abs', range(self.num_atoms), lowBound=0)
#         max_residual = pulp.LpVariable('max_charge', lowBound=0)
#
#         ### Constraints
#
#         for atom_index_internal in range(self.num_atoms):
#             # constructing the absolute value variables of each charge
#             # need powers of 10 as setup rounded values variables as integers with units of 10^-(round_places)
#             roundProblem += abs_vars[atom_index_internal] >= (
#                     atoms_vars[atom_index_internal] - self._solution[atom_index_internal][0] * 10 ** round_places)
#             roundProblem += abs_vars[atom_index_internal] >= -(
#                     atoms_vars[atom_index_internal] - self._solution[atom_index_internal][0] * 10 ** round_places)
#
#             # set up constraint to define the maximum residual
#             roundProblem += max_residual >= abs_vars[atom_index_internal]
#
#         # enforcing the existing constraints on the system
#
#         # daisy chaining the equivalence (symmetry) groups
#         for group_name, pairs in self._flat_symmetry_constraints.items():
#             for p_1, p_2 in zip(pairs[:-1], pairs[1:]):
#                 roundProblem += atoms_vars[self._index_lookup[p_1]] == atoms_vars[self._index_lookup[p_2]]
#
#         for pairs, charge in self._flat_sum_constraints:
#             roundProblem += sum(atoms_vars[self._index_lookup[p]] for p in pairs) == charge
#
#         # Set the objective as minimising the maximum residual
#         roundProblem += max_residual
#
#         solver = pulp.apis.PULP_CBC_CMD(
#             maxSeconds=timeout,
#             threads=4,
#             timeMode="cpu")
#
#         status = roundProblem.solve(solver=solver)
#
#         if roundProblem.status != 1:
#             print('ILP rounding failure')
#             raise Exception  # todo make own exception
#
#         # slice notation means this line will throw an error if vectors are incorrectly sized
#         # as it will attempt to broadcast the values rather than overwrite the variable
#         self._solution_round[:] = np.array([pulp.value(a) for a in atoms_vars.values()]).reshape(-1,
#                                                                                                  1) * 10 ** -round_places
#
#     def transfer_partial_charges(self):
#         """
#         Function to transfer the calculated partial charges into the Molecule3D partial charge property
#         """
#         if self._status != 1:
#             print('system not yet fitted')
#             raise Exception  # todo make a new exception
#
#         for molecule in self.molecules:
#             for atom_internal_index in range(molecule.num_atoms):
#                 molecule.atom_objects[atom_internal_index].partial_charge = \
#                     self._solution[self._index_lookup[molecule, atom_internal_index]][0]
