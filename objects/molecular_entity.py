import sys

import networkx as nx
import numpy as np
from numpy.linalg import inv
import pulp

try:
    import gurobipy as gp
    from gurobipy import GRB
except ModuleNotFoundError:
    pass

from chemistry_data_structure.objects.base_objects import _2DChemicalObj, _3DChemicalObj
from chemistry_data_structure.objects.atom_bond import Atom2D, Bond2D, Atom3D, Bond3D, RDKitAtom, RDKitBond
from typing import List, Union


class Molecule2D(_2DChemicalObj):
    """
    Represents the 2D structure of a molecular entity.
    """

    def __init__(self,
                 atoms: List[Atom2D] = None,
                 bonds: List[Union[str, str, Bond2D]] = None,
                 name: str = ''
                 ):

        if atoms is None:
            atoms = []
        if bonds is None:
            bonds = []

        # init super class
        super().__init__(atoms, bonds, name)
        self.attributes = {}
        self.angles = {}
        self.dihedrals = {}
        self.qmProperties = {}

        # RMSD fit

    #
    # def index_map(self, index_target: str, index_input: str, index_input_value: str):
    #     # assumption that all atoms have the same index template
    #     temp_key = list(self._graph._node.keys())[0]
    #     if index_target not in self._graph._node[temp_key]._index:


class Molecule3D(_3DChemicalObj, Molecule2D):

    def __init__(self, atoms: List[Atom3D] = None,
                 bonds: List[Union[str, str, Bond3D]] = None,
                 esp_grid_coords: np.array = None,
                 esp_grid_charge: np.array = None,
                 esp_grid_parameters: dict = None,
                 name: str = ''
                 ):
        if atoms is None:
            atoms = []
        if bonds is None:
            bonds = []

        super().__init__(atoms, bonds, name)
        self._esp_grid_coords = esp_grid_coords
        self._esp_grid_charge = esp_grid_charge
        self._esp_grid_parameters = esp_grid_parameters

    def rmsdFit(self):
        return

    def gurobiPartialChargeFit(self,
                               round_charge: bool = False,
                               round_places: int = 3,
                               total_charge: float = 0.0,
                               verbose=True,
                               minmax=False) -> dict:
        if 'gurobipy' not in sys.modules:
            raise ModuleNotFoundError('gurobipy not imported')
        if minmax:
            raise NotImplementedError('MinMax for gurobi solver NYI')

        A, b = self.lsqComponents()

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", verbose)
        env.start()
        model = gp.Model(env=env)
        if not verbose:
            env.setParam('OutputFlag', 0)

        # model = gp.Model('RoundingProblem')
        atom_names = [a for a in self.atoms]

        if round_charge:
            atoms_vars = model.addVars(range(len(atom_names)), vtype=GRB.INTEGER, name=atom_names,
                                       lb=-(10 ** round_places - 1),
                                       ub=(
                                               10 ** round_places - 1))  # effectively sets upper and lower bounds of the charges as 1,-1
            atoms_vars_array = np.array(list(atoms_vars.values())).reshape(-1, 1) / float(10 ** round_places)

        else:

            atoms_vars = model.addVars(range(len(atom_names)), vtype=GRB.CONTINUOUS, name=atom_names, lb=-1.0,
                                       ub=1.0)

            atoms_vars_array = np.array(list(atoms_vars.values())).reshape(-1, 1)

        tot_charge_const = model.addConstr(gp.quicksum(atoms_vars.values()) == total_charge)
        # symmetry

        Q = A.T @ A
        c = -2 * b.T @ A

        obj = atoms_vars_array.T @ Q @ atoms_vars_array + c @ atoms_vars_array + b.T @ b

        # gurobi can solve quadratic expressions
        model.setObjective(obj.sum())
        # this gives exact same result as lsq

        # print(roundProblem)
        model.optimize()
        if round_charge:
            q = np.vectorize(lambda var: var.getValue())(atoms_vars_array)
        else:
            q = np.vectorize(lambda var: var.x)(atoms_vars_array)

        results = {
            'q': q,
            'charge_vars': atoms_vars,
            'total_charge_const': tot_charge_const,
            'model': model

        }

        return results

    def setPartialCharges(self, charges: dict, index_type='name'):
        for atom_id, value in charges.items():
            self.get_atom(atom_id, index_type=index_type).partial_charge = value

    def setfitPartialCharges(self, solver='lsq', round_charge=False, **kwargs):
        # todo need a better name for this function
        self.setPartialCharges(
            charges={atom: value[0] for atom, value in
                     zip(self.atoms, self.partialChargeFit(solver=solver, round_charge=round_charge, **kwargs))}
        )

    def partialChargeFit(self, solver='lsq',
                         round_charge=False,
                         total_charge: int = 0,
                         round_places: int = 3,
                         timeout: int = 60 * 5,
                         minmax: bool = False,
                         verbose: bool = False):
        if self._esp_grid_charge is None or self._esp_grid_coords is None:
            raise AttributeError('No esp grid found')

        if solver == 'lsq':
            q = self.lsqPartialChargeFit(
                total_charge=total_charge,
            )['q_star']
            if not round_charge:
                return q[:-1]
            else:
                roundProblem = pulp.LpProblem('RoundingProblem', pulp.LpMinimize)
                atom_names = [a for a in self.atoms]

                atoms_vars = pulp.LpVariable.dicts('', range(len(atom_names)), lowBound=-(10 ** round_places - 1),
                                                   upBound=(10 ** round_places - 1), cat='Integer')
                abs_vars = pulp.LpVariable.dicts('abs', range(len(atom_names)), lowBound=0)
                if minmax:
                    max_residual = pulp.LpVariable('max_charge', lowBound=0)

                # set up variables to represent the absulte value of the difference
                for a in range(self.num_atoms):
                    # absolute values for the objective function
                    # atoms_vars[a['unique']].setInitialValue(int(a['fit_result']['charge'][0]*10**round_places))
                    roundProblem += abs_vars[a] >= (
                            atoms_vars[a] - q[a][0] * 10 ** round_places)
                    roundProblem += abs_vars[a] >= -(
                            atoms_vars[a] - q[a][0] * 10 ** round_places)
                    # all atoms in group have same charge
                    # need the length, i.e. if there are 3 atoms in the group the residual total is actually 3 times that
                    # constrain the maximum value (although this hopefully shouldn't matter
                    # roundProblem += atoms_vars[a['unique']] <= (10**round_places - 1) # e.g. want to round to 4 decimal places gives max value of 9999
                    if minmax:
                        # for Objective, minimise the maximum deviation
                        roundProblem += max_residual >= abs_vars[a]

                roundProblem += sum(atoms_vars.values()) == total_charge

                if minmax:
                    # minimise the maxiumum residual
                    roundProblem += max_residual
                else:
                    # minimise sum of the residuals
                    roundProblem += sum(abs_vars.values())

                # print(roundProblem)
                if timeout:
                    status = roundProblem.solve(solver=pulp.apis.PULP_CBC_CMD(
                        timeLimit=timeout,
                        threads=4,
                        timeMode="cpu",
                        msg=verbose))
                else:
                    status = roundProblem.solve(
                        solver=pulp.apis.PULP_CBC_CMD(threads=4,
                                                      timeMode="cpu",
                                                      msg=verbose)
                    )  # Solver
                return np.array([pulp.value(x) for x in atoms_vars.values()]).reshape(-1, 1) / float(10 ** round_places)

        elif solver == 'gurobi':
            return self.gurobiPartialChargeFit(round_charge=round_charge,
                                               total_charge=total_charge,
                                               verbose=verbose)['q']

    def partialChargeRMSD(self) -> float:
        A, b = self.lsqComponents()
        partialChargeVector = np.array([a.partial_charge for a in self.atom_objects]).reshape(-1,1)
        return np.sqrt(1/self.num_atoms * sum((A @ partialChargeVector - b)**2))

    def writePDB(self):
        return


class RDKitMolecule(_3DChemicalObj):

    def __init__(self,
                 atoms: List[RDKitAtom] = None,
                 bonds: List[RDKitBond] = None,
                 name: str = ''
                 ):
        super().__init__(atoms, bonds, name)

    def __repr__(self):
        return 'RDKitMol'

    def GetNumAtoms(self):
        return

    def GetAtoms(self):
        return

    def GetBonds(self):
        return


if __name__ == "__main__":
    # testing

    molecule = Molecule3D()
    # molecule.add_edge()
    molecule.graph.add_node(1)
    molecule.graph.add_node(2)
    molecule.graph.add_edge(2, 1)
    print(molecule.atoms)
    print(molecule.graph.edges)
    print(molecule.graph.nodes)
    print(molecule.bonds)

    print("Atoms: ")
    print(molecule.graph[1])
    print(molecule.graph[2])

    print("Bonds: ")
    print(molecule.graph[1][2])
    print(molecule.graph[2][1])
    e = [('a', 'b', 0.3), ('b', 'c', 0.9), ('a', 'c', 0.5), ('c', 'd', 1.2)]
    molecule.graph.add_weighted_edges_from(e)
    print(nx.dijkstra_path(molecule.graph, 'a', 'd'))

    test = Molecule2D([Atom2D('C1', 'C')])
