import networkx as nx
import numpy as np
from numpy.linalg import inv
from scipy.spatial import distance_matrix

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
        super().__init__(atoms, bonds,  name)
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

    def partialChargeFit(self, method='lsq'):
        if not (self._esp_grid_charge and self._esp_grid_coords):
            raise AttributeError('No esp grid found')
        if method == 'lsq':
            four_pi_eps_rcp = 138.9354  # // kJ * nm * mol * e ^ -2 *taken directly from field fit) (units.h)
            NM_per_BOHR = 0.0529177
            # default load units are bohrs

            # currently unkown units of the surface
            # want rows to correspond to every atom for each grid point

            # assumes a.u.
            distance_pairs = distance_matrix(self._esp_grid_coords, self.atom_coord_matrix)
            # atom coords read in as BOHR, might just convert this to Metres
            A = 1/distance_pairs  # don't need constant in a.u.

            # assmuning units of esp surface are kj*bohr/q
            b = self._esp_grid_charge.reshape(-1, 1)  # turn 1d array into n arrays with 1 element each
            q = inv(A.T @ A) @ A.T @ b

            return q





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
    #molecule.add_edge()
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

