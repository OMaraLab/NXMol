import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix

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

    def setPartialCharges(self, charges: dict, index_type='name'):
        for atom_id, value in charges.items():
            self.get_atom(atom_id, index_type=index_type).partial_charge = value

    def setfitPartialCharges(self, solver='lsq', round_charge=False, **kwargs):
        # todo need a better name for this function
        self.setPartialCharges(
            charges={atom: value[0] for atom, value in
                     zip(self.atoms, self.partialChargeFit(solver=solver, round_charge=round_charge, **kwargs))}
        )

    def partialChargeRMSD(self) -> float:
        distance_pairs = distance_matrix(self._esp_grid_coords, self.atom_coord_matrix)
        # atom coords read in as BOHR, might just convert this to Metres
        A = 1 / distance_pairs  # don't need constant in a.u.
        b = self._esp_grid_charge.reshape(-1, 1)
        partialChargeVector = np.array([a.partial_charge for a in self.atom_objects]).reshape(-1, 1)
        return np.sqrt(1 / self.num_atoms * sum((A @ partialChargeVector - b) ** 2))

    def writePDB(self):
        return

    def writeMol2(self):
        raise NotImplementedError
        Mol2Template = '''
        @<TRIPOS>MOLECULE
*****
 {num_atoms} {num_bonds} 0 0 0
SMALL
GASTEIGER'''

    def calculate_new_hydrogen_coordinates(self, marked_heavy_atom_ids: list):
        """
        Places new hydrogens into simple idealised geometries to later be optimised
        by subsequent MMF/QM calculations.
        TODO: THIS!!!

        """

        # find new neighbours
        first_neighbours = self.first_neighbours

        for heavy_atom_id in marked_heavy_atom_ids:

            # define neighbours of each heavy atom stereocenter
            neighbour_ids = list(first_neighbours[heavy_atom_id])
            h_neighbour_ids = []
            fixed_neighbour_ids = []
            heavy_atom = self.atoms[heavy_atom_id]

            # assumes no radicals
            num_lone_pairs = self.non_bonded_electrons[heavy_atom_id] / 2
            hybridisation = len(neighbour_ids) + num_lone_pairs

            for atom_id in neighbour_ids:
                if self.atoms[atom_id].element == 'H':
                    h_neighbour_ids.append(atom_id)
                else:
                    fixed_neighbour_ids.append(atom_id)

            fixed_neighbour_atoms = [
                atom for atom in self.atoms.values()
                if atom.index in fixed_neighbour_ids
            ]

            num_h_to_place = len(h_neighbour_ids)

            # skip if nothing to be done
            if num_h_to_place == 0:
                continue

            # get atom coordinates
            points = [vector(heavy_atom.coordinates)]
            for atom in fixed_neighbour_atoms:
                points.append(vector(atom.coordinates))

            # blank list of new coordinates
            new_coordinates = []

            # Determine coordinates based on hybridisation
            if hybridisation == LINEAR:
                # work out vector from direct H neighbour to its only other neighbour
                new_vector = points[0] - points[1]
                new_coordinates.append(tuple((points[0] + new_vector).tolist()))

            if hybridisation == TRIGONAL_PLANAR:
                if num_h_to_place == 1:

                    if num_lone_pairs == 1:
                        new_coordinates.append(place_first_hydrogen(points, TRIGONAL_PLANAR_BOND_ANGLE))
                    else:
                        new_coordinates.append(gromos_trigonal_planar_1H(points))
                else:
                    new_coordinates.extend(gromos_trigonal_planar_2H(points))

            if hybridisation == TETRAHEDRAL:

                # Case 1: only 1 hydrogen to add
                if num_h_to_place == 1:

                    # consider lone pairs for bond geometry
                    if num_lone_pairs == 2:
                        new_coordinates.append(place_first_hydrogen(points, TETRAHEDRAL_BOND_ANGLE))
                    elif num_lone_pairs == 1:
                        new_coordinates.extend(gromos_tetrahedral_from_2_vectors(points, 1))
                    else:
                        new_coordinates.append(gromos_tetrahedral_1H(points))

                # Case 2: 2 hydrogens to add
                if num_h_to_place == 2:

                    # consider lone pairs for bond geometry
                    if num_lone_pairs == 1:
                        new_coordinates.append(place_first_hydrogen(points, TETRAHEDRAL_BOND_ANGLE))
                        points.extend(new_coordinates)
                        new_coordinates.extend(gromos_tetrahedral_from_2_vectors(points, 1))
                    else:
                        new_coordinates.extend(gromos_tetrahedral_from_2_vectors(points, 2))

                # Case 3: 3 hydrogens to add
                if num_h_to_place == 3:
                    new_coordinates.extend(gromos_tetrahedral_3H(points))

                # Case 4: methyl... ignore
                if num_h_to_place == 4:
                    continue

            if hybridisation > 4:
                # do not handle above tetrahedral state - this may need to be fixed.
                # print("Relying on Bertrand's random method instead... Hybridisation = {} Atom = {} Element = {}".format(hybridisation, heavy_atom_id, heavy_atom.element))
                continue

            # Update atom coordinates
            for (n, atom_id) in enumerate(h_neighbour_ids):
                self.atoms[atom_id] = Atom(
                    index=atom_id,
                    name='H{}'.format(atom_id),
                    element='H',
                    valence=1,
                    capped=True,
                    coordinates=new_coordinates[n],
                )


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
