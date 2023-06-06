import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix
from io import StringIO
from operator import itemgetter
from chemistry_data_structure.helpers.chem import LINEAR, TRIGONAL_PLANAR, \
    TRIGONAL_PLANAR_BOND_ANGLE, TETRAHEDRAL, TETRAHEDRAL_BOND_ANGLE

try:
    import gurobipy as gp
    from gurobipy import GRB
except ModuleNotFoundError:
    pass

from chemistry_data_structure.objects.base_objects import _2DChemicalObj, _3DChemicalObj
from chemistry_data_structure.objects.atom_bond import Atom2D, Bond2D, Atom3D, Bond3D, RDKitAtom, RDKitBond
from chemistry_data_structure.helpers.vector_calculations import place_first_hydrogen, \
    gromos_tetrahedral_1H, gromos_trigonal_planar_1H, gromos_trigonal_planar_2H, gromos_tetrahedral_from_2_vectors, \
    gromos_tetrahedral_3H
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

    def pdbStr(self) -> str:
        """
        Returns the PDB representation of this 3D Molecule object as a string.
        """
        from chemistry_data_structure.parsing.pdb import PDB_TEMPLATE, pdb_conect_line
        io = StringIO()

        ordered_atoms = sorted(self.atoms.values(), key=lambda atom: atom.get_index())

        pdb_ids = dict(
            zip(
                [atom.get_index() for atom in ordered_atoms],
                range(1, len(ordered_atoms) + 1),
            ),
        )

        for (atom_index, pdb_id) in sorted(pdb_ids.items(), key=itemgetter(1)):
            atom = self.atoms[atom_index]
            coordinates = atom.coordinates

            if coordinates is None:
                coordinates = (
                    1.3 * pdb_id,
                    0.1 * (-1 if pdb_id % 2 == 0 else +1),
                    0.1 * (pdb_id % 5),
                )

            try:
                print(PDB_TEMPLATE.format(
                    'HETATM',
                    pdb_id,
                    (atom.element.title() + str(atom_index))[:4],
                    'R',
                    '',
                    pdb_id,
                    *coordinates,
                    '',
                    '',
                    atom.element.title(),
                    '',
                ), file=io)
            except:
                raise Exception(pdb_id, atom.element, coordinates)

        for (atom_index, pdb_id) in sorted(pdb_ids.items(), key=itemgetter(1)):
            print(
                pdb_conect_line(
                    [pdb_id]
                    +
                    [pdb_ids[list(frozenset(bond) - frozenset([atom_index]))[0]] for bond in self.bonds if atom_index in bond]
                ),
                file=io,
            )

        return io.getvalue()

    def calculateNewHydrogenCoordinates(self, marked_heavy_atom_ids: list):
        """
        Places new hydrogens into simple idealised geometries to later be optimised
        by subsequent MMF/QM calculations.
        """

        from numpy import array as vector

        # find new neighbours
        first_neighbours = self.first_neighbours

        for heavy_atom_id in marked_heavy_atom_ids:

            # define neighbours of each heavy atom stereocenter
            neighbour_ids = list(first_neighbours[heavy_atom_id])
            h_neighbour_ids = []
            fixed_neighbour_ids = []
            heavy_atom = self.atoms[heavy_atom_id]

            # assumes no radicals
            num_lone_pairs = self.non_bonded_electrons[heavy_atom_id]
            hybridisation = (len(neighbour_ids) + num_lone_pairs)

            for atom_id in neighbour_ids:
                if self.atoms[atom_id].element == 'H':
                    h_neighbour_ids.append(atom_id)
                else:
                    fixed_neighbour_ids.append(atom_id)

            fixed_neighbour_atoms = [
                atom for atom in self.atoms.values()
                if atom.get_index() in fixed_neighbour_ids
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

            elif hybridisation == TRIGONAL_PLANAR:
                if num_h_to_place == 1:

                    if num_lone_pairs == 1:
                        new_coordinates.append(place_first_hydrogen(points, TRIGONAL_PLANAR_BOND_ANGLE))
                    else:
                        new_coordinates.append(gromos_trigonal_planar_1H(points))
                else:
                    new_coordinates.extend(gromos_trigonal_planar_2H(points))

            elif hybridisation == TETRAHEDRAL:

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

            else:
                # TODO: do not handle above tetrahedral state - FIX THIS, USE RANDOM ASSIGNMENT METHOD
                # print("Relying on Bertrand's random method instead... Hybridisation = {} Atom = {} Element = {}".format(hybridisation, heavy_atom_id, heavy_atom.element))
                continue

            # Update atom coordinates
            for (n, atom_id) in enumerate(h_neighbour_ids):
                self._graph._node[atom_id] = Atom3D(
                    index={'name': atom_id},
                    name=atom_id,
                    element='H',
                    formal_charge=0,
                    non_bonded_electrons=0,
                    valence=1,
                    hybridisation=0,
                    is_conjugated=0,
                    is_aromatic=0,
                    coordinates=new_coordinates[n],
                )


    def mol2Str(self) -> str:
        """
        Outputs molecule as mol2 file.
        :return:
        TODO: Finish this!!!
        """

        Mol2Template = '''
        @<TRIPOS>MOLECULE
        *****
         {num_atoms} {num_bonds} 0 0 0
        SMALL
        GASTEIGER'''

        return Mol2Template

    def OBMol(self):
        """
        Returns an openbabel OBMol object from the current 3D NXMol object (Molecule3D).
        :return: OBMol object.
        """

        from openbabel import openbabel as ob

        ob_mol = ob.OBMol()

        # add atoms
        ob_id = 1
        for a_id, a in self.atoms.items():

            ob_a = ob_mol.NewAtom()

            # set index
            a.set_index('OBMol_ID', ob_id)
            ob_a.SetId(ob_id)

            # set parameters
            if a.hybridisation is not None:
                ob_a.SetHyb(a.hybridisation)
            #ob_a.SetAtomicNum(6) # carbon
            #ob_a.SetImplicitHCount()
            ob_a.SetFormalCharge(a.formal_charge)
            ob_a.SetType(a.element)
            x, y, z = a.coordinates
            # ob_a.SetVector(x, y, z) # TODO: debug to see if 2D system works for getting stereo?

            ob_id += 1

        # add bonds
        for b_id, b in self.bonds.items():
            ob_id_1 = self.get_atom(b_id[0]).get_index('OBMol_ID')
            ob_id_2 = self.get_atom(b_id[1]).get_index('OBMol_ID')
            ob_mol.AddBond(ob_id_1, ob_id_2, b.order)

        # debug
        debug = False
        if debug:
            from openbabel import pybel
            for atom in pybel.Molecule(ob_mol):
                NXMol_ID = self.get_atom(atom.idx, index_type='OBMol_ID').get_index()
                print("Original Atom ID: ", NXMol_ID)
                print("Atom ID: ", atom.idx)
                print("Atom Element: ", atom.type)
                print("Atom Charge: ", atom.formalcharge)
                print("Atom Valence: ", atom.degree)


        return ob_mol

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
