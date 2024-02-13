import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix
from io import StringIO
from operator import itemgetter
from typing import List, Union, Optional

from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.graph_features import GraphConvConstants, one_of_k_encoding, find_distance
from deepchem.feat.mol_graphs import WeaveMol

from chemistry_data_structure.helpers.chem import LINEAR, TRIGONAL_PLANAR, \
    TRIGONAL_PLANAR_BOND_ANGLE, TETRAHEDRAL, TETRAHEDRAL_BOND_ANGLE, ELECTRONEGATIVITIES
from chemistry_data_structure.objects.base_objects import _2DChemicalObj, _3DChemicalObj
from chemistry_data_structure.objects.atom_bond import Atom2D, Bond2D, Atom3D, Bond3D, _Bond, _Atom
from chemistry_data_structure.helpers.vector_calculations import place_first_hydrogen, \
    gromos_tetrahedral_1H, gromos_trigonal_planar_1H, gromos_trigonal_planar_2H, gromos_tetrahedral_from_2_vectors, \
    gromos_tetrahedral_3H, place_h_using_ilp
from chemistry_data_structure.parsing.sybyl import sybyl_atom_type


try:
    import gurobipy as gp
except ModuleNotFoundError:
    pass


class Molecule2D(_2DChemicalObj):
    """
    Represents the 2D structure of a molecular entity.
    """

    def __init__(self,
                 atoms: List[Atom2D] = None,
                 bonds: List[Union[str, str, Bond2D]] = None,
                 name: str = '',
                 net_charge: int = None
                 ):

        if atoms is None:
            atoms = []
        if bonds is None:
            bonds = []

        # init super class
        super().__init__(atoms, bonds, name, net_charge)
        self.attributes = {}
        self.angles = {}
        self.dihedrals = {}
        self.qmProperties = {}

        # RMSD fit

    # def index_map(self, index_target: str, index_input: str, index_input_value: str):
    #     # assumption that all atoms have the same index template
    #     temp_key = list(self._graph._node.keys())[0]
    #     if index_target not in self._graph._node[temp_key]._index:


class Molecule3D(_3DChemicalObj, Molecule2D):

    def __init__(self, atoms: List[Atom3D] = None,
                 bonds: List[List[Union[str, Bond3D]]] = None,
                 esp_grid_coords: np.array = None,
                 esp_grid_charge: np.array = None,
                 esp_grid_parameters: dict = None,
                 name: str = '',
                 net_charge: int = None
                 ):
        if atoms is None:
            atoms = []
        if bonds is None:
            bonds = []

        super().__init__(atoms, bonds, name, net_charge)
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

        for (atom_name, pdb_id) in sorted(pdb_ids.items(), key=itemgetter(1)):
            atom = self.atoms[atom_name]
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
                    str(atom_name),
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

        for (atom_name, pdb_id) in sorted(pdb_ids.items(), key=itemgetter(1)):
            print(
                pdb_conect_line(
                    [pdb_id]
                    +
                    [pdb_ids[list(frozenset(bond) - frozenset([atom_name]))[0]] for bond in self.bonds if atom_name in bond]
                ),
                file=io,
            )

        return io.getvalue()

    def mol2Str(self, use_ar_bonds: bool = True) -> str:
        """
        Outputs molecule as mol2 string.
        """
        io = StringIO()

        def print_to_io(*args):
            print(*args, file=io)

        # write header
        print_to_io('@<TRIPOS>MOLECULE')
        print_to_io(self.name)
        print_to_io(
            '{num_atoms} {num_bonds}'.format(
                num_atoms=len(self.atoms),
                num_bonds=len(self.bonds)
            ),
        )
        print_to_io('SMALL')
        print_to_io('USER_CHARGES')

        # write atoms
        print_to_io('@<TRIPOS>ATOM')
        n_id = 1
        name_id_map = {}
        for atom in self.atom_objects:
            print_to_io(
                '{index} {name} {coordinates} {sybyl_atom_type} {subst_id} {subst_name} {charge}'.format(
                    index=n_id,
                    name=atom.get_index('name'),
                    coordinates=' '.join(map(lambda x: '{0:.3f}'.format(float(x)), atom.coordinates)),
                    sybyl_atom_type=sybyl_atom_type(atom.element, atom.valence),
                    subst_id=1,
                    subst_name='<1>',
                    charge=self.formal_charges[atom.get_index()],
                ),
            )
            name_id_map[atom.get_index('name')] = n_id
            n_id += 1


        # write bonds
        print_to_io('@<TRIPOS>BOND')
        for (bond_id, bond) in enumerate(self.bonds, start=1):
            print_to_io(
                '{0} {1} {2} {3}'.format(
                    bond_id,
                    name_id_map[self.get_atom(bond[0]).get_index('name')],
                    name_id_map[self.get_atom(bond[1]).get_index('name')],
                    'ar' if use_ar_bonds and (bond in self.aromatic_bonds) else self.bond_orders[frozenset(bond)],
                ),
            )

        return io.getvalue()

    def molStr(self, use_ar_bonds: bool = True, use_bond_stereo: bool = True) -> str:
        """
        Outputs molecule as mol or sdf string.
        """

        io = StringIO()

        def print_to_io(*args):
            print(*args, file=io)

        # write header
        print_to_io(self.name)
        print_to_io("  atbmol_generated\n")
        print_to_io(
            '{num_atoms: >3}{num_bonds: >3}  0  0  0  0            999 V2000'.format(
                num_atoms=len(self.atoms),
                num_bonds=len(self.bonds)
            ),
        )

        # dictionary for mol formatting of charges conversion
        mol_charge_dict = {-3: 7, -2: 6, -1: 5, 0: 0, 1: 3, 2: 2, 3: 1}

        # write atoms
        n_id = 1
        name_id_map = {}
        for atom in self.atom_objects:
            # TODO: fix the format specifier for coordinates
            print_to_io(
                '{coordinates} {element: <3} 0  {charge}  0  0  0  0  0  0  0  0  0  0'.format(
                    coordinates=''.join(map(lambda x: '{coord:>10}'.format(coord=f"{float(x):.4f}"), atom.coordinates)),
                    element=atom.element,
                    charge=mol_charge_dict[self.formal_charges[atom.get_index()]],
                ),
            )
            name_id_map[atom.get_index('name')] = n_id
            n_id += 1

        # write bonds
        for (bond_id, bond) in enumerate(self.bonds, start=1):
            print_to_io(
                '  {0}  {1}  {2}  {3}  0  0  0'.format(
                    name_id_map[self.get_atom(bond[0]).get_index('name')],
                    name_id_map[self.get_atom(bond[1]).get_index('name')],
                    '4' if use_ar_bonds and (bond in self.aromatic_bonds) else self.bond_orders[frozenset(bond)],
                    0  # TODO: implement cis / trans stereo info in each bond
                ),
            )

        # write charge info
        charge_str = "".join([f"{name_id_map[atom.get_index('name')]:>4}{atom.formal_charge:>4}"
                              for atom in self.atom_objects if atom.formal_charge != 0])
        num_charges = len([charge for charge in self.formal_charges.values() if charge != 0])
        if num_charges >= 8:
            # TODO: implement this
            print("We need to write the code to split over multiple lines the charges, but for now should do.")
            raise NotImplementedError
        print_to_io(f"M  CHG  {num_charges}{charge_str}")

        # terminate
        print_to_io("M  END")
        print_to_io("$$$$")

        return io.getvalue()

    def saveMolToFile(self, fpath: str, format: str):
        """
        Writes the molecule as an output file in the specified format.
        TODO: make this override the _2DChemicalObj method (for saving in graph formats for e.g.)

        :param fpath: the file path to save the mol object to
        :param format: the output format one of ['pdb', 'mol2', 'mol', 'sdf']
        """

        # get output string in right format
        if format == 'pdb':
            out_str = self.pdbStr()
        elif format == 'mol2':
            out_str = self.mol2Str()
        elif format == 'mol':
            out_str = self.molStr()
        elif format == 'sdf':
            out_str = self.molStr()
        else:
            raise AssertionError("format must be one of: 'pdb', 'mol2', 'mol' or 'sdf'")

        print(f'Writing {fpath}...')

        # write to out file
        with open(fpath, 'w') as out_file:
            out_file.write(out_str)

    def calculateNewHydrogenCoordinates(self, marked_heavy_atom_ids: list):
        """
        Places new hydrogens into simple idealised geometries to later be optimised
        by subsequent MMF/QM calculations.
        """

        from numpy import array as vector

        # find new neighbours
        first_neighbours = self.first_neighbours

        # assign coordinates for all attached H atoms
        for heavy_atom_id in self.heavy_atoms:

            # define neighbours of each heavy atom center
            neighbour_ids = first_neighbours[heavy_atom_id]
            h_neighbour_ids = []
            fixed_neighbour_ids = []
            heavy_atom = self.atoms[heavy_atom_id]

            # assumes no radicals
            num_lone_pairs = self.non_bonded_electrons[heavy_atom_id]
            hybridisation = (len(neighbour_ids) + num_lone_pairs)

            # TODO: make sure hybridisations are assigned in the original molecule

            h_neighbour_ids = [atom_id for atom_id in neighbour_ids
                               if self.atoms[atom_id].element == 'H']

            fixed_neighbour_atoms = [self.get_atom(atom_id) for atom_id in neighbour_ids
                                     if self.get_atom(atom_id).element != 'H']

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
                        new_point = place_first_hydrogen(points, TETRAHEDRAL_BOND_ANGLE)
                        new_coordinates.append(tuple(new_point.tolist()))
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

                # Case 4: methyl... use ilp method
                if num_h_to_place == 4:

                    for i in range(num_h_to_place):
                        coords = place_h_using_ilp(points, debug=False)
                        new_coordinates.append(coords)
                        points.append(np.array(coords))

            else:

                # for hybridisations larger than 4, use ilp method to place hydrogens
                for i in range(num_h_to_place):
                    coords = place_h_using_ilp(points, debug=False)
                    new_coordinates.append(coords)
                    points.append(np.array(coords))

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

    def OBMol(self):
        """
        Returns an openbabel OBMol object from the current 3D NXMol object (Molecule3D).

        # TODO: WARNING! not fully implemented...

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


class NXMolWeaveFeaturizer(MolecularFeaturizer):
    """This class implements the featurization to implement Weave convolutions.

  Weave convolutions were introduced in [1]_. Unlike Duvenaud graph
  convolutions, weave convolutions require a quadratic matrix of interaction
  descriptors for each pair of atoms. These extra descriptors may provide for
  additional descriptive power but at the cost of a larger featurized dataset.


  Examples
  --------
  >>> import deepchem as dc
  >>> mols = ["C", "CCC"]
  >>> featurizer = dc.feat.WeaveFeaturizer()
  >>> X = featurizer.featurize(mols)

  References
  ----------
  .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond
         fingerprints." Journal of computer-aided molecular design 30.8 (2016):
         595-608.

  Note
  ----
  This class requires RDKit to be installed.
  """

    name = ['weave_mol']

    def __init__(self,
                 graph_distance: bool = True,
                 explicit_H: bool = False,
                 use_chirality: bool = False,
                 max_pair_distance: Optional[int] = None):
        """Initialize this featurizer with set parameters.

    Parameters
    ----------
    graph_distance: bool, (default True)
      If True, use graph distance for distance features. Otherwise, use
      Euclidean distance. Note that this means that molecules that this
      featurizer is invoked on must have valid conformer information if this
      option is set.
    explicit_H: bool, (default False)
      If true, model hydrogens in the molecule.
    use_chirality: bool, (default False)
      If true, use chiral information in the featurization
    max_pair_distance: Optional[int], (default None)
      This value can be a positive integer or None. This
      parameter determines the maximum graph distance at which pair
      features are computed. For example, if `max_pair_distance==2`,
      then pair features are computed only for atoms at most graph
      distance 2 apart. If `max_pair_distance` is `None`, all pairs are
      considered (effectively infinite `max_pair_distance`)
    """
        # Distance is either graph distance(True) or Euclidean distance(False,
        # only support datasets providing Cartesian coordinates)
        self.graph_distance = graph_distance
        # Set dtype
        self.dtype = object
        # If includes explicit hydrogens
        self.explicit_H = explicit_H
        # If uses use_chirality
        self.use_chirality = use_chirality
        if isinstance(max_pair_distance, int) and max_pair_distance <= 0:
            raise ValueError(
                "max_pair_distance must either be a positive integer or None")
        self.max_pair_distance = max_pair_distance
        if self.use_chirality:
            self.bt_len = int(GraphConvConstants.bond_fdim_base) + len(
                GraphConvConstants.possible_bond_stereo)
        else:
            self.bt_len = 9  # int(GraphConvConstants.bond_fdim_base)

    def _featurize(self, mol: _2DChemicalObj):
        """Encodes mol as a WeaveMol object."""
        # Atom features
        idx_nodes = [(a.get_index('nid'),
                      nxmol_atom_features(
                          a,
                          explicit_H=self.explicit_H,
                          use_chirality=self.use_chirality))
                     for a in mol.atom_objects]
        idx_nodes.sort()  # Sort by ind to ensure same order as rd_kit
        idx, nodes = list(zip(*idx_nodes))

        # Stack nodes into an array
        nodes = np.vstack(nodes)

        # Get bond lists
        bond_features_map = {}
        for a1_name, a2_name in mol.bonds:
            a1_id = mol.get_atom(a1_name).get_index('nid')
            a2_id = mol.get_atom(a2_name).get_index('nid')
            bond_features_map[tuple(sorted([a1_id, a2_id]))] = bond_features(mol.get_bond(a1_name, a2_name),
                                                                       use_chirality=self.use_chirality)

        # print("Bond features map: ", bond_features_map)
        # print("Num nodes: ", len(nodes))
        # print("Numeric atom ids: ", [atom.name for atom in mol.atom_objects])

        # Get canonical adjacency list
        bond_adj_list = [[] for _ in range(len(nodes))]
        for bond in bond_features_map.keys():
            bond_adj_list[bond[0]].append(bond[1])
            bond_adj_list[bond[1]].append(bond[0])

        # Calculate pair features
        pairs, pair_edges = pair_features(
            mol,
            bond_features_map,
            bond_adj_list,
            bt_len=self.bt_len,
            graph_distance=self.graph_distance,
            max_pair_distance=self.max_pair_distance
        )

        return WeaveMol(nodes, pairs, pair_edges)

def pair_features(mol: _2DChemicalObj,
                  bond_features_map: dict,
                  bond_adj_list: List,
                  bt_len: int = 9,
                  graph_distance: bool = True,
                  max_pair_distance: Optional[int] = None) -> np.ndarray:
    """Helper method used to compute atom pair feature vectors.

  Many different featurization methods compute atom pair features
  such as WeaveFeaturizer. Note that atom pair features could be
  for pairs of atoms which aren't necessarily bonded to one
  another.

  Parameters
  ----------
  mol: RDKit Mol
    Molecule to compute features on.
  bond_features_map: dict
    Dictionary that maps pairs of atom ids (say `(2, 3)` for a bond between
    atoms 2 and 3) to the features for the bond between them.
  bond_adj_list: list of lists
    `bond_adj_list[i]` is a list of the atom indices that atom `i` shares a
    bond with . This list is symmetrical so if `j in bond_adj_list[i]` then `i
    in bond_adj_list[j]`.
  bt_len: int, optional (default 6)
    The number of different bond types to consider.
  graph_distance: bool, optional (default True)
    If true, use graph distance between molecules. Else use euclidean
    distance. The specified `mol` must have a conformer. Atomic
    positions will be retrieved by calling `mol.getConformer(0)`.
  max_pair_distance: Optional[int], (default None)
    This value can be a positive integer or None. This
    parameter determines the maximum graph distance at which pair
    features are computed. For example, if `max_pair_distance==2`,
    then pair features are computed only for atoms at most graph
    distance 2 apart. If `max_pair_distance` is `None`, all pairs are
    considered (effectively infinite `max_pair_distance`)

  Note
  ----

  Returns
  -------
  features: np.ndarray
    Of shape `(N_edges, bt_len + max_distance + 1)`. This is the array
    of pairwise features for all atom pairs, where N_edges is the
    number of edges within max_pair_distance of one another in this
    molecules.
  pair_edges: np.ndarray
    Of shape `(2, num_pairs)` where `num_pairs` is the total number of
    pairs within `max_pair_distance` of one another.
  """
    if graph_distance:
        max_distance = 7
    else:
        max_distance = 1
    N = mol.num_atoms
    pair_edges = max_pair_distance_pairs(mol, max_pair_distance)
    num_pairs = pair_edges.shape[1]
    N_edges = pair_edges.shape[1]
    features = np.zeros((N_edges, bt_len + max_distance + 1))
    # Get mapping
    mapping = {}
    for n in range(N_edges):
        a1, a2 = pair_edges[:, n]
        mapping[(int(a1), int(a2))] = n
    num_atoms = mol.num_atoms

    rings = mol.rings  # mol.GetRingInfo().AtomRings()
    for a1 in range(num_atoms):
        for a2 in bond_adj_list[a1]:
            # first `bt_len` features are bond features(if applicable)
            if (int(a1), int(a2)) not in mapping:
                raise ValueError(
                    "Malformed molecule with bonds not in specified graph distance.")
            else:
                n = mapping[(int(a1), int(a2))]
            features[n, :bt_len] = np.asarray(
                bond_features_map[tuple(sorted((a1, a2)))], dtype=float)
        for ring in rings:
            if a1 in ring:
                for a2 in ring:
                    if (int(a1), int(a2)) not in mapping:
                        # For ring pairs outside max pairs distance continue
                        continue
                    else:
                        n = mapping[(int(a1), int(a2))]
                    # `bt_len`-th feature is if the pair of atoms are in the same ring
                    if a2 == a1:
                        features[n, bt_len] = 0
                    else:
                        features[n, bt_len] = 1
        # graph distance between two atoms
        if graph_distance:
            # distance is a matrix of 1-hot encoded distances for all atoms
            distance = find_distance(
                a1, num_atoms, bond_adj_list, max_distance=max_distance)
            for a2 in range(num_atoms):
                if (int(a1), int(a2)) not in mapping:
                    # For ring pairs outside max pairs distance continue
                    continue
                else:
                    n = mapping[(int(a1), int(a2))]
                    features[n, bt_len + 1:] = distance[a2]

    # Euclidean distance between atoms
    # if not graph_distance:
    #   coords = np.zeros((N, 3))
    #   for atom in range(N):
    #     pos = mol.GetConformer(0).GetAtomPosition(atom)
    #     coords[atom, :] = [pos.x, pos.y, pos.z]
    #   features[:, :, -1] = np.sqrt(np.sum(np.square(
    #     np.stack([coords] * N, axis=1) - \
    #     np.stack([coords] * N, axis=0)), axis=2))

    return features, pair_edges


def nxmol_atom_features(atom: _Atom,
                        bool_id_feat=False,
                        explicit_H=True,
                        use_chirality=False):
    """Helper method used to compute per-atom feature vectors.

  Many different featurization methods compute per-atom features such as ConvMolFeaturizer, WeaveFeaturizer. This method computes such features.

  Parameters
  ----------
  bool_id_feat: bool, optional
    Return an array of unique identifiers corresponding to atom type.
  explicit_H: bool, optional
    If true, model hydrogens explicitly
  use_chirality: bool, optional
    If true, use chirality information.

  Returns
  -------
  np.ndarray of per-atom features.
  """
    # if bool_id_feat:
    #     return np.array([atom_to_id(atom)])
    # else:
    # from rdkit import Chem
    # TODO: this is where we can modify atom featurization
    results = \
        one_of_k_encoding(atom.element, ELECTRONEGATIVITIES.keys()) + \
        [atom.formal_charge] + \
        [atom.non_bonded_electrons] + \
        [atom.is_conjugated] + \
        one_of_k_encoding(atom.valence, [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        #one_of_k_encoding(atom.hybridisation, [-SP3D2_HYB, -SP3D_HYB, -SP3_HYB, -SP2_HYB, -SP_HYB, 0, SP_HYB, SP2_HYB, SP3_HYB, SP3D_HYB, SP3D2_HYB])  # MAY LEAD TO MULTICOLINEARITY, DUE TO DEPENDENCE ON OTHER FACTORS
        #one_of_k_encoding(atom.hybridisation, [SP_HYB, SP2_HYB, SP3_HYB, SP3D_HYB, SP3D2_HYB])
    # TODO: CHECK THESE VALUES!!!

    # print("Atom one hot encoding: ", results)

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    # if not explicit_H:
    #     results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
    #                                               [0, 1, 2, 3, 4])
    # if use_chirality:
    #     try:
    #         results = results + one_of_k_encoding_unk(
    #             atom.GetProp('_CIPCode'),
    #             ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
    #     except:
    #         results = results + [False, False
    #                              ] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)


def bond_features(bond: _Bond, use_chirality=False):
    """Helper method used to compute bond feature vectors.

  Many different featurization methods compute bond features
  such as WeaveFeaturizer. This method computes such features.

  Parameters
  ----------
  use_chirality: bool, optional
    If true, use chirality information.

  Note
  ----
  This method requires RDKit to be installed.

  Returns
  -------
  bond_feats: np.ndarray
    Array of bond features. This is a 1-D array of length 6 if `use_chirality`
    is `False` else of length 10 with chirality encoded.
  """
    # try:
    #     from rdkit import Chem
    # except ModuleNotFoundError:
    #     raise ImportError("This method requires RDKit to be installed.")
    bt = bond.order

    bond_feats = [one_of_k_encoding(bt, [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]),
                  # bt == SINGLE, bt == DOUBLE, bt == TRIPLE, bt == AROMATIC,  # (this implies we store 1.5 as aromatic bond order)
                  # bond.is_conjugated,  # TODO: include this stuff, and get aromatic bond order working above^^
                  # bond.is_in_ring
                  ]

    # if use_chirality:
    #     bond_feats = bond_feats + one_of_k_encoding_unk(
    #         str(bond.GetStereo()), GraphConvConstants.possible_bond_stereo)
    return np.array(bond_feats)


def max_pair_distance_pairs(mol: _2DChemicalObj,
                            max_pair_distance: Optional[int] = None) -> np.ndarray:
    """Helper method which finds atom pairs within max_pair_distance graph distance.

      This helper method is used to find atoms which are within max_pair_distance
      graph_distance of one another. This is done by using the fact that the
      powers of an adjacency matrix encode path connectivity information. In
      particular, if `adj` is the adjacency matrix, then `adj**k` has a nonzero
      value at `(i, j)` if and only if there exists a path of graph distance `k`
      between `i` and `j`. To find all atoms within `max_pair_distance` of each
      other, we can compute the adjacency matrix powers `[adj, adj**2,
      ...,adj**max_pair_distance]` and find pairs which are nonzero in any of
      these matrices. Since adjacency matrices and their powers are positive
      numbers, this is simply the nonzero elements of `adj + adj**2 + ... +
      adj**max_pair_distance`.

  Parameters
  ----------
  mol: rdkit.Chem.rdchem.Mol
    RDKit molecules
  max_pair_distance: Optional[int], (default None)
    This value can be a positive integer or None. This
    parameter determines the maximum graph distance at which pair
    features are computed. For example, if `max_pair_distance==2`,
    then pair features are computed only for atoms at most graph
    distance 2 apart. If `max_pair_distance` is `None`, all pairs are
    considered (effectively infinite `max_pair_distance`)

  Examples
  --------
  >>> from rdkit import Chem
  >>> mol = Chem.MolFromSmiles('CCC')
  >>> features = dc.feat.graph_features.max_pair_distance_pairs(mol, 1)
  >>> type(features)
  <class 'numpy.ndarray'>
  >>> features.shape  # (2, num_pairs)
  (2, 7)

  Returns
  -------
  np.ndarray
    Of shape `(2, num_pairs)` where `num_pairs` is the total number of pairs
    within `max_pair_distance` of one another.
  """
    #from rdkit import Chem
    #from rdkit.Chem import rdmolops
    N = mol.num_atoms
    if (max_pair_distance is None or max_pair_distance >= N):
        max_distance = N
    elif max_pair_distance is not None and max_pair_distance <= 0:
        raise ValueError("max_pair_distance must either be a positive integer or None")
    elif max_pair_distance is not None:
        max_distance = max_pair_distance

    adj = mol.get_graph_adj_mat() #rdmolops.GetAdjacencyMatrix(mol)
    # print(type(adj))

    # Handle edge case of self-pairs (i, i)
    sum_adj = np.eye(N)
    for i in range(max_distance):
        # Increment by 1 since we don't want 0-indexing
        power = i + 1
        sum_adj += np.linalg.matrix_power(adj, power)
    nonzero_locs = np.where(sum_adj != 0)
    num_pairs = len(nonzero_locs[0])
    # This creates a matrix of shape (2, num_pairs)
    pair_edges = np.reshape(np.array(list(zip(nonzero_locs))), (2, num_pairs))
    return pair_edges


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
