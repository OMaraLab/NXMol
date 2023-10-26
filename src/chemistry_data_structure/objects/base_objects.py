import queue
from functools import reduce
from sys import stderr

import networkx as nx
from typing import List, Union, Optional, TextIO, Tuple, Any, Iterable, FrozenSet, Set
import numpy as np
from matplotlib import pyplot as plt
from chemistry_data_structure.helpers.chem import ELECTRONEGATIVITIES, VALENCE_ELECTRONS, FULL_VALENCES, \
    AROMATIC_BOND_ORDER
from chemistry_data_structure.helpers.io import write_to_debug
from chemistry_data_structure.helpers.rings import bonds_for_ring
from chemistry_data_structure.objects.atom_bond import Atom2D, Atom3D, _Bond, _Atom
from chemistry_data_structure.objects.atom_bond import Bond2D, Bond3D

ELEMENT_COLOURS = {'H': '#eeeeee',
                   'C': 'grey',
                   'O': '#ff91a4',
                   'N': '#5dabf5',
                   'S': '#b7b70f',
                   'P': 'orange',
                   'F': 'green',
                   'CL': 'purple',
                   'BR': 'pink',
                   'SI': 'yellow'}

MARKED_COLOURS = {'Marked': '#ff91a4', 'Unmarked': '#eeeeee'}


class _2DChemicalObj:
    """
    Class representing a molecular entity, i.e. with a unique structural connectivity
    represented as a graph and stereo-isomeric form. This object contains a networkx graph to represent
    it. This can be parsed to numerous common string formats.
    """

    def __init__(self,
                 atoms: List[_Atom] = None,
                 bonds: List[Union[str, str, _Bond]] = None,
                 name: str = '',
                 net_charge: Optional[int] = None
                 ):

        # iterate for all init methods
        self._net_charge = net_charge

        # init graph
        self._name = name
        self._graph = nx.Graph()
        if atoms is not None:
            self._graph.add_nodes_from([a.name for a in atoms])
            for a in atoms:
                self._graph._node[a.name] = a

            if bonds is not None:
                self._graph.add_edges_from([(a1, a2) for a1, a2, _ in bonds])
                for a1, a2, bond in bonds:
                    self._graph._adj[a1][a2] = bond
                    self._graph._adj[a2][a1] = bond

        # sub_class entities
        self._fragments = {}
        self._conformations = {}

        # molecule properties
        # TODO: store dictionaries in object for quick access

        # dict of dict of properties for iterative updates
        # TODO: @Callum I'm sure theres a more class correct way of doing this, do you know?
        #   i.e. a data structure for storing all class properties that can be iterated through easily
        #   i'm thinking we can use this when adding/removing atoms and bonds to update other properties
        #   if we store them explicitly
        #   @Joseph Maybe using either the default dataclass object or one of the packages might work here?
        self._properties = {'fragments': self._fragments}

    def __repr__(self):
        # The Multifit ESP pipeline often creates copies of objects, and it's useful to know if you're dealing
        # with copies or references
        return f'{self._name if self._name else type(self).__name__} at {hex(id(self))}'

    #
    # @property
    # def chemical_formula(self):
    #     elements = [a.element for a in self.atom_objects]
    #     return ''.join(f'{e}{elements.count(e)}' for e in list(set(elements)))

    # TODO: so with these properties, they are very convenient but actually
    #   require some loops etc. each time they are called. What would be cool
    #   is if we can somehow allow all of these properties to be updated
    #   whenever there is a change in the molecule, and otherwise just
    #   return the value for that property.

    @property
    def graph(self):
        return self._graph

    @property
    def name(self):
        return self._name

    @property
    def atoms(self):
        # TODO: we should modify this to be atom_ids, just need to refactor
        return self.graph.nodes

    @property
    def heavy_atoms(self):
        return list(atom.get_index() for atom in self.atom_objects if atom.element != 'H')

    @property
    def atom_objects(self):
        return list(self.graph.nodes.values())

    @property
    def num_atoms(self):
        return len(self.graph.nodes)

    @property
    def bonds(self):
        return self.graph.edges

    @property
    def bond_objects(self):
        return list(self.graph.edges.values())

    @property
    def bond_orders(self):
        return {frozenset(bond_ids): self.get_bond(bond_ids[0], bond_ids[1]).order for bond_ids in self.bonds}

    @property
    def rings(self):
        return list(map(tuple, nx.cycle_basis(self.graph)))

    @property
    def aromatic_atoms(self):
        return list(atom.get_index() for atom in self.atom_objects if atom.is_aromatic)

    @property
    def aromatic_bonds(self):
        return list(frozenset(bond_ids) for bond_ids in self.bonds
                    if self.get_bond(bond_ids[0], bond_ids[1]).is_aromatic == True)

    @property
    def formal_charges(self):
        return {atom_id: self.get_atom(atom_id).formal_charge for atom_id in self.atoms}

    @property
    def non_bonded_electrons(self):
        return {atom_id: self.get_atom(atom_id).non_bonded_electrons for atom_id in self.atoms}

    @property
    def hybridisations(self):
        return {atom_id: self.get_atom(atom_id).hybridisation for atom_id in self.atoms}

    @property
    def atom_conjugations(self):
        return {atom_id: self.get_atom(atom_id).is_conjugated for atom_id in self.atoms}

    @property
    def valences(self):
        return {atom_id: self.get_atom(atom_id).valence for atom_id in self.atoms}

    @property
    def coordinates(self):
        return {atom_id: self.get_atom(atom_id).coordinates for atom_id in self.atoms}

    @property
    def chirality(self):
        return {atom_id: self.get_atom(atom_id).chirality for atom_id in self.atoms
                if self.get_atom(atom_id).element == 'C'}

    @property
    def neighbour_counts(self):
        return {atom_id: len([x for x in self.graph.neighbors(atom_id)]) for atom_id in self.atoms}

    @property
    def first_neighbours(self):
        return {atom_id: [x for x in self.graph.neighbors(atom_id)] for atom_id in self.atoms}

    @property
    def net_charge(self):
        return self._net_charge

    @net_charge.setter
    def net_charge(self, value):
        assert isinstance(value, int)
        self._net_charge = value

    # @graph.setter
    # def graph(self, value):
    #     self._graph = value

    # def __repr__(self):
        # I've got a fun idea coming for this one
        # https://github.com/vfscalfani/teletype_mols/blob/main/rdkit_print_mol_ascii.ipynb

    def set_name(self, name: str):
        self._name = name

    def add_atom(self, atom: Atom2D) -> None:
        if not isinstance(atom, Atom2D):
            # not sure if we actually want to add atoms this way
            # might make it  easier to enforce minimum information
            raise TypeError(f'atom must be of type Atom2D, but is of type f{type(atom)}')
        if atom.name in self._graph.nodes:
            raise IndexError # Error type subject to change
        self._graph.add_node(atom.name)
        self._graph._node[atom.name] = atom

    def replace_atom(self, atom: Atom2D, atom_id: str):
        """
        :param atom:
        :param atom_id:
        :return:
        """
        self._graph._node[atom_id] = atom

    def rename_atom(self, atom: Atom2D, name):
        nx.relabel_nodes(self._graph,
                         {atom.name: name},
                         copy=False)
        atom.index['name'] = name

    def add_bond(self, a1: str, a2: str, bond: _Bond) -> None:
        """
        :param bond:
        :return:
        """
        if not isinstance(bond, _Bond):
            raise TypeError('bond must be of type Bond')

        if a1 not in self._graph.nodes or a2 not in self._graph.nodes:
            raise IndexError

        self._graph.add_edge(a1, a2)
        self._graph._adj[a1][a2] = bond
        self._graph._adj[a2][a1] = bond

    def remove_bond(self, a1: str, a2: str) -> None:
        self._graph.remove_edge(a1, a2)

    def remove_atom(self, index: Any, index_type='name') -> None:
        """
        Removes an atom from the molecular graph.
        :param index: the atom index to remove
        :param index_type: the type of atom index used for lookup
        """
        if index_type == 'name':
            # use name index
            self._graph.remove_node(index)
        else:
            # account for alternate indexing system
            for a_id, a in self.atoms.items():
                if a.index[index_type] == index:
                    self._graph.remove_node(a_id)

    def remove_atoms(self, indices: set, index_type='name') -> None:
        """
        Removes multiple atoms from the molecular graph.
        :param indices: the set of atom indices to remove
        :param index_type: the type of atom index used for lookup
        """
        for index in indices:
            self.remove_atom(index, index_type=index_type)

    def get_atom(self, index: Any, index_type='name') -> Atom2D:
        if index_type == 'name':
            return self._graph.nodes[index]
        else:
            match_list = [a for a in self._graph._node.values() if a.index[index_type] == index]
            return match_list[0] if len(match_list) > 0 else None
            # return next((d for d in self._graph._node.values() if d[index_type] == index), None)

    def get_atoms(self, index: List, index_type='name'):
        """
        Return list of atom objects within the molecule object associated with the id
        :param index:
        :param index_type:
        :return:
        """
        if index_type == 'name':
            return [self._graph.nodes[i] for i in index]
        else:
            return [[a for a in self._graph._node.values() if a.index[index_type] == i][0] for i in index]
            # return [next((d for d in self._graph._node.values() if d[index_type] == i), None)
            #         for i in index]

    def get_heavy_atoms(self):
        """
        Returns all heavy atoms in the molecule.
        """
        return [a for a in self.atoms if self.get_atom(a).element != 'H']

    def get_heavy_atom_bonds(self):
        """
        Returns bonds between only heavy atoms in the molecule.
        """
        return [bond for bond in self.bonds
                if self.get_atom(bond[0]).element != 'H' and self.get_atom(bond[1]).element != 'H']

    def get_formal_charges(self, index: Iterable[str]):
        """
        Get the formal charges for a specified list of atoms. Uses atom name indexing.
        :return:
        """
        return {atom_id: self.formal_charges[atom_id] for atom_id in index}

    def get_non_bonded_electrons(self, index: Iterable[str]):
        """
        Get the non-bonded electrons for a specified list of atoms. Uses atom name indexing.
        :return:
        """
        return {atom_id: self.non_bonded_electrons[atom_id] for atom_id in index}

    def get_hybridisations(self, index: Iterable[str]):
        """
        Get the hybridisations for a specified list of atoms. Uses atom name indexing.
        :return:
        """
        return {atom_id: self.hybridisations[atom_id] for atom_id in index}

    def get_valences(self, index: Iterable[str]):
        """
        Get the valences for a specified list of atoms. Uses atom name indexing.
        :return:
        """
        return {atom_id: self.valences[atom_id] for atom_id in index}

    def get_conjugations(self, index: Iterable[str]):
        """
        Get the conjugation status for a specified list of atoms. Uses atom name indexing.
        :return:
        """
        return {atom_id: self.atom_conjugations[atom_id] for atom_id in index}

    def get_bond_orders(self, index: Iterable[Tuple[str]]):
        """
        Get the bond orders for a specified list of bonds. Uses atom name indexing.
        :return:
        """
        return {frozenset((a1_id, a2_id)): self.bond_orders[frozenset((a1_id, a2_id))] for a1_id, a2_id in index}

    def get_bond(self, a1_id: Any, a2_id: Any, index_type='name'):
        if index_type == 'name':
            return self.graph[a1_id][a2_id]
        # TODO: add other index types?

    def get_atoms_in_ring_with_size(self, size: int) -> set:
        """
        Get atoms in molecule rings with a specific maximum ring size.
        :return:
        """
        return set(a for ring in self.rings for a in ring if len(ring) < size)

    def get_bonds_in_ring_with_size(self, size: int) -> set:
        """
        Get bonds in  molecule rings with a specific maximum ring size.
        :return:
        """
        ring_bonds = set()
        for ring in self.rings:
            if len(ring) < size:
                for bond in self.bonds:
                    if set(bond).issubset(set(ring)):
                        ring_bonds.add(bond)
        return ring_bonds

    def get_backbone_graph(self):
        """
        Returns a copy of a subgraph view of the molecular graph
        containing only heavy atoms. Uses atom names as indices.
        The returned subgraph view's attributes are linked to the base graph.
        :return: nx.graph
        """
        heavy_atoms = set(a.name for a in self.atoms.values() if a.element != 'H')
        return self.graph.subgraph(heavy_atoms)

    def get_neighbour_element_counts(self, element: str):
        """
        Returns a dictionary of {atom:count} where count is the number of neighbouring
        atoms of the given element.
        :param element: the element to count neighbours for
        :return: dict of {atom_id : count}
        """
        counts = {}
        for atom_id in self.atoms:
            counts[atom_id] = len([n_id for n_id in self.graph.neighbors(atom_id)
                                   if self.get_atom(n_id).element == element])
        return counts

    def get_element_count(self, elements: Set[str]) -> int:
        """
        Returns the number of atoms of a certain set of elements.
        :param elements: set of elements to count
        :return: int count
        """
        return len([a for a in self.atoms if self.get_atom(a).element in elements])

    def draw_graph(self,
                   fixed_heavy_atoms: dict = None,
                   backbone_only: bool = False,
                   show: bool = True,
                   save_fp: str = None,
                   font_sizes: dict = None,
                   offsets: tuple = None,
                   node_size: int = 300,
                   node_label_mode: str = 'element',
                   draw_formal_charges: bool = False,
                   draw_chiral: bool = False,
                   draw_marked_atoms: bool = False):
        """
        Draws the molecular graph in kamada kawai layout.
        :param fixed_heavy_atoms: a dictionary of fixed node positions for heavy atoms.
        :param backbone_only: if true, only draw the backbone of the molecule (only heavy atoms, no hydrogens)
        :param show: if true, show plot, otherwise don't
        :param save_fp: if specified, save the png to the given fp.
        :param font_sizes: a dictionary of font sizes: {'node': int, 'edge': int, 'label': int}
        :param offsets: position offsets for the labels
        :param node_label_mode: one of 'id', 'element' or other
        :param draw_formal_charges: if true, draw atomic formal charge labels
        :param draw_chiral: if true, draw R and S chirality info on chiral centers
        :param draw_marked_atoms: if true, uses a colour scheme to show marked atoms as red and others as grey
        """
        node_font_sz, edge_font_sz, charge_font_sz = font_sizes['node'] if font_sizes else node_size // 50, \
                                                     font_sizes['edge'] if font_sizes else node_size // 45, \
                                                     font_sizes['label'] if font_sizes else None

        # get marked atom ids
        marked_atom_ids = set()
        if draw_marked_atoms:
            for atom in self._graph.nodes.values():
                for frag_name, fragment in self._fragments.items():
                    if 'marked' in frag_name and atom.get_index() in fragment:
                        marked_atom_ids.add(atom.get_index())

        # graph node drawing options
        if backbone_only:

            # only draw backbone (non-hydrogen atoms)
            graph = self.get_backbone_graph()
            formal_charges = {a.get_index(): a.formal_charge for a in graph.nodes.values()}
            chirality = {a.get_index(): a.chirality for a in graph.nodes.values()}
            bond_orders = {frozenset(bond_ids): self.get_bond(bond_ids[0], bond_ids[1]).order for bond_ids in graph.edges()}

        elif draw_marked_atoms:

            # remove hydrogens that aren't attached to a marked atom
            first_neighbours = self.first_neighbours
            draw_atoms = set(self.atoms.keys())
            for a_id, a in self.atoms.items():
                if a.element == 'H' and set(first_neighbours[a_id]).issubset(marked_atom_ids):
                    draw_atoms.remove(a_id)

            print("Draw Atoms: ", draw_atoms)
            print("Marked Atom IDs: ", marked_atom_ids)
            graph = self._graph.subgraph(draw_atoms)

            formal_charges = {a_id: '?' for a in graph.nodes()}
            chirality = {a.get_index(): a.chirality for a in graph.nodes.values()}
            bond_orders = {frozenset(bond_ids): '?' for bond_ids in graph.edges()}
        else:

            # otherwise draw normal graph
            graph = self._graph
            print(self.atoms)
            formal_charges = self.formal_charges
            chirality = self.chirality
            bond_orders = self.bond_orders

        # create colour map of element colours
        colour_map = []
        for atom in graph.nodes.values():

            # apply marking colour scheme
            if draw_marked_atoms:
                colour_map.append(MARKED_COLOURS['Marked'] if atom.get_index() in marked_atom_ids else MARKED_COLOURS['Unmarked'])
            else:
                colour_map.append(ELEMENT_COLOURS[atom.element.strip("0123456789").upper()])

        # setup layouts
        kamada_kawai = False
        if fixed_heavy_atoms is not None:
            pos = nx.spring_layout(graph, pos=fixed_heavy_atoms, fixed=fixed_heavy_atoms.keys())
        else:
            kamada_kawai = True
            pos = nx.kamada_kawai_layout(graph, scale=2)

        if offsets is None:
            if backbone_only:
                offsets = (0.5, 0.15)
            else:
                offsets = (0.65, 0.2)
        offset_pos = {k: (v[0] + offsets[0], v[1] + offsets[1]) for k, v in pos.items()}

        # override if kamada kawai
        if kamada_kawai:
            #offset_pos = {k: (v[0] + 0.06, v[1] + 0.05) for k, v in pos.items()}
            offset_pos = pos


        # set node labels to ids, labels or by default only the elements
        if node_label_mode == 'id':
            node_labels = {a_id: a_id for a_id in graph.nodes.keys()}
        elif node_label_mode == 'element':
            node_labels = {a_id: a.element for a_id, a in graph.nodes.items()}
        else:
            node_labels = {}

        # draw graph, with node and edge labels
        fig = plt.figure(dpi=600, figsize=(6, 5))
        nx.draw(graph,
                pos=pos,
                labels=node_labels,
                font_size=node_font_sz,
                font_color='black',
                font_family='serif',#'monospace',
                node_color=colour_map,
                node_size=node_size,
                edge_color='black')

        # node and edge formatting
        node_path_coll = plt.gca().collections[0]
        node_path_coll.set_edgecolor("#afafaf")
        node_path_coll.set_lw(1.5)
        edge_path_coll = plt.gca().collections[1]
        edge_path_coll.set_edgecolor("#575C5D")
        edge_path_coll.set_lw(1.5)

        # draw formal charges
        if draw_formal_charges:
            nx.draw_networkx_labels(graph, offset_pos, formal_charges,
                                    font_color='red', font_size=charge_font_sz)

        # draw chiral centers
        if draw_chiral:
            nx.draw_networkx_labels(graph,
                                    offset_pos,
                                    chirality,
                                    font_color='blue',
                                    font_size=charge_font_sz)

        # draw bond orders
        nx.draw_networkx_edge_labels(graph,
                                     pos,
                                     bond_orders,
                                     font_size=edge_font_sz,
                                     verticalalignment="center",
                                     )#bbox=dict(alpha=0))

        # if backbone_only:
        #     plt.margins(x=0.2, y=0.2)
        # else:
        #     plt.margins(x=0.1, y=0.1)

        # optionally show or save molecular graph
        if show:
            plt.show()
        else:
            if save_fp is not None:
                plt.savefig(save_fp, format='png')
            plt.cla()
            plt.close()

    def get_graph_adj_mat(self, as_numpy: bool = True):
        """
        Return adjacency matrix representation of molecular graph.
        :param: as_numpy: if true, return as numpy array, otherwise return as list of lists
        """
        adj = nx.to_numpy_array(self._graph).astype('int32')
        if as_numpy:
            return adj
        else:
            return adj.tolist()

    def set_atom_attributes(self, attrs: dict):
        """
        Update the atom node objects with the attributes as specified by the attrs dictionary.
        :param attrs: dictionary of atom keys to attributes to set, i.e. {'H1': {'formal_charge': 0, 'nbes': 10}}
        """
        nx.set_node_attributes(self._graph, attrs)

        # TODO: temporary until atom update() is implemented
        # for a_id, attr in attrs.items():
        #     self._graph.
        #     pass

    def set_bond_attributes(self, attrs: dict):
        """
        Update the edge objects with the attributes as specified by the attrs dictionary.
        :param attrs: dictionary of bond keys to attributes to set, i.e. {('H1', 'C1'): {'order': 1}}
        """
        nx.set_edge_attributes(self._graph, attrs)

        # TODO: temporary until atom update() is implemented
        # for a_id, attr in attrs.items():
        #     self._graph.
        #     pass

    def assign_bond_orders_and_charges_with_ILP(
            self,
            net_charge: int,
            total_electrons: Optional[int] = None,
            enforce_octet_rule: bool = True,
            allow_radicals: bool = False,
            debug: Optional[TextIO] = None,
            bond_order_constraints: List[Tuple[frozenset, int]] = [],
            disallow_triple_bond_in_small_rings: bool = True,
            disallow_allenes_in_small_rings: bool = True,
            disallow_allenes_completely: bool = True,
    ) -> None:
        """

        Assigns bond orders and charges using a consistent resonance form for a molecule.

        TODO: possibly rewrite this, it is extremely bloated and possibly overcomplicated.

        :param net_charge:
        :param total_electrons:
        :param enforce_octet_rule:
        :param allow_radicals:
        :param debug:
        :param bond_order_constraints:
        :param disallow_triple_bond_in_small_rings:
        :param disallow_allenes_in_small_rings:
        :param disallow_allenes_completely:
        :return:
        """
        from pulp import LpProblem, LpMinimize, LpInteger, LpVariable, LpBinary, LpStatus, value
        from pulp import PULP_CBC_CMD, PulpSolverError

        MIN_ABSOLUTE_CHARGE, MAX_ABSOLUTE_CHARGE = 0, 9
        MIN_BOND_ORDER, MAX_BOND_ORDER = 1, 3
        MAX_NONBONDED_ELECTRONS = 18
        ELECTRONS_PER_BOND = 2
        ILP_SOLVER_TIMEOUT = 30
        ELECTRON_MULTIPLIER = (2 if not allow_radicals else 1)

        print_if_debug = lambda *args: write_to_debug(debug, *args)

        assert not (allow_radicals and enforce_octet_rule), "Can't simultaneously allow_radicals and enforce octet rule."

        problem = LpProblem("Lewis problem (bond order and charge assignment)", LpMinimize)

        non_allene_atoms = {}
        for atom in self.atom_objects:
            if disallow_allenes_completely:
                atom_bonds = [bond for bond in self.bonds if atom.get_index() in bond]
                if atom.element == 'C' and len(atom_bonds) == 2:
                    non_allene_atoms[atom] = atom_bonds

        # ===== VARIABLES =====

        # formal charges of atoms
        charges = {
            atom.get_index(): LpVariable("C_{i}".format(i=atom.get_index()), -MAX_ABSOLUTE_CHARGE, MAX_ABSOLUTE_CHARGE, LpInteger)
            for atom in self.atom_objects
        }

        # variable to bind absolute values of charges
        absolute_charges = {
            atom_id: LpVariable("Z_{i}".format(i=atom_id), MIN_ABSOLUTE_CHARGE, MAX_ABSOLUTE_CHARGE, LpInteger)
            for atom_id in charges.keys()
        }

        # non-bonded electrons of each atom
        non_bonded_electrons = {
            atom_id: LpVariable("N_{i}".format(i=atom_id), 0, MAX_NONBONDED_ELECTRONS // ELECTRON_MULTIPLIER, LpInteger)
            for (atom_id, atom) in self.atoms.items()
        }

        # Maps a bond to an integer
        bond_mapping = {
            bond: i
            for (i, bond) in enumerate(self.bonds)
        }

        # Maps an integer to a bond
        bond_reverse_mapping = {v: k for (k, v) in bond_mapping.items()}

        # if disallow_triple_bond_in_small_rings:
        #     print_if_debug('Note: Excluding triple bonds in small rings (<= {0})'.format(SMALL_RING))
        #     bonds_in_small_rings = bonds_in_small_rings_for(self)
        # else:
        #     bonds_in_small_rings = set()

        bond_orders = {
            bond: LpVariable(
                "B_{i}".format(i=bond_mapping[bond]),
                MIN_BOND_ORDER,
                MAX_BOND_ORDER, #if bond not in bonds_in_small_rings else 2,
                LpInteger,
            )
            for bond in self.bonds
        }

        # ===== OBJECTIVES =====
        OBJECTIVES = [
                         sum(absolute_charges.values()),
                         # FIXME: sum(charges.values()) as close to zero as possible (cf example_wang_8 and example_wang_9)
                         sum([charge * ELECTRONEGATIVITIES[self.get_atom(atom_id).element.upper()] for (atom_id, charge) in
                                  charges.items()]),
                     ] + (
                         [sum(
                             [bond_order * ELECTRONEGATIVITIES[self.get_atom(atom_id).element.upper()] for (bond, bond_order) in
                              bond_orders.items() for atom_id in bond])]
                         if len(bond_orders) > 0
                         else []
                     )

        # ===== CONSTRAINTS =====
        problem += sum(charges.values()) == net_charge, 'Total net charge'

        if total_electrons is not None:
            problem += (ELECTRONS_PER_BOND * sum(bond_orders.values()) + ELECTRON_MULTIPLIER * sum(
                non_bonded_electrons.values()) == total_electrons, 'Known total electrons')

        for atom in self.atoms.values():
            problem += charges[atom.get_index()] == VALENCE_ELECTRONS[atom.element.upper()] - sum(
                [bond_orders[bond] for bond in self.bonds if atom.get_index() in bond]) - ELECTRON_MULTIPLIER * \
                       non_bonded_electrons[atom.get_index()], '{element}_{index}'.format(element=atom.element.upper(),
                                                                                    index=atom.get_index())

        # Deal with absolute values
        for atom in self.atoms.values():
            problem += charges[atom.get_index()] <= absolute_charges[atom.get_index()], 'Absolute charge contraint 1 {i}'.format(
                i=atom.get_index())
            problem += -charges[atom.get_index()] <= absolute_charges[atom.get_index()], 'Absolute charge contraint 2 {i}'.format(
                i=atom.get_index())

            if enforce_octet_rule:
                if atom.element.upper() not in {'B', 'BE', 'P', 'S'}:
                    problem += (
                        ELECTRONS_PER_BOND * sum(
                            [bond_orders[bond] for bond in self.bonds if atom.get_index() in bond]) + ELECTRON_MULTIPLIER *
                        non_bonded_electrons[atom.get_index()] == (2 if atom.element.upper() in {'H', 'HE'} else 8),
                        'Octet for atom {element}_{index}'.format(element=atom.element.upper(), index=atom.get_index()),
                    )

        for (bond, bond_order) in bond_order_constraints:
            problem += bond_orders[frozenset(bond)] == bond_order, 'Constraint bond {0} == {1}'.format(bond, bond_order)

        for (atom, (bond_1, bond_2)) in non_allene_atoms.items():
            new_allene_switch = LpVariable('A_{i}'.format(i=atom.get_index()), 0, 1, LpBinary)
            problem += 2 * bond_orders[bond_1] - bond_orders[bond_2] + 4 * new_allene_switch >= 3
            problem += 2 * bond_orders[bond_1] - bond_orders[bond_2] + 4 * new_allene_switch <= 5

        # for atom in atoms_in_small_rings_for(self):
        #     if disallow_allenes_in_small_rings:
        #         if atom.element in {'C', 'N'}:
        #             adjacent_non_hydrogen_bonds = [bond for bond in self.bonds if atom.get_index() in bond]
        #             if len(adjacent_non_hydrogen_bonds) == 2:
        #                 problem += sum(bond_orders[bond] for bond in
        #                                adjacent_non_hydrogen_bonds) <= 3, 'No allenes for atom {atom_desc} in short ring'.format(
        #                     atom_desc=atom_short_desc(atom))

        # ===== SOLVING =====
        try:
            #problem.setObjective(sum(OBJECTIVES))
            #problem.solve(PULP_CBC_CMD(maxSeconds=100))
            problem.sequentialSolve(OBJECTIVES) #timeout=ILP_SOLVER_TIMEOUT)
            assert problem.status == 1, (self.name, LpStatus[problem.status])
        except (AssertionError, PulpSolverError) as e:
            args_id = ','.join(map(str, [enforce_octet_rule, allow_radicals, bond_order_constraints]))
            debug_file = '{0}_{1}_debug.lp'.format(self.name, args_id)
            problem.writeLP(debug_file)
            #self.write_graph('DEBUG', output_size=(1000, 1000))
            stderr.write('\n' + 'Failed LP written to "{0}"'.format(debug_file))
            raise PulpSolverError

        write_to_debug(debug, 'Objective function values: {0}'.format([value(objective) for objective in OBJECTIVES]))

        # set solution variables in molecule atoms/bonds
        for v in problem.variables():
            variable_type, variable_substr = v.name.split('_')
            if variable_type == 'C':
                self.get_atom(variable_substr).formal_charge = round(v.varValue)
            elif variable_type == 'B':
                bond_index = int(variable_substr)
                a1, a2 = bond_reverse_mapping[bond_index]
                self.get_bond(a1, a2).order = round(v.varValue)
            elif variable_type == 'Z':
                pass
            elif variable_type == 'N':
                self.get_atom(variable_substr).non_bonded_electrons = round(v.varValue) * ELECTRON_MULTIPLIER
                if allow_radicals and self.get_atom(variable_substr).non_bonded_electrons % 2 == 1:
                    stderr.write('Warning: Radical molecule...')
            elif variable_type == 'A':
                pass
            else:
                raise Exception('Unknown variable type: {0}'.format(variable_type))
        write_to_debug(debug, 'molecule_name', self.name)
        write_to_debug(debug, 'bond_orders:', self.bond_orders)
        write_to_debug(debug, 'formal_charges', self.formal_charges)
        write_to_debug(debug, 'non_bonded_electrons', self.non_bonded_electrons)

    def assign_aromatic_bonds(self):
        """
        Assigns aromatic bonds using huckel rules.
        :return:
        """

        rings = self.rings

        ring_bonds = {
            ring: bonds_for_ring(ring)
            for ring in rings
        }

        try:
            ring_bond_orders = {
                ring: [self.bond_orders[bond] for bond in bonds]
                for (ring, bonds) in ring_bonds.items()
            }
        except KeyError:
            raise Exception('Please assign bond orders first.')

        neighbour_counts = self.neighbour_counts

        def is_sp2(atom_id: int) -> bool:
            return neighbour_counts[atom_id] == 3

        def is_hucklel_compatible(bond_orders: List[int]) -> bool:
            '''
            Implement the Huckel rule of aromaticity: 4n +2.
            Source: https://en.wikipedia.org/wiki/Hückel%27s_rule
            '''
            return (sum([2 for bond_order in bond_orders if bond_order == 2]) - 2) % 4 == 0

        def is_bond_sequence_aromatic(bond_orders: List[int]) -> bool:
            '''
            For even-membered ring, ensure alternance of single and double bonds.
            '''
            cyclic_bond_orders = [bond_orders[-1]] + list(bond_orders) + [bond_orders[0]]
            for pair_of_bond_orders in zip(cyclic_bond_orders, [cyclic_bond_orders[-1]] + cyclic_bond_orders[:-1]):
                if set(pair_of_bond_orders) == {1, 2}:
                    continue
                else:
                    return False
            else:
                return True

        def is_aromatic_ring(ring: List[int]) -> bool:
            if len(ring) % 2 == 0:
                # For even-membered rings, ensure alternance of single and double bonds, and ensure Huckel's rule is enforced.
                return is_bond_sequence_aromatic(ring_bond_orders[ring]) and is_hucklel_compatible(ring_bond_orders[ring])
            else:
                # For odd-membered rings, include sp2 (pi electrons) lone pairs, and ensure Huckel's rule is enforced.
                non_bonded_pairs = reduce(
                    lambda acc, e: acc + e,
                    [[2 for _ in range(0, self.non_bonded_electrons[atom_id] // 2)] for atom_id in ring if is_sp2(atom_id)],
                    [],
                )
                return is_hucklel_compatible(ring_bond_orders[ring] + non_bonded_pairs)

        # set aromatic flag and bond order in atom and bond objects
        for ring in rings:
            if is_aromatic_ring(ring):
                # update atom properties for aromaticity
                atom_prop_dict = {a: {'is_aromatic': True} for a in ring}
                self.set_atom_attributes(atom_prop_dict)

                # update bond properties for aromaticity
                bond_prop_dict = {(a1, a2): {'order': AROMATIC_BOND_ORDER, 'is_aromatic': True} for a1, a2 in ring_bonds[ring]}
                self.set_bond_attributes(bond_prop_dict)

    def assign_hybridisations_and_valences(self):
        """
        Use valency and non-bonded electron info to assign
        hybdridisations to each atom.
        :return:
        """

        assert all([x is not None for x in self.non_bonded_electrons])

        neighbour_counts = self.neighbour_counts

        for atom in self.atom_objects:

            # assign valences
            atom.valence = neighbour_counts[atom.get_index()]

            # assign hybridisations
            if max(FULL_VALENCES[atom.element.upper()]) > 1:
                atom.hybridisation = neighbour_counts[atom.get_index()] + self.non_bonded_electrons[atom.get_index()] // 2 - 1
            else:
                atom.hybridisation = 0  # monovalent atoms

    def assign_conjugated_atoms(self):
        """
        Use hybridisation info of atoms to determine whether given
        heavy atoms are conjugated or not.
        :return:
        """

        assert all([x is not None for x in self.hybridisations])
        assert all([x is not None for x in self.non_bonded_electrons])

        first_neighbours = self.first_neighbours
        neighbour_counts = self.neighbour_counts

        conjugated_atoms = {}
        hybridisation_scores = {}
        for atom in self.atom_objects:

            num_heavy_atom_neighbours = len([neighbour_id for neighbour_id in first_neighbours[atom.get_index()]
                                             if self.get_atom(neighbour_id).element != 'H'])

            if num_heavy_atom_neighbours > 1:

                # get normalised hybridisation score:
                # (sum(hybridisations of atom and its neighbours) - No. Lone Pairs) / Number of atoms involved in calc
                # this was designed to account for conjugation involving C/N/O as a central atom,
                # giving a numerical score which has been tested to distinguish between conjugated and non-conjugated atoms
                hybridisation_sum = sum([self.hybridisations[atom.get_index()]] + [self.hybridisations[neighbour_id]
                                                                             for neighbour_id in first_neighbours[atom.get_index()]
                                                                             if max(FULL_VALENCES[self.get_atom(neighbour_id).element.upper()]) > 1])

                num_heavy_atoms_in_calc = (len([neighbour_id
                                            for neighbour_id in first_neighbours[atom.get_index()]
                                            if self.get_atom(neighbour_id).element != 'H']) + 1)

                normalised_hybridisation = (hybridisation_sum - self.non_bonded_electrons[atom.get_index()] // 2) / num_heavy_atoms_in_calc

                hybridisation_scores[atom.get_index()] = normalised_hybridisation

                # Conjugation variable CJ = 1 if normalised_hybridisation <= 2.25
                self.get_atom(atom.get_index()).is_conjugated = 1 if normalised_hybridisation <= 2.25 else 0

            else:
                self.get_atom(atom.get_index()).is_conjugated = 0

        # print("Norm Hybridisation Scores: ", hybridisation_scores)
        # print("Conjugated atoms: ", self.conjugated_atoms)

    def get_fragments_around_elements(self, elements: set,
                                      max_depth: int,
                                      restrict_arom: bool = False) -> graph:
        """
        Use a Breadth-First Traversal of the molecular graph around groups of interest
        and return the molecular fragments covered by the traversal. Useful for analysis
        of regions of a molecule.

        :param elements: elements used to specify where to start each search
        :param max_depth: the maximum depth of the search about each element
        :return: frag_marked: a dictionary of {central atom id --> set(atom ids in the fragment)}

        TODO: in the future we can make this more general than just using elements.
        """

        marked_frags = {}

        # run graph traversal search starting at all groups
        for e in elements:

            # create list of all atom indices of each specific group
            g_ids = [a_id for a_id, a in self.atoms.items() if a.element == e]

            # do BFS search starting from each atom of the given element type
            for g_id in g_ids:
                search_queue = queue.Queue()
                search_queue.put((g_id, 0))  # queue item is: (a_id, depth)
                marked_frags[g_id] = set()
                while not search_queue.empty():

                    # pop queue
                    a_id, depth = search_queue.get()

                    # don't mark aromatic atoms, if specified
                    if restrict_arom and a_id in self.aromatic_atoms:
                        continue

                    # add atom id to marked group fragment and get neighbouring ids
                    marked_frags[g_id].add(a_id)
                    n_ids = set(self.graph.neighbors(a_id))

                    # get H atom ids and heavy atom neighbour ids
                    h_ids = set(n_id for n_id in n_ids if self.get_atom(n_id).element == 'H')
                    heavy_ids = n_ids.difference(h_ids)

                    # add h_ids to marked fragment
                    # for h_id in h_ids:
                    #     marked_frags[g_id].add(h_id)

                    if depth == max_depth:
                        # stop BFS at max depth
                        pass
                    else:
                        # add unvisited neighbours to search
                        for n_id in heavy_ids:
                            if not n_id in marked_frags[g_id]:
                                search_queue.put((n_id, depth + 1))

        # add all marked fragments to fragments dictionary, as sub-graphs
        for g_id, a_ids in marked_frags.items():
            self._fragments[f'marked_{g_id}'] = self.graph.subgraph(a_ids)
        return marked_frags

    def add_fragment(self, index: str, atom_ids: set):
        """
        Adds an indexed fragment as a subgraph view of the molecular graph
        containing only atoms in the input list. Uses atom names as indices.
        The returned subgraph view's attributes are linked to the base graph.
        :return: nx.graph sub-graph view of specified fragment.
        """
        fragment = self.graph.subgraph(atom_ids)
        self._fragments[index] = fragment
        return fragment

    def get_fragment(self, index: str):
        return self._fragments.get(index)

    def write_gml(self, fpath: str):
        """
        Save molecular graph as GML file.
        :param fpath: the file path to save the molecular graph to.
        """

        # TODO: work in progress, need Atom dictionaries to be fixed for this to work
        nx.write_gml(self.graph, fpath)


class _3DChemicalObj(_2DChemicalObj):
    def __init__(self, atoms, bonds, name: str = '', net_charge=None):
        super().__init__(atoms, bonds, name, net_charge)

    def add_atom(self, atom: _Atom) -> None:
        if not isinstance(atom, Atom2D):
            # not sure if we actually want to add atoms this way
            # might make it  easier to enforce minimum information
            raise TypeError(f'atom must be of type Atom2D, but is of type f{type(atom)}')
        if atom.name in self._graph.nodes:
            raise IndexError # Error type subject to change
        self._graph.add_node(atom.name)
        self._graph._node[atom.name] = atom

    # def add_bond(self, atom1_name: str, atom2_name: str, bond: Bond3D) -> None:
    #     """
    #     :param bond:
    #     :return:
    #     """
    #     if not isinstance(bond, Bond3D):
    #         raise TypeError('bond must be of type Bond')
    #
    #     if atom1_name not in self._graph.nodes or atom2_name not in self._graph.nodes:
    #         raise IndexError
    #
    #     self._graph.add_edge(atom1_name, atom2_name, bond)

    @property
    def atom_coord_matrix(self):
        return np.array(
            [atom.coordinates for atom in self.atom_objects]
        )
