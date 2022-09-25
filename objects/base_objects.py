from sys import stderr

import networkx as nx
from typing import List, Union, Optional, TextIO, Tuple, Any, Iterable

import pulp
from matplotlib import pyplot as plt

from chemistry_data_structure.helpers.chem import ELECTRONEGATIVITIES, VALENCE_ELECTRONS
from chemistry_data_structure.helpers.io import write_to_debug
from chemistry_data_structure.objects.atom_bond import Atom2D, Atom3D, _Bond, _Atom
from chemistry_data_structure.objects.atom_bond import Bond2D, Bond3D

# TODO: might rework this to replace the factory classes for proper integration
ELEMENT_COLOURS = {'H': '#eeeeee', 'C': 'grey', 'O': '#ff91a4', 'N': '#5dabf5'}

class _2DChemicalObj:
    """
    Class representing a molecular entity, i.e. with a unique structural connectivity
    represented as a graph and stereo-isomeric form. This object contains a networkx graph to represent
    te . This can be parsed to numerous common string formats.
    """
    def __init__(self,
                 atoms: List[_Atom] = [],
                 bonds: List[Union[str, str, _Bond]] = [],
                 name: str = ''
                 ):

        # iterate for all init methods

        # init graph
        self._name = name
        self._graph = nx.Graph()
        if atoms:
            self._graph.add_nodes_from([a.name for a in atoms])
            for a in atoms:
                self._graph._node[a.name] = a

            if bonds:
                self._graph.add_edges_from([(a1, a2) for a1, a2, _ in bonds])
                for a1, a2, bond in bonds:
                    self._graph._adj[a1][a2] = bond
                    self._graph._adj[a2][a1] = bond

        # other properties

    @property
    def atoms(self):
        # TODO: we should modify this to be atom_ids, just need to refactor
        return self.graph.nodes

    @property
    def atom_objects(self):
        return self.graph.nodes.values()

    @property
    def num_atoms(self):
        return len(self.graph.nodes)

    @property
    def bonds(self):
        return self.graph.edges

    @property
    def bond_objects(self):
        return self.graph.edges.values()

    @property
    def graph(self):
        return self._graph

    @property
    def name(self):
        return self._name

    @property
    def bond_orders(self):
        return {frozenset(bond_ids): self.get_bond(bond_ids[0], bond_ids[1]).order for bond_ids in self.bonds}

    @property
    def formal_charges(self):
        return {atom_id: self.get_atom(atom_id).formal_charge for atom_id in self.atoms}

    @property
    def non_bonded_electrons(self):
        return {atom_id: self.get_atom(atom_id).non_bonded_electrons for atom_id in self.atoms}

    # @graph.setter
    # def graph(self, value):
    #     self._graph = value

    def add_atom(self, atom: Atom2D) -> None:
        if not isinstance(atom, Atom2D): # not sure if we actually want to add atoms this way
            # might make it  easier to enforce minimum information
            raise TypeError('atom must be of type Atom2D')
        if atom.name in self._graph.nodes:
            raise IndexError # Error type subject to change
        self._graph.add_node(atom.name)
        self._graph._node[atom.name] = atom

    def add_bond(self, atom1_name: str, atom2_name: str, bond: Bond2D) -> None:
        """
        :param bond:
        :return:
        """
        if not isinstance(bond, Bond2D):
            raise TypeError('bond must be of type Bond')

        if atom1_name not in self._graph.nodes or atom2_name not in self._graph.nodes:
            raise IndexError

        self._graph.add_edge(atom1_name, atom2_name, bond)

    def get_atom(self, index: Any, index_type='name'):
        if index_type == 'name':
            return self._graph.nodes[index]
        else:
            return [a for a in self._graph._node.values() if a._index[index_type] == index][0]

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
            return [[a for a in self._graph._node.values() if a._index[index_type]==i][0] for i in index]

    def get_heavy_atoms(self):
        """
        Returns all heavy atoms of the molecule.
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

    def get_rings(self):
        """
        Get atoms in rings
        :return:
        """
        return list(map(tuple, nx.cycle_basis(self._graph)))

    def get_backbone_graph(self):
        """
        Returns a copy of a subgraph view of the molecular graph
        containing only heavy atoms. Uses atom names as indices.
        The returned subgraph view's attributes are linked to the base graph.
        :return: nx.graph
        """
        heavy_atoms = [a.name for a in self.atoms.values() if a.element != 'H']
        return self.graph.subgraph(heavy_atoms)

    def draw_graph(self,
                   fixed_heavy_atoms: dict = None,
                   backbone_only: bool = False,
                   show: bool = True,
                   save_fp: str = None):
        """
        Draws the molecular graph in kamada kawai layout.
        :param show: if true, show plot, otherwise don't
        :param save_name: if specified, save the png to the given fp.
        """

        if backbone_only:
            graph = self.get_backbone_graph()
            formal_charges = {a.get_index(): a.formal_charge for a in graph.nodes.values()}
            bond_orders = {frozenset(bond_ids): self.get_bond(bond_ids[0], bond_ids[1]).order for bond_ids in graph.edges()}
        else:
            graph = self._graph
            formal_charges = self.formal_charges
            bond_orders = self.bond_orders

        # create colour map of element colours
        colour_map = []
        for atom_name in graph.nodes():
            colour_map.append(ELEMENT_COLOURS[atom_name.strip("0123456789")])

        # setup layouts
        if fixed_heavy_atoms is not None:
            pos = nx.spring_layout(graph, pos=fixed_heavy_atoms, fixed=fixed_heavy_atoms.keys())
        else:
            pos = nx.spring_layout(graph)

        #pos = nx.spring_layout(self._graph)
        if backbone_only:
            offset_pos = {k: (v[0] + 0.5, v[1] + 0.15) for k, v in pos.items()}
        else:
            offset_pos = {k: (v[0] + 0.65, v[1] + 0.2) for k, v in pos.items()}


        # draw graph, with node and edge labels
        plt.cla()
        nx.draw(graph, pos=pos, with_labels=True, font_size=10,
                font_color='black', node_color=colour_map, node_size=500, edge_color='black')
        node_path_coll = plt.gca().collections[0]
        node_path_coll.set_edgecolor("#afafaf")
        node_path_coll.set_lw(2)
        edge_path_coll = plt.gca().collections[1]
        edge_path_coll.set_lw(1.5)
        nx.draw_networkx_labels(graph, offset_pos, formal_charges,
                                font_color='red', font_size=10)
        nx.draw_networkx_edge_labels(graph, pos, bond_orders)

        if backbone_only:
            plt.margins(x=0.2, y=0.2)
        else:
            plt.margins(x=0.1, y=0.1)

        # optionally show or save molecular graph
        if show:
            plt.show()
            plt.cla()
        if save_fp is not None:
            plt.savefig(save_fp, format='png')
            plt.cla()

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

    def get_neighbour_counts(self, element: str):
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

    def get_element_count(self, element: str):
        """
        Returns the number of atoms of a certain type of element.
        :param element: element to count
        :return: int count
        """
        return len([a for a in self.atoms if self.get_atom(a).element == element])

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

        # TODO: Make bonds have numerical indices as well...

        non_allene_atoms = {}
        for atom in self.atoms.values():
            if disallow_allenes_completely:
                atom_bonds = [bond for bond in self.bonds if atom.get_index() in bond]
                if atom.element == 'C' and len(atom_bonds) == 2:
                    non_allene_atoms[atom] = atom_bonds

        # ===== VARIABLES =====

        # formal charges of atoms
        charges = {
            atom.get_index(): LpVariable("C_{i}".format(i=atom.get_index()), -MAX_ABSOLUTE_CHARGE, MAX_ABSOLUTE_CHARGE, LpInteger)
            for atom in self.atoms.values()
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

    def weave_featurize_molecule(self):
        """
        Create Weave Convultuion featurization of molecule object.
        This code is sourced from DeepChem, modified for the current data structure.
        TODO: THIS!
        :return:
        """
        return


class _3DChemicalObj(_2DChemicalObj):
    def __init__(self, atoms, bonds, name: str = ''):
        super().__init__(atoms, bonds, name)

    def add_atom(self, atom: Atom3D):
        if not isinstance(atom, Atom3D): # not sure if we actually want to add atoms this way
            # might make it  easier to enforce minimum information
            raise TypeError('atom must be of type Atom2D')
        
    def add_bond(self, bond: Bond3D):
        if not isinstance(bond, Bond3D):
            raise TypeError('bond must be of type Bond')