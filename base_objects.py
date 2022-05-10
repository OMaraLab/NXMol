import networkx as nx
from typing import List, Union

from atombond import Atom2D, Atom3D
from atombond import Bond2D, Bond3D

# TODO: might rework this to replace the foactory classes for proper integration

class _2DChemicalObj:
    """
    Class representing a molecular entity, i.e. with a unique structural connectivity
    represented as a graph and stereo-isomeric form. This object contains a networkx graph to represent
    te . This can be parsed to numerous common string formats.
    """
    def __init__(self,atoms: List[Atom2D] = [],
                 bonds: List[Union[str, str, Bond2D]] = [],
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
        return self.graph.nodes

    @property
    def bonds(self):
        return self.graph.edges

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, value):
        self._graph = value

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
        TODO Need too decide if a bond will reference atom names or actual atom elements, or just elements
        and then adding a bond will create the link
        :param bond:
        :return:
        """
        if not isinstance(bond, Bond2D):
            raise TypeError('bond must be of type Bond')

        if atom1_name not in self._graph.nodes or atom2_name not in self._graph.nodes:
            raise IndexError

        self._graph.add_edge(atom1_name, atom2_name, bond)



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