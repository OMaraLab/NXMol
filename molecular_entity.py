import networkx as nx
from base_objects import _2DChemicalObj, _3DChemicalObj
from atom_bond import Atom2D, Atom3D, Bond2D, Bond3D
from typing import List, Union


class Molecule2D(_2DChemicalObj):
    """
    Represents the 2D structure of a molecular entity.
    """

    def __init__(self,
                 atoms: List[Atom2D] = [],
                 bonds: List[Union[str, str, Bond2D]] = [],
                 name: str = ''
                 ):

        # init super class
        super().__init__(atoms, bonds,  name)
        self.attributes = {}
        self.angles = {}
        self.dihedrals = {}
        self.qmProperties = {}


class Molecule3D(_3DChemicalObj, Molecule2D):
    def __init__(self, atoms: List[Atom2D] = [],
                 bonds: List[Union[str, str, Bond2D]] = [],
                 name: str = ''
                 ):
        super().__init__(atoms, bonds,name)

    def rmsdFit(self):
        return

    def writePDB(self):
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


