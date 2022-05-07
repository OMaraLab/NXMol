import networkx as nx
from base_objects import _2DChemicalObj, _3DChemicalObj


class Molecule_2D(_2DChemicalObj):
    """
    Represents an individual conformational state of a molecular entity.
    """

    def __init__(self):

        # init super class
        super().__init__()
        self.attributes = {}
        self.angles = {}
        self.dihedrals = {}
        self.qmProperties = {}

        # RMSD fit

    def rmsdFit(self):
        return

    def writePDB(self):
        return

class Molecule_3D(_3DChemicalObj, Molecule_2D):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":

    # testing

    molecule = MolecularEntity()
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

