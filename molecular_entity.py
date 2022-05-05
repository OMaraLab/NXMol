import networkx as nx


class MolecularEntity(nx.Graph):
    """
    Class representing a molecular entity, i.e. with a unique structural connectivity and
    stereo-isomeric form. This object inherits from networkx to represent
    a molecule as a graph. This can be parsed to numerous common string formats.
    """

    @property
    def atoms(self):
        return self.nodes

    @property
    def bonds(self):
        return self.edges

    def __init__(self):

        # init super class
        super().__init__()

        self.dihedrals = {}  # but these would be of a conformer??
        self.conformers = {}  # ??


class Conformer(MolecularEntity):
    """
    Represents an individual conformational state of a molecular entity.
    """

    def __init__(self, **kwargs):

        # init super class
        super().__init__()
        self.attributes = {kwargs}


class Fragment(MolecularEntity):
    """
    Represents information about a fragment of the graph object. I.e. a collection of nodes/edges.
    """

    def __init__(self, **kwargs):
        self.index = (None, None)
        self.attributes = {kwargs}


class Atom():

    def __init__(self):
        pass


class Bond():

    def __init__(self):
        pass


if __name__ == "__main__":

    # testing

    molecule = MolecularEntity()
    #molecule.add_edge()
    molecule.add_node(1)
    molecule.add_node(2)
    molecule.add_edge(2, 1)
    print(molecule.atoms)
    print(molecule.bonds)
    print(molecule.nodes)
    print(molecule.edges)

    print("Atoms: ")
    print(molecule[1])
    print(molecule[2])

    print("Bonds: ")
    print(molecule[1][2])
    print(molecule[2][1])
    e = [('a', 'b', 0.3), ('b', 'c', 0.9), ('a', 'c', 0.5), ('c', 'd', 1.2)]
    molecule.add_weighted_edges_from(e)
    print(nx.dijkstra_path(molecule, 'a', 'd'))

