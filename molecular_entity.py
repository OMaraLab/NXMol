import networkx as nx


class MolecularEntity(nx.Graph):
    """
    Class representing a molecular entity, i.e. with a unique structural connectivity and
    stereo-isomeric form. This object inherits from networkx to represent
    a molecule as a graph. This can be parsed to numerous common string formats.
    """

    def __init__(self):

        # init super class
        super(nx.Graph, self).__init__(self)



        self.dihedrals = {}  # but these would be of a conformer??
        self.conformers = {}  # ??


class Conformer(MolecularEntity):
    """
    Represents an individual conformational state of a molecular entity.
    """

    def __init__(self, **kwargs):

        # init super class
        super(MolecularEntity, self).__init__(self)
        self.attributes = {kwargs}


class Fragment(MolecularEntity):
    """
    Represents information about a fragment of the graph object. I.e. a collection of nodes/edges.
    """

    def __init__(self, **kwargs):
        self.index = (None, None)
        self.attributes = {kwargs}


if __name__ == "__main__":

    molecule = MolecularEntity()
    molecule.add_edge(2, 1)
    print(molecule[2][1])
