import networkx as nx


class _2DChemicalObj:
    """
    Class representing a molecular entity, i.e. with a unique structural connectivity
    represented as a graph and stereo-isomeric form. This object contains a networkx graph to represent
    te . This can be parsed to numerous common string formats.
    """
    def __init__(self, **kwargs):

        if 'smiles' in kwargs:
            # init using smils
            pass
        # iterate for all init methods


        # init graph
        self._graph = nx.Graph()

        # conformers of this molecular entity
        self.conformers = {}

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

    def add_atom(self, atom: Atom2D):
        if not isinstance(atom, Atom2D): # not sure if we actually want to add atoms this way
            # might make it  easier to enforce minimum information
            raise TypeError('atom must be of type Atom2D')

    def add_bond(self, bond: Bond):
        if not isinstance(bond, bond):
            raise TypeError('bond must be of type Bond')


class _3DChemicalObj(_2DChemicalObj):
    def __init__(self):
        super().__init__()

    def add_atom(self, atom: Atom3D):
        if not isinstance(atom, Atom3D): # not sure if we actually want to add atoms this way
            # might make it  easier to enforce minimum information
            raise TypeError('atom must be of type Atom2D')


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

class Fragment_2D(_2DChemicalObj):
    """
    Represents information about a fragment of the graph object. I.e. a collection of nodes/edges.
    """

    def __init__(self):
        super().__init__()
        self.index = (None, None)

class Fragment_3D(_3DChemicalObj,Fragment_2D):
    def __init__(self):
        super().__init__()


class _Atom:
    element = None
    valence = None

    def get(self):
        pass

    def __getitem__(self, item):
        # Will prbably need to set up this for networkx to properly interface
        pass



class Atom2D(_Atom):

    def __init__(self):
        pass


class Atom3D(_Atom):

    def __init__(self):
        pass

class Bond:

    def __init__(self):
        pass


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

