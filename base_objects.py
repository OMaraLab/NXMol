from atombond import Atom2D, Atom3D


class _2DChemicalObj:
    """
    Class representing a molecular entity, i.e. with a unique structural connectivity
    represented as a graph and stereo-isomeric form. This object contains a networkx graph to represent
    te . This can be parsed to numerous common string formats.
    """
    def __init__(self, **kwargs):

        if 'smiles' in kwargs:
            # init using smiles
            pass
        # iterate for all init methods


        # init graph
        self._graph = nx.Graph()

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
        self._graph.add_node(atom)

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