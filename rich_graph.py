import networkx as nx


class RichGraph(nx.Graph):
    """
    Class representing a rich graph object. This object represents a networkX graph with additional information,
    which can be parsed to numerous common file formats and be controlled more tightly.
    """

    def __init__(self, rules, attributes: dict):



        super().__init__(attributes)

        # can specify rules to adhere by in creating graph data structure, specified by a json file
        self.rules = {}

        # can store information about sub graphs
        self.sub_graphs = {}  # or sub-graph? i.e. can store dihedrals as an index of (a1, a2, a3) or (e1, e2, e3)


class SubGraph:
    """
    Represents information about a fragment of the graph object. I.e. a collection of nodes/edges.
    Subgraph?
    """

    def __init__(self, **kwargs):
        self.index = (None, None)
        self.attributes = {kwargs}