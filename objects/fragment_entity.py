from chemistry_data_structure.objects.base_objects import _2DChemicalObj, _3DChemicalObj


class Fragment_2D(_2DChemicalObj):
    """
    Represents information about a fragment of the graph object. I.e. a collection of nodes/edges.
    """

    def __init__(self):
        super().__init__()
        self.index = (None, None)


class Fragment_3D(_3DChemicalObj, Fragment_2D):
    def __init__(self):
        super().__init__()