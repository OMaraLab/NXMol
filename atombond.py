class _Atom:
    def __init__(self):
        self.element = None
        self.valence = None
        self.name = ''

    def get(self):
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'

    def __getitem__(self, item):
        # Will prbably need to set up this for networkx to properly interface
        pass

    def __setitem__(self, key, value):
        # might need to set this up given the way networkx interfaces
        self.__setattr__(key, value)


class Atom2D(_Atom):

    def __init__(self):
        super().__init__()


class Atom3D(_Atom):

    def __init__(self):
        super().__init__()


class _Bond:

    def __init__(self):
        pass

class Bond2D(_Bond):
    def __init__(self):
        super().__init__()

class Bond3D(_Bond):
    def __init__(self):
        super().__init__()