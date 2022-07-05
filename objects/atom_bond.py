"""
Contains abstract and implemented classes representing atoms and bonds.
"""

from typing import Tuple

Coordinate = Tuple[float, float, float]

class AtomIndexError(Exception):
    pass

class AtomIndexType(Exception):
    pass

class _Atom:
    def __init__(self,
                 name,
                 element,
                 index: dict = {},
                 **kwargs):
        # TODO: Possibly rework the atom class to replace the atom dict factory for more consistency
        self.element = element
        # may or may not check element types

        self._index = index
        self._index['name'] = name
        self._attributes = {}

        # self.name = name
        self.valence = None

        if 'valence' in kwargs:
            self.valence = kwargs['valence']

    def get_index(self, id_type):
        if id_type in self._index:
            return self._index[id_type]
        else:
            raise AtomIndexError(f'Atom not instantiated using method associated with {id_type}')

    @property
    def name(self):
        return self._index['name']

    @name.setter
    def name(self, value):
        self._index['name'] = value

    def get_index(self, index_type):
        if index_type not in self._index.keys():
            raise AtomIndexError('Index type not associated for this atom')
        return self._index[index_type]

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.name}", "{self.element}")'

    def __getitem__(self, item):
        # TODO: Might just make this class a subclass of dictionary
        # Will prbably need to set up this for networkx to properly interface
        return self._attributes.__getitem__(item)

    def __setitem__(self, key, value):
        # might need to set this up given the way networkx interfaces
        self._attributes.__setitem__(key, value)

    def upate(self):
        raise NotImplemented


class Atom2D(_Atom):

    def __init__(self, name, element,  **kwargs):
        super().__init__(name, element,  **kwargs)


class Atom3D(_Atom):

    def __init__(self, name, element, coordinates, **kwargs):
        super().__init__(name, element, **kwargs)
        self.coordinates = coordinates

class _Bond:

    def __init__(self):
        pass

class Bond2D(_Bond):
    def __init__(self):
        super().__init__()

class Bond3D(_Bond):
    def __init__(self):
        super().__init__()