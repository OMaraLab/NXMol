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
                 name: str,
                 element: str,
                 index: dict = {},
                 **kwargs):
        """
        The base class used for atom types. These are used as the second level data structure within the graph objects
        of chemical objects
        :param name: The unique name of the atom, can be any string
        :param element: Element type of the atom, currently no restrictions on the value
        :param index: A dictionary of different unique index types for the atom, structure of ID_TYPE: VALUE
        :param kwargs:
        """
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

    def get_index(self, id_type: str):
        """
        Method for getting the internal id by accessing the internal index dictionary
        :param id_type:
        :return:
        """
        if id_type in self._index:
            return self._index[id_type]
        else:
            raise AtomIndexError(f'Id type{id_type} not associated with this atom')

    @property
    def name(self):
        """
        method for retrieving the atom name
        :return:
        """
        return self._index['name']

    @name.setter
    def name(self, value):
        """
        Method for renaming atoms
        :param value:
        :return:
        """
        self._index['name'] = value

    def __repr__(self):
        """
        repr method, displays atom name and element
        :return:
        """
        return f'{self.__class__.__name__}("{self.name}", "{self.element}")'

    def __getitem__(self, item):
        """
        Pass thorough dictionary methods to the attributes dictionary to maintain compatibility with networkx
        Networkx requires dict of dict of dict structure
        merges index and attributes for plotting
        :param item:
        :return:
        """
        # TODO: Might just make this class a subclass of dictionary
        # Will prbably need to set up this for networkx to properly interface
        # TODO raise warning if overlap between index and dictionary
        return {**self._attributes, **self._index}.__getitem__(item)

    def __setitem__(self, key, value):
        """
        Pass thorough dictionary methods to the attributes dictionary to maintain compatibility with networkx
        Networkx requires dict of dict of dict structure
        :param key:
        :param value:
        :return:
        """
        # TODO raise warning if overlap between index and dictionary
        self._attributes.__setitem__(key, value)

    def __contains__(self, item):
        """
        Pass thorough dictionary methods to the attributes dictionary to maintain compatibility with networkx
        Networkx requires dict of dict of dict structure
        :param key:
        :param value:
        :return:
        """
        return self._attributes.__contains__(item)

    def update(self):
        """
        Pass thorough dictionary methods to the attributes dictionary to maintain compatibility with networkx
        Networkx requires dict of dict of dict structure
        :param key:
        :param value:
        :return:
        """
        raise NotImplemented


class Atom2D(_Atom):
    """
    2D Atom representation, has no additional functionality from base class
    """
    def __init__(self, name, element, **kwargs):
        super().__init__(name, element, **kwargs)


class Atom3D(_Atom):

    def __init__(self, name, element, coordinates, **kwargs):
        super().__init__(name, element, **kwargs)
        self.coordinates = coordinates


class _Bond:

    def __init__(self):
        self._attributes = {}

    def __getitem__(self, item):
        # TODO: Might just make this class a subclass of dictionary
        # Will prbably need to set up this for networkx to properly interface
        return self._attributes.__getitem__(item)

    def __setitem__(self, key, value):
        # might need to set this up given the way networkx interfaces
        self._attributes.__setitem__(key, value)

    def __contains__(self, item):
        return self._attributes.__contains__(item)

    def update(self):
        raise NotImplementedError

    def get(self, *args, **kwargs):
        return self._attributes.get(*args, **kwargs)


class Bond2D(_Bond):
    def __init__(self):
        super().__init__()


class Bond3D(_Bond):
    def __init__(self):
        super().__init__()
