"""
Contains abstract and implemented classes representing atoms and bonds.
"""
import copy
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
                 index=None,
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
        if index is None:
            index = {}
        self.element = element
        # may or may not check element types

        self._index = index
        self._index['name'] = name
        self._attributes = {}

        # self.name = name
        self.valence = None
        self.formal_charge = None
        self.non_bonded_electrons = None
        self.hybridisation = None
        self.is_aromatic = None
        self.is_conjugated = None
        self.partial_charge = None
        self.radical_electrons = 0
        self.stereo = None

        if 'valence' in kwargs:
            self.valence = kwargs['valence']
        if 'formal_charge' in kwargs:
            self.formal_charge = kwargs['formal_charge']
        if 'non_bonded_electrons' in kwargs:
            self.non_bonded_electrons = kwargs['non_bonded_electrons']
        if 'hybridisation' in kwargs:
            self.hybridisation = kwargs['hybridisation']
        if 'is_aromatic' in kwargs:
            self.is_aromatic = kwargs['is_aromatic']
        if 'is_conjugated' in kwargs:
            self.is_conjugated = kwargs['is_conjugated']
        if 'partial_charge' in kwargs:
            self.partial_charge= kwargs['partial_charge']

    def get_index(self, id_type: str = 'name'):
        """
        Method for getting the internal id by accessing the internal index dictionary
        :param id_type:
        :return:
        """
        if id_type in set(self._index.keys()):
            return self._index[id_type]
        else:
            raise AtomIndexError(f'Id type {id_type} not associated with this atom')

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

    # def __copy__(self):
    #     cls = self.__class__
    #     result = cls.__new__(cls)
    #     result.__dict__.update(self.__dict__)
    #     return result

    def copy(self):
        return self.__dict__.copy()

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


class RDKitAtom(_Atom):

    def __init__(self,
                 name,
                 element,
                 degree,
                 valence,
                 formal_charge,
                 hybridisation,
                 is_aromatic):
        super().__init__(name, element)

        self.degree = degree
        self.valence = valence
        self.formal_charge = formal_charge
        self.hybridisation = hybridisation
        self.is_aromatic = is_aromatic

    def GetDegree(self):
        return self.degree

    def GetImplicitvalence(self):
        return self.valence

    def GetFormalCharge(self):
        return self.formal_charge

    def GetNumRadicalElectrons(self):
        # TODO: Hard coded to zero at this point (we are not allowing radicals)
        return 0

    def GetHybridisation(self):
        return self.hybridisation

    def GetProp(self):
        return

    def HasProp(self):
        return


class _Bond:

    def __init__(self, **kwargs):
        self._attributes = {}
        self._atoms = set()

        if 'order' in kwargs:
            self.order = kwargs['order']

    def __getitem__(self, item):
        # TODO: Might just make this class a subclass of dictionary
        # Will prbably need to set up this for networkx to properly interface
        return self._attributes.__getitem__(item)

    def __setitem__(self, key, value):
        # might need to set this up given the way networkx interfaces
        self._attributes.__setitem__(key, value)

    def __contains__(self, item):
        return self._attributes.__contains__(item)

    def copy(self):
        return self.__dict__.copy()

    def update(self):
        raise NotImplementedError

    def get(self, *args, **kwargs):
        return self._attributes.get(*args, **kwargs)


class Bond2D(_Bond):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Bond3D(_Bond):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RDKitBond(_Bond):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def GetBeginAtomIdx(self):
        return

    def GetEndAtomIdx(self):
        return

    def GetBondType(self):
        return

    def GetIsConjugated(self):
        return

    def isInRing(self):
        return

    def GetStereo(self):
        return
