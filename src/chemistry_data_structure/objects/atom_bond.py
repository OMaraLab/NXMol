"""
Contains abstract and implemented classes representing atoms and bonds.
"""
import copy
from typing import Tuple, Any, Mapping

Coordinate = Tuple[float, float, float]


class AtomIndexError(Exception):
    pass


class AtomIndexType(Exception):
    pass


class _Atom(dict):
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

        self.index = index
        self.index['name'] = name
        self.attributes = {}

        self.full_valence = None
        self.valence = None
        self.formal_charge: int = None
        self.valence_electrons = None
        self.non_bonded_electrons = None
        self.hybridisation: int = None
        self.is_aromatic: bool = None
        self.is_conjugated: bool = None
        self.partial_charge = None
        self.radical_electrons = 0
        self.chirality: str = None

        # TODO: for deletion probably
        self.stereo = None

        if 'full_valence' in kwargs:
            self.valence = kwargs['full_valence']
        if 'valence' in kwargs:
            self.valence = kwargs['valence']
        if 'formal_charge' in kwargs:
            self.formal_charge = kwargs['formal_charge']
        if 'valence_electrons' in kwargs:
            assert type(kwargs['valence_electrons']) == int
            self.valence_electrons = kwargs['valence_electrons']
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
        :param id_type: 'name' is default lookup (i.e. C1, H2...),
                        'pdb': pdb specific id, assigned when parsed from a pdb
                        'nid': numerical id, assigned when parsed from a pdb
        :return:
        """
        if id_type in set(self.index.keys()):
            return self.index[id_type]
        else:
            raise AtomIndexError(f'Id type {id_type} not associated with this atom')

    def set_index(self, id_type: str, value: Any, overwrite: bool = False):
        if not overwrite:
            if id_type in self.index:
                raise AtomIndexError(f'Index {id_type} already exists! specify overwrite=True to overwrite')
        self.index[id_type] = value

    @property
    def name(self):
        """
        method for retrieving the atom name
        :return:
        """
        return self.index['name']

    # @name.setter
    # def name(self, value):
    #     """
    #     Method for renaming atoms
    #     :param value:
    #     :return:
    #     """
    #     self._index['name'] = value

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
        return {**self.attributes, **self.index}.__getitem__(item)

    def __setitem__(self, key, value):
        """
        Pass thorough dictionary methods to the attributes dictionary to maintain compatibility with networkx
        Networkx requires dict of dict of dict structure
        :param key:
        :param value:
        :return:
        """
        # TODO raise warning if overlap between index and dictionary
        self.__dict__[key] = value
        #self._attributes.__setitem__(key, value)
        #print(f"Key to Update: {key} with Value: {value}")

    def __contains__(self, item):
        """
        Pass thorough dictionary methods to the attributes dictionary to maintain compatibility with networkx
        Networkx requires dict of dict of dict structure
        :param key:
        :param value:
        :return:
        """
        return self.attributes.__contains__(item)

    def items(self):
        print(self.__dict__.items())
        return self.__dict__.items()

    # def __copy__(self):
    #     cls = self.__class__
    #     result = cls.__new__(cls)
    #     result.__dict__.update(self.__dict__)
    #     return result

    def copy(self):
        return self.__dict__.copy()

    def update(self, other=None, **kwargs):
        """
        Pass thorough dictionary methods to the attributes dictionary to maintain compatibility with networkx
        Networkx requires dict of dict of dict structure
        :param key:
        :param value:
        :return:
        """
        if other is not None:
            for k, v in other.items() if isinstance(other, Mapping) else other:
                self.__setitem__(k, v)
        for k, v in kwargs.items():
            self.__setitem__(k, v)


class Atom2D(_Atom):
    """
    2D Atom representation, has no additional functionality from base class
    """

    def __init__(self, name, element, **kwargs):
        super().__init__(name, element, **kwargs)

    def __str__(self):

        return f"Atom2D(Name: {self.name}, Element: {self.element}, Full Valence: {self.full_valence}, " \
               f"Valence: {self.valence}, Formal Charge: {self.formal_charge}, Valence Electrons: {self.valence_electrons}, " \
               f"Non Bonded Electrons: {self.non_bonded_electrons}, Hybridisation: {self.hybridisation}, " \
               f"Is Aromatic: {self.is_aromatic}, Is Conjugated: {self.is_conjugated}, Radical Electrons: {self.radical_electrons}, " \
               f"Chirality: {self.chirality})"


class Atom3D(_Atom):

    def __init__(self, name, element, coordinates, **kwargs):
        super().__init__(name, element, **kwargs)
        self.coordinates: tuple = coordinates

    def __str__(self):
        return f"Atom3D(Name: {self.name}, Element: {self.element}, Coordinates: {self.coordinates}, " \
               f"Full Valence: {self.full_valence}," \
               f"Valence: {self.valence}, Formal Charge: {self.formal_charge}, Valence Electrons: {self.valence_electrons}," \
               f"Non Bonded Electrons: {self.non_bonded_electrons}, Hybridisation: {self.hybridisation}," \
               f"Is Aromatic: {self.is_aromatic}, Is Conjugated: {self.is_conjugated}, Radical Electrons: {self.radical_electrons}," \
               f"Chirality: {self.chirality})"

    def set_coordinates(self, coordinates):
        """
        Updates the coordinates of this atom.
        """
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

    def __init__(self,
                 atoms: set,
                 **kwargs):

        # assign atoms
        assert atoms is not None, "Bond atoms cannot be none..."
        self._atoms = atoms

        # properties
        self.order: int = None
        self.is_aromatic: bool = None

        # general attributes dictionary TODO: either use this or remove... consult with Callum
        self._attributes = {}

        if 'order' in kwargs:
            self.order = kwargs['order']

    def __getitem__(self, item):
        # TODO: Might just make this class a subclass of dictionary
        # Will probably need to set up this for networkx to properly interface
        return self._attributes.__getitem__(item)

    def __setitem__(self, key, value):
        # might need to set this up given the way networkx interfaces
        #self._attributes.__setitem__(key, value)
        self.__dict__[key] = value

    def __contains__(self, item):
        return self._attributes.__contains__(item)

    def copy(self):
        return self.__dict__.copy()

    def update(self, other=None, **kwargs):
        """
        Pass thorough dictionary methods to the attributes dictionary to maintain compatibility with networkx
        Networkx requires dict of dict of dict structure
        """
        if other is not None:
            for k, v in other.items() if isinstance(other, Mapping) else other:
                self.__setitem__(k, v)
        for k, v in kwargs.items():
            self.__setitem__(k, v)

    def get(self, *args, **kwargs):
        return self._attributes.get(*args, **kwargs)

    def get_order(self) -> int:
        return self.order

    def set_order(self, order: int):
        # TODO: I don't know if this is what we have in mind, i.e. getters/setters
        #   or if we want to use a more general method, but i'm writing this for use in the
        #   short term
        tmp = self.order
        self.order = order
        print(f"Order was set from {tmp} to {self.order}")

    def get_atoms(self):
        return self._atoms


class Bond2D(_Bond):

    def __init__(self,
                 atoms: set,
                 **kwargs):
        super().__init__(atoms, **kwargs)


class Bond3D(_Bond):

    def __init__(self, atoms: set, **kwargs):
        super().__init__(atoms, **kwargs)


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
