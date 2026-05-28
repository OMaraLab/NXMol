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
        # TODO: add all dictionary methods to _Atom class
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
        return self.__dict__[item]

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
        self.radius = kwargs["radius"]
        self.mass = kwargs["mass"]
        self.electronegativity = kwargs["electronegativity"]
        self.atomic_number = kwargs["atomic_number"]

    def __str__(self):
        return f"Atom3D(Name: {self.name}, Element: {self.element}, Coordinates: {self.coordinates}, " \
               f"Full Valence: {self.full_valence}," \
               f"Valence: {self.valence}, Formal Charge: {self.formal_charge}, Valence Electrons: {self.valence_electrons}," \
               f"Non Bonded Electrons: {self.non_bonded_electrons}, Hybridisation: {self.hybridisation}," \
               f"Is Aromatic: {self.is_aromatic}, Is Conjugated: {self.is_conjugated}, Radical Electrons: {self.radical_electrons}," \
               f"Chirality: {self.chirality}, Index: {self.index})"

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


class _Bond(dict):

    def __init__(self,
                 atoms: set,
                 **kwargs):

        # assign atoms
        assert atoms is not None, "Bond atoms cannot be none..."
        self.atoms = atoms

        # properties
        self.order: int = None
        self.is_aromatic: bool = None

        # general attributes dictionary
        self.attributes = {}

        if 'order' in kwargs:
            self.order = kwargs['order']

    def get_order(self) -> int:
        return self.order

    def get_atoms(self):
        return self.atoms

    def get_atoms_list(self):
        return list(self.atoms)

    def set_order(self, order: int):
        tmp = self.order
        self.order = order
        print(f"Order was set from {tmp} to {self.order}")

    def get(self, *args, **kwargs):
        return self.attributes.get(*args, **kwargs)

    def items(self):

        # format dictionary to stringify None and to replace sets with lists
        # str_format_dict = {}
        # for key, value in self.__dict__.items():
        #
        #     if value is None:
        #         str_format_dict[key] = 'None'
        #     elif isinstance(value, set):
        #         str_format_dict[key] = list(value)
        #     else:
        #         str_format_dict[key] = value

        #print(str_format_dict.items())
        return self.__dict__.items()#str_format_dict.items()

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

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        # might need to set this up given the way networkx interfaces
        self.attributes.__setitem__(key, value)
        # self.__dict__[key] = value

    def __contains__(self, item):
        return self.attributes.__contains__(item)


class Bond2D(_Bond):

    def __init__(self,
                 atoms: set,
                 **kwargs):
        super().__init__(atoms, **kwargs)

    def __str__(self):
        return f"Bond2D({self.get_atoms_list()[0]}-{self.get_atoms_list()[1]} with Order {self.get_order()})"


class Bond3D(_Bond):

    def __init__(self, atoms: set, **kwargs):
        super().__init__(atoms, **kwargs)

    def __str__(self):
        return f"Bond3D({self.get_atoms_list()[0]}-{self.get_atoms_list()[1]} with Order {self.get_order()})"


class RDKitBond(_Bond):
    pass
