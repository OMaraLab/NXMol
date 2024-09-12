import statistics
from typing import List, Tuple, Callable, Sequence, Set

from networkx import Graph
from networkx.algorithms import is_isomorphic

from atb_outputs.mol_data import MolData, MolDataFailure
from chemical_equivalence.calcChemEquivalency import getChemEquivGroups
from chemistry_data_structure.objects.molecular_entity import Molecule3D
from chemistry_data_structure.objects.atom_bond import _Atom, _Bond
from chemistry_data_structure.objects.base_objects import _2DChemicalObj


def lewis_graph(molecule: _2DChemicalObj, use_non_bonded_electrons: bool = True) -> Graph:
    '''
    Return the Lewis graph of a molecule (Using networkx' Graph() class).
    TODO: this will eventually be replaced once we have properly integrated the atom
        object into networkx.

        THIS HAS BEEN MADE REDUNDANT!
    '''
    G = Graph()

    for a_id in molecule.atoms:
        G.add_node(
            a_id,
            element=molecule.get_atom(a_id).element,
            non_bonded_electrons=molecule.non_bonded_electrons[a_id] if use_non_bonded_electrons else None,
        )

    for bond in molecule.bonds:
        G.add_edge(*bond, order=molecule.get_bond(bond[0], bond[1]).order)

    return G

def calc_equal_bonds(mol: Molecule3D):
    mol_data = MolData(mol)
    eq_grps, _ = getChemEquivGroups(mol_data)
    eq_grps_sorted = {}
    for i, j in mol.bonds:
        #The atom indices in getChemEquivGroups start from 1, not 0
        i = int(i)+1
        j = int(j)+1
        if (eq_grps[i], eq_grps[j]) not in eq_grps_sorted and (eq_grps[j], eq_grps[i]) not in eq_grps_sorted:
            eq_grps_sorted[(eq_grps[i], eq_grps[j])] = [(str(i-1), str(j-1))]
        elif (eq_grps[i], eq_grps[j]) in eq_grps_sorted:
            eq_grps_sorted[(eq_grps[i], eq_grps[j])].append((str(i-1), str(j-1)))
        elif (eq_grps[j], eq_grps[i]) in eq_grps_sorted:
            eq_grps_sorted[(eq_grps[j], eq_grps[i])].append((str(i-1), str(j-1)))
    mol.eq_grps_sorted = eq_grps_sorted

def cull_equal_bonds(mol: Molecule3D, aggregator="mean", graph=False):
    for v in mol.eq_grps_sorted.values():
        if len(v) > 1: 
            if aggregator == "mean":
                mean_fc = statistics.mean([mol.bonds[(i, j)].get("force_constant")
                                                            for (i, j) in v])
                for eq_bonds in v:
                    mol.bonds[eq_bonds].update(force_constant=mean_fc)
            elif aggregator == "median":
                mean_fc = statistics.mean([mol.bonds[(i, j)].get("force_constant")
                                                            for (i, j) in v])
                for eq_bonds in v:
                    mol.bonds[eq_bonds].update(force_constant=mean_fc)
        if not graph:
            for (rm1, rm2) in v[1:]:
                mol.remove_bond(rm1, rm2)

def are_atoms_equivalent(node_1: _Atom, node_2: _Atom) -> bool:
    return node_1['element'] == node_2['element'] #and node_1['non_bonded_electrons'] == node_2['non_bonded_electrons']

def are_atoms_and_formal_charges_equivalent(node_1: _Atom, node_2: _Atom) -> bool:
    return node_1['element'] == node_2['element'] and node_1['formal_charge'] == node_2['formal_charge']

def are_bonds_equivalent(edge_1: _Bond, edge_2: _Bond) -> bool:
    return edge_1['order'] == edge_2['order']


def are_graphs_isomorphic(
    graphs: Sequence[Graph],
    node_match: Callable[['Node', 'Node'], bool] = are_atoms_equivalent,
    edge_match: Callable[['Edge', 'Edge'], bool] = None  # by default do nothing
) -> bool:
    return is_isomorphic(
        *graphs,
        node_match=node_match,
        edge_match=edge_match,
    )


def unique_molecules(molecules: List[_2DChemicalObj], debug: bool = False) -> List[_2DChemicalObj]:
    '''
    Return list of one-by-one non-isomorphic graphs (based only on atom elements and connectivity)

    TODO: make this able to have modifiable uniqueness callables (i.e. from functions above)
    '''
    lewis_graphs = [
        lewis_graph(molecule)
        for molecule in molecules
    ]

    unique_molecules: List[Tuple[_2DChemicalObj, Graph]] = []

    print_if_debug = lambda *args, **kwargs: print(*args, **kwargs) if debug else None

    for (molecule_1, graph_1) in zip(molecules, lewis_graphs):
        print_if_debug('Assessing {0} (unique_molecules={1})'.format(molecule_1.name, [m.name for (m, _) in unique_molecules]))
        if all(
            not are_graphs_isomorphic(
                (graph_1, graph_2),
                node_match=are_atoms_equivalent,
            )
            for (_, graph_2) in unique_molecules
        ):
            print_if_debug('UNIQUE: {0}'.format(molecule_1.name))
            unique_molecules.append(
                (molecule_1, graph_1),
            )
        else:
            print_if_debug('NOT UNIQUE: {0}'.format(molecule_1.name))

    return [
        molecule
        for (molecule, _) in unique_molecules
    ]


def get_molecule_matches(ref_mols: Set[_2DChemicalObj], test_mols: Set[_2DChemicalObj]) -> dict:
    """
    Returns a dictionary of matches between test set of mol objects and a reference set of mol objects
    by graph isomorphism.
    :param ref_mols: the list of reference mol objects to match against
    :param test_mols: the list of test mol objects to be matched
    :return: match dictionary: dict[test_mol --> ref_mol]
    """

    # dictionary of matches {test_mol --> ref_mol...}
    matches = {}
    for ref_mol in ref_mols:
        for test_mol in test_mols:
            if are_graphs_isomorphic((ref_mol.graph, test_mol.graph), node_match=are_atoms_equivalent):
                matches[test_mol] = ref_mol

    return matches
