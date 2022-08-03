"""
Parsers for the creation of MolecularEntity objects and their derivatives.
"""

from chemistry_data_structure.objects.molecular_entity import Molecule3D
from chemistry_data_structure.objects.atom_bond import Atom3D, Bond3D
from chemistry_data_structure.parsing.pdb import bonds_for_pdb_line, is_pdb_connect_line, pdb_atoms_in
from functools import reduce

############# mol2 Parser


def _atom_for_atom_line(line: str):
    index_str, name_str, x, y, z, sybil_atom_type, _, _, partial_charge = line.split()
    element, *_ = sybil_atom_type.split('.')

    return (
        Atom3D(
            index={'mol2': int(index_str)}, # currently an arbitrary dictionary
            name=f'{element}{index_str}',
            element=element,
            valence=None,
            coordinates=(float(x), float(y), float(z))
            # capped=True,
        ),
        float(partial_charge),
    )

AROMATIC_BOND, AMIDE_BOND = 'ar', 'am'
def _bond_for_atom_line(line: str):
    bond_label, atom_id_1, atom_id_2, bond_order_str = line.split()
    if bond_order_str == AROMATIC_BOND:
        bond_order = 1.5
    elif bond_order_str == AMIDE_BOND:
        bond_order = 1
    else:
        bond_order = int(bond_order_str)
    return ([int(atom_id_1), int(atom_id_2)], bond_order)


def mol2_to_Molecule3D(mol2_str: str) -> Molecule3D:
    """
    Primarily adapted from fragment_capping/helpers/molecule.py
    :param mol2_str:
    :return:
    """
    assert mol2_str.count('@<TRIPOS>MOLECULE'), 'Error: MOL2 file does not start with "@<TRIPOS>MOLECULE"'
    assert mol2_str.count('@<TRIPOS>MOLECULE') == 1, 'Only one molecule at a time'

    read_lines, atoms, bonds = False, [], []
    for (i, line) in enumerate(mol2_str.splitlines()):
        if i == 1:
            molecule_name = line
        elif line.startswith('@<TRIPOS>ATOM'):
            container, line_reading_fct, read_lines = atoms, _atom_for_atom_line, True
        elif line.startswith('@<TRIPOS>BOND'):
            container, line_reading_fct, read_lines = bonds, _bond_for_atom_line, True
        elif line.startswith('@'):
            read_lines = False
        else:
            if read_lines:
                container.append(line_reading_fct(line))

    bond_objects = []
    for b in bonds:
        (mol2_id1, mol2_id2), bond_order = b
        for a in atoms:
            a1_name = [a.name for (a,_) in atoms if a.get_index('mol2')==mol2_id1][0]
            a2_name = [a.name for (a,_) in atoms if a.get_index('mol2')==mol2_id2][0]
            bond_objects.append((a1_name, a2_name, Bond3D()))

    total_net_charge = sum(partial_charge for (atom, partial_charge) in atoms)
    assert abs(total_net_charge - round(total_net_charge)) <= 0.01, total_net_charge

    return Molecule3D(
        [atom for (atom, _) in atoms],
        bond_objects,
        # formal_charges={atom.index: round(partial_charge) for (atom, partial_charge) in atoms},
        # bond_orders={bond: bond_order for (bond, bond_order) in bonds},
        name=molecule_name,
        # net_charge=round(total_net_charge),
    )


def pdb_to_Molecule3D(pdb_str: str,
                      mol_name: str = "",
                      net_charge: int = None,
                      assign_bond_orders_and_charges: bool = False) -> Molecule3D:
    """
    Create a 3D molecular entity from a PDB file.
    :param pdb_str:
    :param net_charge:
    :return:
    """

    # TODO: assert statements

    # get pdb atoms
    pdb_atoms = [pdb_atom for pdb_atom in pdb_atoms_in(pdb_str)]

    # convert to chem_ds atoms
    atoms = []
    n_id = 0
    for pdb_atom in pdb_atoms:

        atoms.append(
            Atom3D(
                index={'pdb': int(pdb_atom.index), 'nid': n_id},
                name=pdb_atom.name,
                element=pdb_atom.element,
                coordinates=pdb_atom.coordinates
            )
        )
        n_id += 1

    # get pdb bonds
    pdb_bonds = reduce(
        lambda acc, e: acc | e,
        [
            bonds_for_pdb_line(line)
            for line in pdb_str.splitlines()
            if is_pdb_connect_line(line)
        ],
        set(),
    )

    #print("PDB Bonds: ", pdb_bonds)

    # convert pdb_bonds to chem_ds bonds
    pdb_atom_index_name_map = {pdb_atom.index: pdb_atom.name for pdb_atom in pdb_atoms}
    bonds = []
    for pdb_bond in pdb_bonds:
        a1_ind, a2_ind = list(pdb_bond)
        bonds.append((pdb_atom_index_name_map[a1_ind], pdb_atom_index_name_map[a2_ind], Bond3D()))

    #print("Bonds: ", bonds)

    molecule = Molecule3D(
        atoms,
        bonds,
        name=mol_name
    )

    # assign bond orders and charges with ILP
    if assign_bond_orders_and_charges and net_charge is not None:
        molecule.assign_bond_orders_and_charges_with_ILP(net_charge)

    # if assign aromatic bonds
    # molecule.assign_aromatic_bonds() # TODO: implement this!!!

    return molecule


if __name__ == '__main__':
    with open('data/benxene.mol2.txt','r') as f:

        test = mol2_to_Molecule3D(f.read())

        import networkx as nx

        nx.set_node_attributes(test.graph,'test', 'test')
        nx.get_node_attributes(test.graph, 'test')