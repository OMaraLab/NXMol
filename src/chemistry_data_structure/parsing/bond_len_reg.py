import cmath

# Generates distance between two covalently bonded atoms.
def cal_bond_length(qm_data, bond_atom_ids):
    A, B = bond_atom_ids
    qm_coords = qm_data["primary_axis_coords"]
    diff_vec = [qm_coords[A][i] - qm_coords[B][i] for i in range(len(qm_coords[A]))]
    return cmath.sqrt(sum([x**2 for x in diff_vec])).real
