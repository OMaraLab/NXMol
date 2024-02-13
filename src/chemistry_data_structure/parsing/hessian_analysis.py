from numpy import reshape, zeros, array, dot

from numpy.linalg import linalg


def cal_stretching(bond_atom_ids, umatrix, eigmatrix):
    """cal bond stretching force constant, i and j are atomid"""

    A, B = bond_atom_ids

    assert linalg.norm(umatrix[A][B]) - 1 < 0.0000001
    u_AB = umatrix[A][B]

    # harmonic force constant
    hfc = 0.0
    for i in range(3):
        eig_AB = eigmatrix[A][B]["values"][i]
        eig_AB_vect = eigmatrix[A][B]["vectors"][i]
        k = eig_AB * abs(dot(u_AB, eig_AB_vect))
        hfc += k

    # take norm of complex number and convert to float
    hfc = float(abs(hfc))

    return hfc


def cal_eigen_matrix(coords, hessian):
    # matrix of eigenvalues and eigenvectors for i-j pair
    eigmatrix = {}
    for i, columns in list(sorted(hessian.items())):
        eigmatrix[i] = {}
        for j, pair_matrix in list(sorted(columns.items())):
            if j > i:
                continue
            eig_values, eig_vectors = linalg.eig(reshape(pair_matrix, (3, 3)))
            eigmatrix[i][j] = {"values": eig_values, "vectors": eig_vectors}
            eigmatrix[j][i] = eigmatrix[i][j]
    # the unit vector along i-j direction
    umatrix = {}
    for i, columns in list(sorted(eigmatrix.items())):
        umatrix[i] = {}
        for j in list(columns.keys()):
            if i == j:
                umatrix[i][j] = zeros(3)
                continue
            vector = array(coords[j]) - array(coords[i])
            nvector = vector / linalg.norm(vector)
            umatrix[i][j] = nvector

    return umatrix, eigmatrix
