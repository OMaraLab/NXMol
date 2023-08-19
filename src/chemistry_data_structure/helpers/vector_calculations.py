"""
Using GROMOS method for virtual hydrogen atom placement.
Source: Biomolecular Simulation: The GROMOS96 Manual and User Guide chII, p67-72
"""

from typing import List
from math import sqrt, cos, sin, pi, acos
import numpy as np
from numpy import array as vector, cross, dot
from numpy.linalg import norm
from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, LpStatus, LpContinuous
from pulp import PULP_CBC_CMD, PulpSolverError

C_H_BOND_LENGTH = 1.0
TETRAHEDRAL_BOND_ANGLE = 109.5 * (pi / 180)
TRIGONAL_PLANAR_BOND_ANGLE = 120 * (pi / 180)

""" Used to calculate position of the first hydrogen when more than one are to
    be placed, and only one vector is known (i.e. C-C)

    Returns new coordinates for a hydrogen atom.
"""
def place_first_hydrogen(points: list, theta: float) -> np.ndarray:

    # define base vector (e.g. C-C bond)
    base_vector = points[0] - points[1]

    # define unit vector in x-direction
    u = base_vector / norm(base_vector)

    # create arbitrary vector
    arb_vector = vector([0, 0, 1])

    # create unit vector in y-direction
    v = cross(u, arb_vector)
    v = v / norm(v)

    # calculate x, y of right angled triangle
    x = C_H_BOND_LENGTH * cos(pi - theta)
    y = C_H_BOND_LENGTH * sin(pi - theta)

    new_coordinates = points[0] + (x * u) + (y * v)

    return new_coordinates

def get_angle_between_vectors(v1: vector, v2: vector):
    return acos(dot(v1, v2) / (norm(v1) * norm(v2)))

def gromos_trigonal_planar_1H(points: list):

    # define point vectors
    ri = points[0]
    rj = points[1]
    rk = points[2]

    # define sum vector and norm of sum
    s = 2 * ri - rj - rk
    s_norm = norm(s)

    # define hydrogen bond distance
    d = C_H_BOND_LENGTH

    # calculate resulting hydrogen vector
    rn = ri + d * (s / s_norm)
    new_coordinates = tuple(rn.tolist())

    return new_coordinates

def gromos_trigonal_planar_2H(points: list) -> list:

    theta = TRIGONAL_PLANAR_BOND_ANGLE
    new_coordinates = []

    # calculate first hydrogen vector and add to list of points/new coordinates
    rn = place_first_hydrogen(points, theta)
    points.append(rn)
    new_coordinates.append(tuple(rn.tolist()))

    # calculate other hydrogen position using 1H method
    new_coordinates.append(gromos_trigonal_planar_1H(points))

    return new_coordinates

def gromos_tetrahedral_1H(points: list):

    # define point vectors
    ri = points[0]
    rj = points[1]
    rk = points[2]
    rl = points[3]

    # SPECIAL CASE FOR PRIOR TRIGONAL-PLANAR BOND ANGLE
    v1 = ri - rj
    v2 = ri - rk
    # 2 radians = approx. 115 degrees
    angle = get_angle_between_vectors(v1, v2)
    if get_angle_between_vectors(v1, v2) > 2:
        new_coordinates = ri + cross(v1, v2)
        return tuple(new_coordinates.tolist())

    # define sum vector and norm of sum
    s = 3 * ri - rj - rk - rl
    s_norm = norm(s)

    # define hydrogen bond distance
    d = C_H_BOND_LENGTH

    # calculate resulting hydrogen vector
    rn = ri + d * (s / s_norm)
    new_coordinates = tuple(rn.tolist())

    return new_coordinates


def gromos_tetrahedral_from_2_vectors(points: list, h_num: int) -> list:

    theta = TETRAHEDRAL_BOND_ANGLE
    new_coordinates = []

    while h_num > 0:

        # define point vectors differently for each hydrogen
        ri = points[0]
        if h_num == 1:
            rj = points[1]
            rk = points[2]
        else:
            rk = points[1]
            rj = points[2]

        # define sum vector and norm of sum
        s = 2 * ri - rj - rk
        s_norm = norm(s)

        # define q vector and norm of q
        q = cross((ri - rj), (ri - rk))
        q_norm = norm(q)

        # define hydrogen bond distance
        d = C_H_BOND_LENGTH

        # define alpha and beta parameters
        alpha = d * cos(theta / 2)
        beta = d * sin(theta / 2)

        # calculate resulting hydrogen vector
        rn = ri + alpha * (s / s_norm) + beta * (q / q_norm)
        new_coordinates.append(tuple(rn.tolist()))

        h_num -= 1

    return new_coordinates

def gromos_tetrahedral_3H(points: list) -> list:

    theta = TETRAHEDRAL_BOND_ANGLE

    new_coordinates = []

    # calculate first hydrogen vector and add to list of points/new coordinates
    rn = place_first_hydrogen(points, theta)
    points.append(rn)
    new_coordinates.append(tuple(rn.tolist()))

    # calculate other two hydrogen positions using 2H method
    new_coordinates.extend(gromos_tetrahedral_from_2_vectors(points, 2))

    return new_coordinates


def place_h_using_ilp(points: List[np.ndarray]):
    """
    Uses an ILP solver to places one new hydrogen atom in the position that maximises
    the dot product between it's coordinates and all existing vector coordinates. This is a more
    general placement strategy useful for centers with hybridisations above a tetrahedral geometry
    (where the number of cases to consider increases drastically).

    Use this repeatedly to add hydrogens one at a time for atomic centers that are above tetrahedral.

    :param points: all points involved in the placement of the new hydrogen, including the central atom.
    :return:

    """

    # transform points to vector
    print('Points: ', points)
    if len(points) == 1:
        return tuple(points[0] + np.array([1.0, 1.0, 1.0]))

    problem = LpProblem("Hydrogen vector placement problem", LpMaximize)
    center_point, vector_points = points[0], points[1:]
    unit_vectors = [(v - center_point) / norm(v - center_point) for v in vector_points]

    print("Unit Vectors: ", unit_vectors)

    # ===== SETS =====
    V = range(len(unit_vectors))      # existing vectors to consider in the placement

    # ===== DATA =====
    PX = [float(unit_vectors[i][0]) for i in V]      # x coords of existing vectors
    PY = [float(unit_vectors[i][1]) for i in V]      # y coords of existing vectors
    PZ = [float(unit_vectors[i][2]) for i in V]      # z coords of existing vectors

    print('PX: ', PX)
    print('PY: ', PY)
    print('PZ: ', PZ)

    # ===== VARIABLES =====
    d = [LpVariable(f"d_{i}", lowBound=-1.0, upBound=1.0, cat=LpContinuous) for i in V]         # distance variable
    x = LpVariable("x", lowBound=-1.0, upBound=1.0, cat=LpContinuous)                           # coordinate variables for new vector
    y = LpVariable("y", lowBound=-1.0, upBound=1.0, cat=LpContinuous)
    z = LpVariable("z", lowBound=-1.0, upBound=1.0, cat=LpContinuous)

    # ===== OBJECTIVE =====

    # solve a min max problem, maximize the minimum of the distance between the new vector and each other vector
    problem.setObjective(sum(d[i] for i in V))

    # TODO: we actually want to maximize the ABSOLUTE DISTANCE VALUE... may need absolute binding variable...

    # TODO: why don't we have a d distance parameter that is literal euclidean distance between the points,
    #   we find the point that is furthest away from all other points, and then constrain the point to exist on
    #   the unit sphere from the central point??? (INSTEAD OF USING DOT PRODUCT)

    # ===== CONSTRAINTS =====

    # 1) bound each distance variable by the dot product between the new vector and the other vector
    for i in V:
        problem += d[i] <= x * PX[i] + y * PY[i] + z * PZ[i]

    # ===== SOLVING =====
    problem.solve()

    # get solution unit vector
    if problem.status == 1:
        for var in problem.variables():
            print(f'{var.name}: {var.varValue}')

        # check for possible null var values
        x_val = x.varValue if x.varValue is not None else 0.0     # TODO: check that 0.0 is an appropriate value to use for this
        y_val = y.varValue if y.varValue is not None else 0.0
        z_val = z.varValue if z.varValue is not None else 0.0
        new_h_vector = np.array([x_val, y_val, z_val])
    else:
        print("Solving failed...")
        raise PulpSolverError

    print('New H Vector: ', new_h_vector)

    # transform the resulting unit vector into relative coordinates with the hydrogen carbon bond length
    new_coordinates = tuple(center_point + new_h_vector * C_H_BOND_LENGTH)

    print('New H Coordinates: ', new_coordinates)

    return new_coordinates