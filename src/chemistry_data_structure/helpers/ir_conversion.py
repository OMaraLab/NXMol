import os
import sys
import json
import random
from math import sqrt

from chemical_equivalence.calcChemEquivalency import Exception_Searching_Function
from chemistry_data_structure.helpers.chem import MASS
from chemistry_data_structure.parsing.hessian_analysis import (
    cal_stretching,
    cal_eigen_matrix,
)
from chemistry_data_structure.parsing.input_parsers import ATB_QMData_to_Molecule3D
from chemistry_data_structure.refactor.featurize import load_qm_data
from scipy.constants import pi, c


def load_fdb_data(fdb_id):
    fdb_fn = json.load(
    open(f"/home/yaofu/data/atb_fc/NXMol/src/fbd/{fdb_id}.json", "r"))
    return fdb_fn

def wavenumber_to_gromacs_fc(
    wavenumber,
    atom1,
    atom2,
    backward=False,
    gmx_fc=None,
):
    """
    Convert wavenumber from IR spectrum to GROMACS force constant (in kg/(mol*nm^
    2)).
    """
    # reduced mass in kg/mol
    reduced_mass = (MASS[atom1] * MASS[atom2] / (MASS[atom1] + MASS[atom2])) * 1e-3
    if not backward:
        """
        angular_frequency is in s^-1; since wavenumber is in cm^-1, we need to
        convert speed of light to cm/s
        """
        angular_frequency = wavenumber * 2 * pi * c * 100
        assert atom1 in MASS and atom2 in MASS, f"Unknown atom {atom1} or {atom2}"
        """
        conversion in the following refers to converting from kg/(mol*s^2) to 
        kg(mol*nm^2)
        """
        fc_pre_conversion = angular_frequency**2 * reduced_mass
        conv_factor = 1e-21

        return fc_pre_conversion * conv_factor

    else:
        assert atom1 in MASS and atom2 in MASS, f"Unknown atom {atom1} or {atom2}"
        assert gmx_fc is not None, "GROMACS force constant must be provided"
        conv_factor = 1e-21
        fc_pre_conversion = gmx_fc / conv_factor
        try:
            angular_frequency = sqrt(fc_pre_conversion / reduced_mass)
        except ValueError:
            return None
        wavenumber = angular_frequency / (2 * pi * c * 100)

        # return sqrt(fc_pre_conversion / reduced_mass) / (2 * pi * c * 100)
        return wavenumber


def get_distr_from_hessian_FDB(
    fdb_id, wave=False, exclude=None, len_exclude=None, bond_order_exclude=None, fdb_path = "/home/yaofu/data/atb_fc/NXMol/src/fbd/",):
    """
    Find the Hessian force constant (Gromacs units) for all bonds in a FDB match,
    or the equivalent wavenumbers.

    Returns a list of force constants or wavenumbers, with the last element being
    the reference bond types (tuple of atom types).
    """
    fdb_fn = json.load(
        open(f"{fdb_path}/{fdb_id}.json", "r")
    )
    bonds = {}
    for x in fdb_fn["atom_mappings"]:
        bonds[x] = [(i["1"], i["2"]) for i in fdb_fn["atom_mappings"][x]]

    distr = []
    ref = None
    for idx, (k, v) in enumerate(bonds.items()):
        try:
            qm_data = load_qm_data(k)
        except FileNotFoundError:
            continue
        # found = 0
        ref = (qm_data["type"][v[0][0]], qm_data["type"][v[0][1]])
        for i, j in v:

            if exclude:
                for x in exclude:
                    if x == (
                        qm_data["type"][i],
                        qm_data["type"][j],
                    ) or x == (
                        qm_data["type"][j],
                        qm_data["type"][i],
                    ):
                        # print("name exclude")
                        return None
            if len_exclude:
                if len(fdb_fn["atom_mappings"].keys()) < len_exclude:
                    # print("length exclude")
                    return None
            if bond_order_exclude:
                for x, y, bond_order in qm_data["bond_order"]:
                    if (i, j) == (x, y) or (i, j) == (y, x):
                        if bond_order < bond_order_exclude:
                            print("bond order exclude")
                            return None

            umatrix, eigmatrix = cal_eigen_matrix(
                qm_data["primary_axis_coords"], qm_data["hessian"]
            )
            fc = cal_stretching((i, j), umatrix, eigmatrix)
            if wave==True:
                wavenumber = wavenumber_to_gromacs_fc(
                    None, qm_data["type"][i], qm_data["type"][j], True, fc
                )
                distr.append(wavenumber)
            else:
                distr.append(fc)
        # printProgressBar(idx, len(bonds.items()))
    distr.append(ref)
    return distr

def get_kv_pair_from_FDB(fdb_id, path='/home/yaofu/data/atb_fc/NXMol/src/fbd/'):
    fdb_fn = json.load(
        open(f"{path}/{fdb_id}.json", "r")
    )
    bonds = {}
    for x in fdb_fn["atom_mappings"]:
        bonds[x] = [(i["1"], i["2"]) for i in fdb_fn["atom_mappings"][x]]
    return bonds

def get_molecules_in_FDB_fragment(fdb_id, path, charge_table=None, id_table=None):
    bonds = get_kv_pair_from_FDB(fdb_id, path=path)

    assert charge_table, "Charge table must be provided if not getting index only"
    for k, v in bonds.items():
        if id_table:
            if k not in id_table:
                continue
        try:
            qm_data = load_qm_data(k)
            mol = ATB_QMData_to_Molecule3D(
            qm_data, net_charge=int(charge_table[k]), name=k
        )
        except (FileNotFoundError, Exception):
            continue
        yield mol, v

def get_molIDs_in_FDB_fragment(*args):
    molIDs = []
    for x in args:
        fdb_fn = load_fdb_data(x)
        molIDs.extend(fdb_fn["atom_mappings"].keys())
    return molIDs
