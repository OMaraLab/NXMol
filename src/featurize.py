import pickle
import os
import re
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

import pickle

from chemistry_data_structure.objects.molecular_entity import NXMolWeaveFeaturizer
from chemistry_data_structure.parsing.input_parsers import ATB_QMData_to_Molecule3D
from chemistry_data_structure.parsing.test import single_test, bond_order_hist


def load_qm_data(molid: str):
    for dirpath, dirname, filename in os.walk("test_dataset_big"):
        if dirpath == f"test_dataset_big/{molid}":
            with open(f"{dirpath}/{filename[0]}", "rb") as fh:
                return pickle.load(fh)

def load_qm_data_small(molid: str):
    with open(f"test_dataset_small/{molid}/b3lyp_631Gd_PCM_water_hessian.pickle", "rb") as fh:
        return pickle.load(fh)

if __name__ == "__main__":
    COVALENT_BOND_ORDER_THRESHOLD = 0.5

    included_dirs = []

    for mol in os.listdir("test_dataset_small"):
        if random.random():
            included_dirs.append(mol)

    nbonds_mol = 0
    for mol in included_dirs:
        qm_data = load_qm_data_small(mol)
        for b in qm_data["bond_order"]:
            if b[-1] > COVALENT_BOND_ORDER_THRESHOLD:
                nbonds_mol += 1

    X = []
    Y = np.zeros((nbonds_mol, 1))
    idx = 0
    for mol in included_dirs:
        qm_data = load_qm_data(mol)
        mol3D = ATB_QMData_to_Molecule3D(qm_data, net_charge=0, name=mol)
        # featurizer = NXMolWeaveFeaturizer()
        # print(mol)
        # weave_mol = featurizer._featurize(mol3D)
        for i, j in mol3D.bonds.keys():
            Y[idx, 0] = mol3D.bonds[i, j].get("force_constant")

            if random.random() < 0.5:
                X.append(
                    [
                        mol3D.atoms[i].element,
                        mol3D.atoms[j].element,
                        mol3D.bonds[i, j].get("bond_length"),
                        mol3D.bonds[i, j].get("fract_bond_order"),
                        mol3D.BFS_edge(i, j, 1),
                    ]
                )
            else:
                X.append(
                    [
                        mol3D.atoms[j].element,
                        mol3D.atoms[i].element,
                        mol3D.bonds[i, j].get("bond_length"),
                        mol3D.bonds[i, j].get("fract_bond_order"),
                        mol3D.BFS_edge(j, i, 1),
                    ]
                )
            idx += 1

    with open("X.pickle", "wb") as handle:
        pickle.dump(X, handle)

    with open("Y.pickle", "wb") as handle:
        pickle.dump(Y, handle)
    # training_set = one_hot_encode_column(training_set, 0, "i")
    # training_set = one_hot_encode_column(training_set, 1, "j")
    # # training_set = one_hot_encode_column(training_set, 2)
    # training_set.columns = training_set.columns.map(str)
    #
    #
    # print(training_set.columns)
    #
    # reg = LinearRegression().fit(training_set, Y)
    # print(reg.score(training_set, Y))
    # # single_test()
