import pickle
import os
import csv
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

    dirs = os.listdir("test_dataset_big")
    nbonds_mol = 0
    for mol in dirs:
        qm_data = load_qm_data(mol)
        for b in qm_data["bond_order"]:
            if b[-1] > COVALENT_BOND_ORDER_THRESHOLD:
                nbonds_mol += 1

    arr = []
    with open('./netcharges.csv', newline='') as csvfile:
        data = csv.reader(csvfile, delimiter='\t')
        
        for idx, row in enumerate(data):
            arr.append((row[0], row[1]))

    X = []
    Y = np.zeros((nbonds_mol, 1))
    net_charge = 0
    idx = 0
    for mol in dirs:
        for x, y in arr:
            if mol == x:
                net_charge = y
        print(mol, net_charge)
        qm_data = load_qm_data(mol)
        mol3D = ATB_QMData_to_Molecule3D(qm_data, net_charge=int(net_charge), name=mol)
        for i, j in mol3D.bonds.keys():
            Y[idx, 0] = mol3D.bonds[i, j].get("force_constant")

            if random.random() < 0.5:
                X.append(
                    [
                        mol3D.atoms[i].element,
                        mol3D.atoms[i].atomic_number,
                        mol3D.atoms[i].radius,
                        mol3D.atoms[i].mass,
                        mol3D.atoms[i].electronegativity,
                        mol3D.calcNumBonds(i),
                        mol3D.atoms[j].element,
                        mol3D.atoms[j].atomic_number,
                        mol3D.atoms[j].radius,
                        mol3D.atoms[j].mass,
                        mol3D.atoms[j].electronegativity,
                        mol3D.calcNumBonds(j),
                        mol3D.bonds[i, j].get("bond_length"),
                        mol3D.bonds[i, j].get("fract_bond_order"),
                        mol3D.BFS_edge(i, j, 1),
                    ]
                )
            else:
                X.append(
                    [
                        mol3D.atoms[j].element,
                        mol3D.atoms[j].atomic_number,
                        mol3D.atoms[j].radius,
                        mol3D.atoms[j].mass,
                        mol3D.atoms[j].electronegativity,
                        mol3D.calcNumBonds(j),
                        mol3D.atoms[i].element,
                        mol3D.atoms[i].atomic_number,
                        mol3D.atoms[i].radius,
                        mol3D.atoms[i].mass,
                        mol3D.atoms[i].electronegativity,
                        mol3D.calcNumBonds(i),
                        mol3D.bonds[i, j].get("bond_length"),
                        mol3D.bonds[i, j].get("fract_bond_order"),
                        mol3D.BFS_edge(j, i, 1),
                    ]
                )
            idx += 1

    with open("X_big.pickle", "wb") as handle:
        pickle.dump(X, handle)

    with open("Y_big.pickle", "wb") as handle:
        pickle.dump(Y, handle)
