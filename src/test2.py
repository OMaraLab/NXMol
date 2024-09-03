import pickle
from pprint import pprint
from pathlib import Path
import matplotlib.pyplot as plt

from chemistry_data_structure.helpers.graphs import calc_equal_bonds, cull_equal_bonds
from chemistry_data_structure.objects.molecular_entity import NXMolWeaveFeaturizer
from chemistry_data_structure.parsing.input_parsers import ATB_QMData_to_Molecule3D
# from chemical_equivalence.calcChemEquivalency import getChemEquivGroups
from atb_outputs.mol_data import MolData


def load_qm_data(molid: str):
    with open(f"hessian_data/{molid}/b3lyp_631Gd_PCM_water_hessian.pickle", "rb") as fh:
        return pickle.load(fh)


def single_test():
    molid = "1009"
    example_qm_data = load_qm_data(molid)
    mol3D = ATB_QMData_to_Molecule3D(example_qm_data, net_charge=0, name=molid)
    mol3D.draw_graph()
    featurizer = NXMolWeaveFeaturizer()
    weave_mol = featurizer._featurize(mol3D)
    print(weave_mol)


def load_all_qm_data():
    mols = {}
    for qm_data in Path("test_dataset").glob("*/b3lyp_631Gd_PCM_water_hessian.pickle"):
        with open(qm_data, "rb") as fh:
            mols[qm_data.parent.name] = pickle.load(fh)
    return mols


def bond_order_hist():
    mols = load_all_qm_data()
    bond_orders = []
    for molid, mol in mols.items():
        bond_orders.extend([bond_data[-1] for bond_data in mol["bond_order"]])
    print(bond_orders)
    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.hist(bond_orders, 50, density=True)

    ax.set_xlabel("Bond order")
    ax.set_ylabel("Probability density")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    qm = load_qm_data("21")
    mol3D = ATB_QMData_to_Molecule3D(qm)
    print(mol3D.bonds)
    calc_equal_bonds(mol3D)
    print(mol3D.eq_grps_sorted)
    cull_equal_bonds(mol3D)
    for b in mol3D.bonds:
        print(b, mol3D.bonds[b].get("force_constant"))
