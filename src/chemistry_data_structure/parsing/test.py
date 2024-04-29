import pickle
from pathlib import Path
import matplotlib.pyplot as plt

from chemistry_data_structure.objects.molecular_entity import NXMolWeaveFeaturizer
from chemistry_data_structure.parsing.input_parsers import ATB_QMData_to_Molecule3D



def load_qm_data(molid: str):
    with open(f"test_dataset/{molid}/b3lyp_631Gd_PCM_water_hessian.pickle", "rb") as fh:
        return pickle.load(fh)


def single_test():
    molid = "8"
    example_qm_data = load_qm_data(molid)
    mol3D = ATB_QMData_to_Molecule3D(example_qm_data, net_charge=0, name=molid)
    print(mol3D.atoms["1"].atomic_number)
    # mol3D.draw_graph()
    # featurizer = NXMolWeaveFeaturizer()
    # weave_mol = featurizer._featurize(mol3D)
    # print(weave_mol)


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

    ax.set_xlabel('Bond order')
    ax.set_ylabel('Probability density')
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    single_test()
    # bond_order_hist()
