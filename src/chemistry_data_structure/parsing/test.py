import pickle

from chemistry_data_structure.objects.molecular_entity import NXMolWeaveFeaturizer
from input_parsers import ATB_QMData_to_Molecule3D


def load_qm_data(molid: str):
    with open(f"test_dataset/{molid}/b3lyp_631Gd_PCM_water_hessian.pickle", "rb") as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    molid = "8"
    example_qm_data = load_qm_data(molid)
    mol3D = ATB_QMData_to_Molecule3D(example_qm_data, net_charge=0, name=molid)
    mol3D.draw_graph()
    featurizer = NXMolWeaveFeaturizer()
    weave_mol = featurizer._featurize(mol3D)
    print(weave_mol)