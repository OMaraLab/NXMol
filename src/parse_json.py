import os
import pickle
from featurize import printProgressBar, load_qm_data, ATB_QMData_to_Molecule3D
from chemistry_data_structure.helpers.graphs import return_fdb_bond_ids
from featurize import load_qm_data, ATB_QMData_to_Molecule3D


with open("X_pre_fdb_merge.pickle", "rb") as handle:
    X_list = pickle.load(handle)
with open("Y_pre_fdb_merge.pickle", "rb") as handle:
    Y_list = pickle.load(handle)

# for id, v in enumerate(X_list):
#     if v[0] == "22498" and v[1] == 18 and v[8] == 34 \
#      or v[0] == "22498" and v[1] == 34 and v[8] == 18:
#         print(Y_list[id])
#     elif v[0] == "22331" and v[1] == 26 and v[8] == 31 \
#        or v[0] == "22331" and v[1] == 31 and v[8] == 26:
#         print(Y_list[id])
#

ids_to_remove = []
for idx, f in enumerate(os.listdir("fdb")):
    if f.endswith(".json"):
        try:
            ids_to_remove.append(return_fdb_bond_ids(f, X_list, Y_list))
        except TypeError:
            pass
        printProgressBar(idx, len(os.listdir("fdb")))
with open("ids_to_remove.pickle", "wb") as handle:
    pickle.dump(ids_to_remove, handle)
assert len(ids_to_remove) == len(set(ids_to_remove))
#
with open("X_post_fdb_merge.pickle", "wb") as handle:
    pickle.dump(X_list, handle)
with open("Y_post_fdb_merge.pickle", "wb") as handle:
    pickle.dump(Y_list, handle)
