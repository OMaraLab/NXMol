import pickle
from pprint import pprint

a = pickle.load(open("fdb_e_label.pickle", "rb"))
b = pickle.load(open("fdb_e_label_star.pickle", "rb"))
# b = pickle.load(open("graph_mean_edatas.pickle", "rb"))

print(a)
print(b)

