from collections import defaultdict
import re
import pickle
import matplotlib.pyplot as plt

def check_everything(list_of_nei):
    fragments = pickle.load(open("fragments_shell_size_1.pickle", "rb"))
    a = pickle.load(open("graph_agg_edatas_shell_size_1.pickle", "rb"))
    b = pickle.load(open("mean_fc_shell_size_1.pickle", "rb"))
    for pair in fragments:
        for nei in list_of_nei:
            nei_fc = b[pair][nei]
            for bond in fragments[pair][nei]:
                a1, a2, id = re.findall(r"\d+", bond)
                if (a2, a1) in a[id]:
                    assert (
                        a[id][a2, a1] == nei_fc
                    ), f"{a[id][a2, a1]}, {id}, {a1}, {a2}, {nei_fc}"
                else:
                    assert (
                        a[id][a1, a2] == nei_fc
                    ), f"{a[id][a1, a2]}, {id}, {a2}, {a1}, {nei_fc}"


def plot_longest_frags(len_plot):
    plot_lists = defaultdict(list)
    edatas = pickle.load(open("graph_agg_edatas_shell_size_1.pickle", "rb"))
    fragments = pickle.load(open("fragments_shell_size_1.pickle", "rb"))
    len_parsed_inner = 0
    len_parsed_outer = 0
    for pair in fragments:
        dict_by_len = {
            k: v
            for k, v in sorted(
                fragments[pair].items(), key=lambda x: len(x[1]), reverse=True
            )
        }
        for nei in dict_by_len:
            for bond in fragments[pair][nei]:
                a1, a2, id = re.findall(r"\d+", bond)
                if (a2, a1) in edatas[id]:
                    plot_lists[nei].append(edatas[id][a2, a1])
                else:
                    plot_lists[nei].append(edatas[id][a1, a2])
            len_parsed_inner += 1
            print(pair, nei)
            if len_parsed_inner == len_plot:
                len_parsed_inner = 0
                break

        len_parsed_outer += 1
        if len_parsed_outer == len_plot:
            break
    print(len(plot_lists))
    assert len(plot_lists) == len_plot**2
    for k, v in plot_lists.items():
        plt.hist(v, bins=100)
        plt.title(f"{k}")
        plt.show()


if __name__ == "__main__":
    plot_longest_frags(3)
