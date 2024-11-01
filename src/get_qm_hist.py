import os
import pickle
import matplotlib.pyplot as plt

from chemistry_data_structure.helpers.ir_conversion import (
    get_wavenumber_from_hessian_FDB,
)

if __name__ == "__main__":
    # a = []
    # for _, _, x in os.walk("/home/yaofu/data/atb_fc/NXMol/src/fbd"):
    #     for i in x:
    #         if i.endswith(".json"):
    #             g = get_wavenumber_from_hessian_FDB(
    #                 i[:-5], (("C", "H"), ("O", "H")), 50, 1.3
    #             )
    #             if g is not [None]:
    #                 a.append((i[:-5], g))
    #             else:
    #                 print("skipped")
    #
    # pickle.dump(a, open("wavenumbers_fdb_cc.pickle", "wb"))

    a = pickle.load(open('wavenumbers_fdb_cc.pickle', 'rb'))

    b = {}
    for x in a:
        if x[1][-1] not in b.keys():
            b[x[1][-1]] = [(x[0], x[1][:-1])]
        else:
            b[x[1][-1]].append((x[0], x[1][:-1]))
    pickle.dump(b, open("wavenumbers_btype_cc.pickle", "wb"))

    # del b[None]

    for k, v in b.items():
        if 'C' in k and 'H' in k:
            plt.axvline(x=3000, color="r", linestyle="--")
            plt.axvline(x=3100, color="r", linestyle="--")
        elif 'O' in k and 'H' in k:
            plt.axvline(x=3200, color="r", linestyle="--")
            plt.axvline(x=2700, color="r", linestyle="--")
        elif 'C' in k and 'N' in k:
            plt.axvline(x=1250, color="r", linestyle="--")
            plt.axvline(x=1020, color="r", linestyle="--")
        elif 'C' in k and 'O' in k:
            plt.axvline(x=1320, color="r", linestyle="--")
            plt.axvline(x=1000, color="r", linestyle="--")
        for id, n in v:
            print(id, len(n))
            plt.hist(n, bins=100, label=id)
        plt.legend(loc="best")
        plt.title(f"{k}")
        plt.xlabel("Wavenumber ($cm^{-1}$)")
        plt.ylabel("Count")
        plt.savefig(f"{k}.png")
        plt.show()
