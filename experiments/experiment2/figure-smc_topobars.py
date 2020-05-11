import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append("../../../toolbox")
from toolbox import plot as ph
import helpers
from pathlib import Path

# data_folder = Path("D:/datasets/paper_topology/03_gullfaks/")
# simulation = Path("smc_jac_paper_hires1")
# priors = helpers.load_priors(data_folder / simulation / "priors.json")
# thresholds = [t for t in os.listdir(data_folder / simulation)
#               if os.path.isdir(data_folder / simulation / t)]
#
#
# def count_unique_topologies(simulated_topologies):
#     unique_topologies = [simulated_topologies[0]]
#     unique_count = [1]
#
#     for topology in simulated_topologies[1:]:
#         skip = False
#         for idx, unique_topology in enumerate(unique_topologies):
#             if topology == unique_topology:
#                 unique_count[idx] += 1
#                 skip = True
#                 break
#         if not skip:
#             unique_topologies.append(topology)
#             unique_count.append(1)
#
#     return unique_topologies, unique_count
#
#
# simulated_topologies_all = {}
# for threshold in thresholds:
#     simulated_topologies = helpers.load_sample(data_folder / simulation / threshold / "edges.pkl")
#     ut, uc = count_unique_topologies(simulated_topologies)
#     simulated_topologies_all[threshold] = {
#         "all_topo": simulated_topologies,
#         "unique_topo": ut,
#         "unique_topo_count": uc
#     }
#
# vals = []
# for threshold, dict_ in simulated_topologies_all.items():
#     uc = dict_["unique_topo_count"]
#     # print(threshold, uc)
#     vals.append(len(uc))



vals = np.load("data/figure-smc_topobars_vals.npy")[::-1][:-1]
labels = np.load("data/figure-smc_topobars_thresholds.npy")[::-1][:-1]
ie_totals = np.load("data/smc_ietotals.npy")[::-1][:-1]

ie_total_fw = 0.223

font_size = 6
with plt.rc_context(ph.get_rcparams(font_size)):
    fig, axes = plt.subplots(nrows=2, sharex=True,
                             figsize=ph.get_figsize(
                                 0.5,
                                 ratio=0.62,
                                 textwidth=helpers.TEXTWIDTH,
                             )
    )
    barkwargs = dict(
        width=0.66,
    )

    ax = axes[0]
    for i, val in enumerate([676] + list(vals)):
        ax.text(i - 1, val + 20, str(val), horizontalalignment="center")
    vals[0] += 4
    vals[1] += 4
    ax.bar(labels, vals, color="black", **barkwargs)
    ax.set_ylabel("# Unique Topologies")

    ax.bar(-1, 676, color="grey", **barkwargs)

    ymax = 700
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.linspace(0, ymax, 3))
    ax.set_yticklabels([f"{int(i)}" for i in np.linspace(0, ymax, 3)])

    ax = axes[1]
    ax.set_ylim(0, 0.25)
    ax.set_yticks(np.linspace(0, 0.25, 3))
    ax.set_yticklabels([f"{i}" for i in np.linspace(0, 0.25, 3)])
    ax.set_xticks(np.arange(-1, len(labels)))
    ax.set_xticklabels(["FW"] + list(labels))
    ax.bar(labels, ie_totals, color="black", **barkwargs)
    ax.bar(-1, 0.233, color="grey", **barkwargs)
    ax.set_ylabel("Total Entropy")
    ax.set_xlabel("$\epsilon$")

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.5)




    plt.savefig("../../paper/figures/exp3_smctopobars.pdf", bbox_inches='tight',
                dpi=300)
    plt.show()

