import matplotlib.pyplot as plt
from matplotlib import colors
from toolbox import plot as ph
import numpy as np
import sys
sys.path.append("../../../gempy/")
import gempy as gp

TEXTWIDTH = 503.61377
geo_model = gp.load_model_pickle("geo_model.p.pickle")
edges = {(1, 2), (1, 6), (1, 7), (2, 3), (2, 7), (2, 8), (3, 4), (3, 8), (3, 9),
         (4, 5), (4, 9), (4, 10), (5, 10), (6, 7), (7, 8), (8, 9), (9, 10)}

ctrs = {1: np.array([12.16046912  + 2, 24.8310141, 40.65527994 + 2]),
        2: np.array([8.93961983   + 2, 24.92955647, 28.73667536 + 2]),
        3: np.array([7.95101873   + 2, 24.71784318, 23.43445153 + 2]),
        4: np.array([6.85111073   + 2, 24.53773105, 17.19230117 + 2]),
        5: np.array([4.92739115   + 2, 23.20962489, 7.25340731  + 2]),
        6: np.array([35.9145365   + 2, 25.00458115, 39.16981059 + 2]),
        7: np.array([33.7571116   + 2, 24.83114515, 26.98018478 + 2]),
        8: np.array([32.89434051  + 2, 24.74326391, 21.65431227+ 2]),
        9: np.array([32.08333333  + 2, 24.65390105, 15.5167301 + 2]),
        10: np.array([31.04821962 + 2, 23.66595199, 5.89393366+ 2])}

img = np.load("geomodel_slice.npy")
extent = [0, 2000, 0, 2000, 0, 2000]

print(np.unique(img))

COLORS = ("#EF5350", "#FFEE58", "#66BB6A", "#8D6E63", "#bdbdbd")
cmap = colors.ListedColormap(COLORS)
bounds = [2, 3, 4, 5, 6, 7]
norm = colors.BoundaryNorm(bounds, ncolors=6)


fontsize = 6
with plt.rc_context(ph.get_rcparams(fontsize)):
    fig, ax = plt.subplots(
        figsize=ph.get_figsize(0.33, textwidth=TEXTWIDTH, ratio=1)
    )

    im = ax.imshow(
        img.T, origin="lower", extent=extent[:4], cmap=cmap, norm=norm
    )

    ax.set_xticks([0, 1000, 2000])
    ax.set_yticks([0, 1000, 2000])
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("(c) Geomodel Topology", fontsize=fontsize + 2)

    gp.plot.plot_topology(
        geo_model,
        edges,
        ctrs,
        scale=True,
        label_kwargs=dict(fontsize=fontsize),
        edge_kwargs=dict(linewidth=0.5)
    )

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)

    plt.savefig(
        "../../paper/figures/exp1_init_topo_.pdf",
        bbox_inches='tight',
        dpi=300
    )
    plt.show()
