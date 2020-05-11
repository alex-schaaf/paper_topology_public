import matplotlib.pyplot as plt
from matplotlib import colors
from toolbox import plot as ph
import numpy as np
import sys
sys.path.append("../../../gempy/")
import gempy as gp
import pickle

TEXTWIDTH = 503.61377
geo_model = gp.load_model_pickle("geo_model.p.pickle")
img = np.load("geomodel_slice.npy")
extent = [0, 2000, 0, 2000, 0, 2000]
COLORS = ("#EF5350", "#FFEE58", "#66BB6A", "#8D6E63", "#bdbdbd")
cmap = colors.ListedColormap(COLORS)
bounds = [2, 3, 4, 5, 6, 7]
norm = colors.BoundaryNorm(bounds, ncolors=6)
fontsize = 6
priors = pickle.load(open("data/abc/priors.p", "rb"))

with plt.rc_context(ph.get_rcparams(fontsize)):
    fig, ax = plt.subplots(
        figsize=ph.get_figsize(0.33, textwidth=TEXTWIDTH, ratio=1)
    )

    im = ax.imshow(
        img.T, origin="lower", extent=extent[:4], cmap=cmap, norm=norm
    )

    ax.plot(
        [400, 1070],
        [0, 2000],
        color="#4372BC"
    )

    for prior, color in zip(priors,
                            ["red", "yellow", "green", "#6d4c1a", "#4372bc",
                             "#4372bc"]):
        f1 = geo_model.surface_points.df["surface"] == prior.name[:-2]
        f2 = geo_model.surface_points.df["Y"] == 1000.
        f = f1 & f2
        X = geo_model.surface_points.df[f].X.values
        Z = geo_model.surface_points.df[f].Z.values

        for x, z in zip(X, Z):
            pos_bool = 1250 < x < 1500 or 650 < x < 750
            if prior.column == "Z":
                if x == 0:
                    x = + 70
                    # y -= 20
                ax.plot(x, z, "o", markersize=3, color=color, markeredgewidth=0.5,
                         markeredgecolor="black")
                if pos_bool:
                    (_, caps, _) = ax.errorbar(x, z, yerr=prior.std(),
                                                color="black", linewidth=0.5, alpha=1, capsize=2)
                    for cap in caps:
                        cap.set_markeredgewidth(1)
                        cap.linewidth = 1
            elif prior.column == "X":
                if pos_bool:
                    (_, caps, _) = ax.errorbar(x, z, xerr=prior.std(),
                                                color="black", linewidth=0.5, alpha=1, capsize=2)
                    for cap in caps:
                        cap.set_markeredgewidth(1)
                        cap.linewidth = 1

    ax.set_xticks([0, 1000, 2000])
    ax.set_yticks([0, 1000, 2000])
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("(b) Geomodel Parametrization", fontsize=fontsize + 2)



    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)

    plt.savefig(
        "../../paper/figures/exp1_parametrization_.pdf",
        bbox_inches='tight',
        dpi=300
    )
    plt.show()
