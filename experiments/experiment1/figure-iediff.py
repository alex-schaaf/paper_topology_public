import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append("../../../toolbox/")
from toolbox import plot as ph

# load data
fp = Path("./data/")
ie_abc = np.load(fp / "ie_abc.npy")
ie_fw = np.load(fp / "ie_fw.npy")
ie_diff = ie_abc - ie_fw
shp = (50, 50, 50)
extent = [0, 2000, 0, 2000, 0, 2000]

fontsize = 8
with plt.rc_context(ph.get_rcparams(fontsize)):
    fig, ax = plt.subplots(
        figsize=ph.get_figsize(0.5, textwidth=503.61377, ratio=1)
    )
    plt.tick_params(
        left=False,
        bottom=False,
    )

    im = ax.imshow(
        ie_diff.reshape(*shp)[:, 24, :].T,
        origin="lower",
        extent=extent[:4],
        interpolation="bicubic",
        cmap="RdBu_r", vmin=-0.8, vmax=0.8
    )
    cbar = ph.colorbar(im)
    cbar.outline.set_linewidth(0.5)
    cbar.set_ticks([-0.8, 0, 0.8])
    cbar.ax.tick_params(size=0)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)

    ax.set_title("Information Entropy Difference\n (XZ-Section)")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    locs = np.linspace(0, 2000, 3).astype(int)
    ax.set_yticks(locs)
    ax.set_yticklabels(locs)
    ax.set_xticks(locs)
    ax.set_xticklabels(locs)

    plt.tight_layout()

    plt.savefig(
        "../../paper/figures/exp1_ie_diff.pdf", dpi=300
    )
    plt.show()
