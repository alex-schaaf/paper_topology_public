import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../../toolbox")
from toolbox import plot as ph
import helpers

ies = np.load("data/smc_ieblocks.npy")
ie_block_forward = np.load("data/ie_block_forward.npy")
ie_total_forward = np.load("data/ie_total_forward.npy")
diff = ies[0] - ie_block_forward

extent = [451000.0, 456000.0, 6782000.0, 6784000.0, -6927.0, -4953.0]

fontsize = 6
with plt.rc_context(ph.get_rcparams(fontsize)):
    fig, ax = plt.subplots(
        figsize=ph.get_figsize(0.6, textwidth=helpers.TEXTWIDTH, ratio=0.45)
    )

    img = diff.reshape(60, 50, 60)[:, 12, :]
    im = ax.imshow(
        img.T, origin="lower", extent=extent[:2] + extent[4:],
        cmap="RdBu_r", interpolation="bicubic", vmin=-1.5, vmax=1.5
    )
    cbar = ph.colorbar(im)
    cbar.outline.set_linewidth(0.5)
    cbar.set_ticks([-1.5, 0, 1.5])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)

    ax.set_title("Information Entropy Difference (XZ-Section)", fontsize=fontsize+1)
    ax.set_xlabel("X")
    ax.set_ylabel("Depth [m]")

    ax.set_yticks(np.linspace(extent[4], extent[5], 3))
    ax.set_yticklabels(
        np.linspace(extent[4] / 3, extent[5] / 3, 3).astype(int))

    xmid = np.mean(extent[:2])
    diff = (xmid - extent[0]) / 1.25
    ax.set_xticks([xmid - diff, xmid, xmid + diff])
    ax.set_xlabel("X")

    plt.tight_layout()

    plt.savefig("../../paper/figures/exp2_iediff.pdf",
                dpi=300)

    plt.show()
