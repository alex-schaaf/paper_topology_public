import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../../toolbox")
from toolbox import plot as ph
import helpers
from mpl_toolkits.axes_grid1 import AxesGrid

ies = np.load("data/smc_ieblocks.npy")
ie_block_forward = np.load("data/ie_block_forward.npy")
ie_total_forward = np.load("data/ie_total_forward.npy")

extent = [451000.0, 456000.0, 6782000.0, 6784000.0, -6927.0, -4953.0]
vmax = 2#np.max([ie_block_forward, ies[0]])

fontsize = 6
with plt.rc_context(ph.get_rcparams(fontsize)):
    fig = plt.figure(
        figsize=ph.get_figsize(1, textwidth=helpers.TEXTWIDTH, ratio=0.3))

    grid = AxesGrid(
        fig, 111,
        nrows_ncols=(1, 2),
        axes_pad=0.1,
        cbar_mode='single',
        cbar_location='right',
        cbar_pad=0.1
    )
    for ax, ie in zip(grid, (ie_block_forward, ies[0])):
        img = ie.reshape(60, 50, 60)[:, 12, :]
        im = ax.imshow(
            img.T, origin="lower", extent=extent[:2]+extent[4:],
            cmap="viridis", vmin=0, vmax=vmax, interpolation="bicubic"
        )
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.5)

        ax.set_yticks(np.linspace(extent[4], extent[5], 3))
        ax.set_yticklabels(
            np.linspace(extent[4] / 3, extent[5] / 3, 3).astype(int))
        xmid = np.mean(extent[:2])
        diff =(xmid - extent[0]) / 1.25
        ax.set_xticks([xmid - diff, xmid, xmid + diff])
        ax.set_xlabel("X")

    grid[0].set_title("Forward Simulation", fontsize=fontsize + 1)
    grid[1].set_title("ABC-REJ/SMC $\epsilon = 0.025$", fontsize=fontsize + 1)

    cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.ax.set_yticks(np.linspace(0, vmax, 3))
    for axis in ['top', 'bottom', 'left', 'right']:
        cbar.ax.spines[axis].set_linewidth(0.5)

    plt.suptitle("Information Entropy", fontsize=fontsize+2)

    grid[0].set_ylabel("Depth [m]")
    #
    plt.savefig("../../paper/figures/exp2_ieplot.pdf",  bbox_inches='tight',
                dpi=300)
    plt.show()