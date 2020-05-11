import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../../toolbox")
from toolbox import plot as ph
import helpers

n = 1000
rej_thresh = [0.025]
rej_times = np.array([0, 0, 0, 0, 0, 144128, 0]) / 60 / 60
smc_thresh = np.array([0.3, 0.2, 0.1, 0.075, 0.05, 0.025, 0.01])
smc_times = np.array([2817, 3176, 1788, 1514, 1950, 3018, 2180]) / 60 / 60

print(f"t rej: {rej_times.sum()}")
print(f"t smc: {smc_times[:-1].sum()}")
print(f"ratio: {rej_times.sum() / smc_times[:-1].sum()}")

font_size = 6
with plt.rc_context(ph.get_rcparams(font_size)):
    fig, ax = plt.subplots(
        figsize=ph.get_figsize(0.5, textwidth=helpers.TEXTWIDTH)
    )
    barwidth = 0.4
    locs = np.arange(len(smc_thresh))

    ax.bar(
        locs + 0.2, smc_times,
        width=barwidth, label="ABC-SMC", color="black"
    )
    ax.bar(
        locs - 0.2, rej_times,
        width=barwidth, label="ABC-REJ", color="grey"
    )
    ax.legend(fontsize=font_size)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)

    plt.savefig(
        "../../paper/figures/exp2_speed.pdf",  bbox_inches='tight', dpi=300
    )
    plt.show()
