import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("../../../toolbox")
from toolbox import plot as ph
import helpers

means = pd.read_csv("data/smc_means.csv")
means.set_index("threshold", inplace=True)
stdevs = pd.read_csv("data/smc_stdevs.csv")
stdevs.set_index("threshold", inplace=True)

color_lot = {
    "BCU": "#ffbe00",
    "ness": "#000000",
    "tarbert": "#ff3f20",
    "etive": "#728f02",
    "fault3": "#527682",
    "fault4": "#527682",
}
font_size = 6
with plt.rc_context(ph.get_rcparams(font_size)):
    fig, axes = plt.subplots(
        ncols=3, nrows=4, sharex=True, sharey=True,
        figsize=ph.get_figsize(1, textwidth=helpers.TEXTWIDTH)
    )
    params_combinations = [
        ("BCU", "fault3", "fault4"),
        ("tarbert A", "tarbert B", "tarbert C"),
        ("ness A", "ness B", "ness C"),
        ("etive A", "etive B", "etive C")
    ]
    for row, params in enumerate(params_combinations):
        for col, param in enumerate(params):
            ax = axes[row, col]
            ax.axhline(0, color="black", linewidth=0.5, linestyle="dashed")
            colorname = param.split(" ")[0]
            x = means[param].index
            y = means[param].values
            c = color_lot[colorname]
            ax.plot(x, y, color=c)
            ax.fill_between(
                x,
                y + stdevs[param].values,
                y - stdevs[param].values,
                color=c, alpha=0.3
            )
            # ax.errorbar(x, y, yerr=stdevs[param].values, c=c, linewidth=0.5)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(0.5)
    plt.show()