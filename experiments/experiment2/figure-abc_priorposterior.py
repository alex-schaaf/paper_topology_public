import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append("../../../toolbox")
from toolbox import plot as ph
import helpers
from helpers import plot_posterior, plot_prior, plot_posteriors
from pathlib import Path
import scipy.stats

data_folder = Path("D:/datasets/paper_topology/03_gullfaks/")
simulation = Path("smc_jac_paper_hires1")
priors = helpers.load_priors(data_folder / simulation / "priors.json")
thresholds = [t for t in os.listdir(data_folder / simulation)
              if os.path.isdir(data_folder / simulation / t)]

# manual threshold selection override
thresholds = ["0.3", "0.2", "0.1", "0.075", "0.05", "0.025"]

# simulation_rej = Path("rej_jac_comparison")
# samples_rej = helpers.load_samples(
#     data_folder / simulation / "1e-06",
#     skip=("vertices", "simplices", "edges", "centroids", "lb")
# )

samples = {}
for t in thresholds:
    samples[t] = helpers.load_samples(
        data_folder / simulation / t,
        skip=("vertices", "simplices", "edges", "centroids", "lb")
    )

nrows = len(thresholds)
names = ("BCU", "tarbert B", "etive B", "fault3", "fault4")
ncols = len(names)


color_lot = {
    "BCU": "#ffbe00",
    "tarbert B": "#ff3f20",
    "etive B": "#728f02",
    "fault3": "#527682",
    "fault4": "#527682",
}
fontsize = 6
hist_kwargs = {}
labels = ("BCU Z", "Tarbert B Z", "Etive B Z", "Fault A X", "Fault B X")
with plt.rc_context(ph.get_rcparams(fontsize)):
    fig, axes = plt.subplots(
        ncols=ncols, nrows=nrows,
        sharex=True,
        figsize=ph.get_figsize(1, textwidth=helpers.TEXTWIDTH, ratio=0.7),
        sharey=True
    )
    colnames = "a b c d e".split()
    for col, name in enumerate(names):
        hist_kwargs.update({"color": color_lot[name]})
        for row, key in enumerate(
                np.sort(np.array(list(samples.keys())).astype(float))[::-1]
        ):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(f"({colnames[col]})", fontsize=fontsize+1)
            xlim = -125, 125
            ax.set_xlim(*xlim)
            ax.set_ylim(0, 0.025)
            ax.set_yticks([0, 0.025])
            if col == 0:
                ax.set_ylabel(f"ε = {key}\n\n Density")
            if row == len(thresholds) - 1:
                if col < 3:
                    ax.set_xlabel("Vertical shift [m]")
                else:
                    ax.set_xlabel("Lateral shift [m]")

            x = np.linspace(*xlim, 500)

            prior = priors[name]
            y = prior.pdf(x)
            ax.fill_between(x, y, color="lightgrey", alpha=1, label="Prior")

            label = labels[col]

            posterior = samples[str(key)][name]
            mean = posterior.mean()
            std = posterior.std()
            if row in (0, 5):
                ax.text(120, 0.02, f"μ {mean:.1f}", horizontalalignment="right")
                ax.text(120, 0.015, f"σ {std:.1f}", horizontalalignment="right")

            post_kde = scipy.stats.gaussian_kde(posterior)
            y = post_kde.pdf(x)
            ax.fill_between(
                x, y, color=color_lot[name], alpha=0.5, linewidth=0, label=label
            )
            if row == 0:
                ax.legend(prop={"size": fontsize - 1}, loc="upper left")


            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(0.5)
        # axes[0, col].set_title(name, fontsize=font_size)


    plt.savefig("../../paper/figures/exp3_distplot.pdf", bbox_inches='tight', dpi=300)
    plt.show()