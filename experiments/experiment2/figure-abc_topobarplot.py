import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append("../../../toolbox")
from toolbox import plot as ph
import helpers
from pathlib import Path
sys.path.append("../../../gempy")
from gempy.assets import topology as tp
import scipy.stats

data_folder = Path("D:/datasets/paper_topology/03_gullfaks/")
simulation = Path("smc_jac_paper")
priors = helpers.load_priors(data_folder / simulation / "priors.json")
thresholds = [t for t in os.listdir(data_folder / simulation)
              if os.path.isdir(data_folder / simulation / t)]

samples = {}

skip = ["vertices", "simplices", "centroids", "lb"] + list(priors.keys())
for t in thresholds:
    samples[t] = helpers.load_samples(
        data_folder / simulation / t,
        skip=skip
    )

nrows = len(samples.keys())
fig, axes = plt.subplots(
    nrows=nrows,
    figsize=ph.get_figsize(1, ratio=0.5)
)

for ax, (threshold, sample) in zip(axes, samples.items()):
    print(f"threshold: {threshold}")
    ax.set_title(str(threshold))
    edges = sample.get("edges")
    print(len(edges))
    if not edges:
        continue
    u, c, idx = tp.count_unique_topologies(edges)

    print(f"# unique topologies: {len(u)}")
    # ax.bar(range(len(c)), c.sort())

plt.show()