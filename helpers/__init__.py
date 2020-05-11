import json
import scipy.stats
import sys
import os
import numpy as np
import pickle
from pathlib import Path
from create_geomodel_gullfaks import create_geomodel
sys.path.append("../gempy")
import gempy as gp
from nptyping import Array
from gempy.assets import topology as tp
from gempy.utils import stochastic_surface as ss
import logging
from typing import Union

TEXTWIDTH = 503.61377


def distance_adj_matrix(matrix:Array, observed:Array, normalize:bool=False):
    diffsum = np.sum(matrix ^ observed)
    if normalize:
        return diffsum / np.product(matrix.shape)
    else:
        return diffsum
from typing import Iterable
import matplotlib.pyplot as plt


def distance_jaccard(edges: set, observed: set) -> float:
    return 1 - tp.jaccard_index(edges, observed)


def write_priors(priors: dict, fp: str):
    prior_vals = {}
    for name, dist in priors.items():
        prior_vals[name] = (
            dist.dist.__str__().split()[0].split(".")[-1].split("_")[0],
            dist.std(),
            dist.mean(),
        )
    with open(fp, "w") as f:
        f.write(
            json.dumps(prior_vals)
        )


def load_priors(fp: str) -> dict:
    with open(fp, "r") as f:    
        priors = json.load(f)
    logging.debug(priors)
    return {
        name: scipy.stats.__dict__[type_](loc, scale) for name,
        (type_, scale, loc) in priors.items()
    }


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__


def store_sample(geo_model, storage: dict):
    keys = storage.keys()
    if "lb" in keys:
        storage["lb"].append(geo_model.solutions.lith_block)
    else:
        storage["lb"] = [geo_model.solutions.lith_block]
    if "vertices" in keys:
        storage["vertices"].append(geo_model.solutions.vertices)
    else:
        storage["vertices"] = [geo_model.solutions.vertices]
    if "simplices" in keys:
        storage["simplices"].append(geo_model.solutions.edges)
    else:
        storage["simplices"] = [geo_model.solutions.edges]
    if "edges" in keys:
        storage["edges"].append(geo_model.solutions.topo_edges)
    else:
        storage["edges"] = [geo_model.solutions.topo_edges]
    if "centroids" in keys:
        storage["centroids"].append(geo_model.solutions.topo_centroids)
    else:
        storage["centroids"] = [geo_model.solutions.topo_centroids]


def save_samples(simulation_name: str, storage: dict):
    for k, v in storage.items():
        fp = simulation_name + "/" + k
        logging.debug(f"{fp}")
        if k in ("edges", "centroids"):
            with open(fp + ".pkl", "wb") as f:
                pickle.dump(v, f)
        else:
            np.save(fp + ".npy", v)


def load_sample(fp: Union[str, Path]):
    fp = str(fp)
    if fp.endswith(".pkl") or fp.endswith(".pickle") or fp.endswith(".p"):
        with open(fp, "rb") as f:
            return pickle.load(f)
    elif fp.endswith(".npy"):
        return np.load(fp)


def load_samples(fp: Path, skip: Iterable = None):
    samples = {}
    for fn in os.listdir(fp):
        skip_ = False
        if skip:
            for name in skip:
                if name in fn:
                    skip_ = True
        
        if skip_:
            continue
        if fn.endswith(".pkl") or fn.endswith(".pickle") or fn.endswith(".p"):
            with open(fp / fn, "rb") as f:
                samples[fn.split(".")[0]] = pickle.load(f)
        elif fn.endswith(".npy"):
            samples[fn.split(".")[0]] = np.load(fp / fn)
    return samples


def prepare_stochastic_model(geo_model):
    groups = geo_model.orientations.df.group.unique().astype(str)
    stochastic_model = ss.StochasticModel(geo_model)
    zfactor = 0.004
    xfactor = 0.0001

    print("Uncertainty parametrization\n")
    for group in groups:
        filter_ = geo_model.surface_points.df.group == group
        average_z = abs(geo_model.surface_points.df[filter_].Z.mean())
        average_x = abs(geo_model.surface_points.df[filter_].X.mean())
        if "fault" in group:
            stdev_x = average_x * xfactor
            dist = scipy.stats.norm(loc=0, scale=stdev_x)
            stochastic_model.prior_surface_single(
                group, dist, column="X", grouping="group", name=group)
            logging.info(f"{group.ljust(15)}σ = {average_x * xfactor:.01f} X")

        else:
            stdev_z = average_z * zfactor
            dist = scipy.stats.norm(loc=0, scale=stdev_z)
            stochastic_model.prior_surface_single(
                group, dist, column="Z", grouping="group", name=group)
            logging.info(f"{group.ljust(15)}σ = {average_z * zfactor:.01f} Z")

    return stochastic_model


def prepare_geomodel(resolution=(30, 30, 30)):
    blockPrint()
    geo_model = create_geomodel(resolution=resolution)
    enablePrint()
    try:
        gp.set_interpolation_data(
            geo_model,
            output=['geology'],
            compile_theano=True,
            theano_optimizer='fast_run',  # fast_compile, fast_run
            dtype="float64",  # for model stability
        )
    except:
        gp.set_interpolation_data(
            geo_model,
            output='geology',
            compile_theano=True,
            theano_optimizer='fast_run',  # fast_compile, fast_run
            dtype="float64",  # for model stability
        )

    gp.compute_model(geo_model)
    # surface_points_init = deepcopy(geo_model.surface_points.df)
    # orientations_init = deepcopy(geo_model.orientations.df)
    topo_init = tp.compute_topology(geo_model, voxel_threshold=1)
    # edges_init, centroids_init = tp.clean_unconformity_topology(
    #     geo_model, 1, *topo_init
    #     )
    return geo_model, topo_init[0], topo_init[1] #edges_init, centroids_init


def simulate(stochastic_model, settings):
    # simulate
    gp.compute_model(
        stochastic_model.geo_model,
        compute_mesh=settings.get("record_surfaces")
        )
        # compute summary statistic (topology)
    edges, centroids = tp.compute_topology(
        stochastic_model.geo_model,
        voxel_threshold=1
        )
    edges, centroids = tp.clean_unconformity_topology(
            stochastic_model.geo_model,
            settings.get("clean_topology"),
            edges,
            centroids
        )
    stochastic_model.geo_model.solutions.topo_edges = edges
    stochastic_model.geo_model.solutions.topo_centroids = centroids
    return edges, centroids


def plot_posteriors(samples: dict, name: str, axes=None, hist_kwargs={}):
    if axes is None:
        fig, axes = plt.subplots(nrows=len(samples.keys()), sharex=True,
                                 figsize=(5, 10), sharey=True)
    # for row, (threshold, tsamples) in enumerate(samples.items()):
    for row, key in enumerate(
            np.sort(np.array(list(samples.keys())).astype(float))[::-1]):
        plot_posterior(samples, name, key, ax=axes[row],
                       hist_kwargs=hist_kwargs)
    axes[0].set_title(name)
    plt.setp(axes, ylim=(0, 0.06))


def plot_posterior(samples: dict, name: str, threshold: str, ax=None,
                   hist_kwargs={}):
    if not ax:
        fig, ax = plt.subplots()

    tsamples = samples[str(threshold)]
    vals = tsamples[name]
    std = np.std(vals)
    mean = np.mean(vals)

    label = f"ε {threshold}\nμ {mean:.02f}\nσ {std:.01f}"

    hkwargs = dict(bins=24, color="black", density=True)
    hkwargs.update(hist_kwargs)
    ax.hist(
        vals,
        label=label, **hkwargs
    )
    ax.legend(
        frameon=False, loc="upper left",
        handlelength=0, handletextpad=0
    )

    ax.axvline(mean, color="black", linewidth=1, linestyle="dashed", alpha=0.5)
    ax.fill_betweenx([0, 0.06], mean - std, mean + std, color="lightgrey",
                     alpha=0.5)


def plot_prior(priors, name, axes):
    x = np.linspace(-70, 70, 500)
    y = priors.get(name).pdf(x)
    for ax in axes.flat:
        ax.plot(x, y, linewidth=1, color="black")