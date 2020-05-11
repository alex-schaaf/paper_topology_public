import warnings
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=cuda"
import sys
from copy import deepcopy
import numpy as np
import click
sys.path.append("../../../../git/gempy")
import gempy as gp
from gempy.assets import topology as tp
from gempy.utils import stochastic_surface as ss
import tqdm
import scipy.stats
import pickle
sys.path.append("../../")
from helpers import write_priors, blockPrint, enablePrint, store_sample, save_samples, prepare_stochastic_model, prepare_geomodel
from pathlib import Path
import json
warnings.filterwarnings('ignore')
import logging
import time


def distance(edges: set, observed: set) -> float:
    return 1 - tp.jaccard_index(edges, observed)


@click.command()
@click.argument("simulation_name")
@click.argument("n_samples")
@click.argument("threshold")
def main(simulation_name, n_samples, threshold):
    settings = dict(
        compute_topology=True,
        record_lithblock=True,
        record_blockmatrix=False,
        record_surfaces=True,
        record_topology=True,
        clean_topology=1
    )
    logging_settings = dict(
        filemode="w",
        filename="./simulation_name/log.txt",
        format="%(message)s"
    )
    logging.basicConfig(level=logging.INFO, **logging_settings)
    # initialize
    geo_model, edges_init, centroids_init = prepare_geomodel()
    stochastic_model = prepare_stochastic_model(geo_model)
    # save priors
    os.mkdir("./" + name)
    write_priors(
        {name: v["dist"] for name, v in stochastic_model.priors.items()},
         Path(simulation_name) / "priors.json"
    )
    n_iter, n_accepted = 0, 0
    n_samples = int(n_samples)
    threshold = float(threshold)
    pbar = tqdm.tqdm(total=n_samples)

    storage = {prior: [] for prior in stochastic_model.priors.keys()}
    logging.info(f"Threshold: {threshold}")
    t0 = time.time()
    while n_accepted < n_samples:
        # sample
        surfpts_samples, orients_samples = stochastic_model.sample()
        # modify
        stochastic_model.modify(surfpts_samples, orients_samples)
        # simulate & compute summary statistics (topology)
        edges, centroids = simulate(stochastic_model)
        # distance
        d = distance(edges, edges_init)
        # acceptance criteria
        if d < threshold:
            n_accepted += 1
            pbar.update(1)
            for k, v in surfpts_samples.items():
                storage[k].append(v)
            store_sample(geo_model, storage)
        n_iter += 1

    acceptance_rate = n_accepted / n_iter
    logging.info(f"Acceptance rate: {acceptance_rate:.05f}")
    t1 = time.time()
    epoch_time = t1 - t0
    logging.info(f"Time: {int(epoch_time)} (s)")
    save_samples(simulation_name, storage, threshold)


if __name__ == "__main__":
    main()
