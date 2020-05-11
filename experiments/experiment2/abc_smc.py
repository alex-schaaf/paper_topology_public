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
from helpers import write_priors, blockPrint, enablePrint, simulate, \
    store_sample, save_samples, prepare_stochastic_model, prepare_geomodel, \
    distance_adj_matrix, distance_jaccard
from pathlib import Path
import json
import logging
import time
from nptyping import Array
warnings.filterwarnings('ignore')


@click.command()
@click.argument("simulation_name")
@click.argument("distance")
@click.argument("n_samples")
@click.argument("thresholds")
@click.option("--debug", is_flag=True, help="Activates debug mode.")
@click.option("--resolution", default="30,30,30")
def main(simulation_name, distance, n_samples, thresholds, debug, resolution):
    os.mkdir("./" + simulation_name)
    logging_settings = dict(
        filemode="w",
        filename=f"./{simulation_name}/log.txt",
        format="%(message)s"
    )
    if debug:
        logging.basicConfig(level=logging.DEBUG, **logging_settings)
    else:
        logging.basicConfig(level=logging.INFO, **logging_settings)
    logging.info(f"Name: {simulation_name}")
    logging.info(f"Distance: {distance}")
    settings = dict(
        compute_topology=True,
        record_lithblock=True,
        record_blockmatrix=False,
        record_surfaces=True,
        record_topology=True,
        clean_topology=1
    )
    logging.info(f"Resolution: {resolution}")
    resolution = np.array(resolution.split(",")).astype(int)
    geo_model, edges_init, centroids_init = prepare_geomodel(resolution=resolution)
    if distance == "matrix":
        matrix_init = tp.get_adjacency_matrix(
            geo_model, edges_init, centroids_init)
    stochastic_model = prepare_stochastic_model(geo_model)
    # save priors
    
    write_priors(
        {name: v["dist"] for name, v in stochastic_model.priors.items()},
        Path(simulation_name) / "priors.json"
    )

    n_samples = list(map(int, n_samples.split(",")))
    thresholds = list(map(float, thresholds.split(",")))
    n_epochs = len(thresholds)
    n_iter = np.zeros(n_epochs, dtype=int)
    n_accepted = np.zeros(n_epochs, dtype=int)

    for epoch, threshold in enumerate(thresholds):
        logging.info("\n")
        logging.info(f"Epoch: {epoch}")
        logging.info(f"Threshold: {threshold}")
        t0 = time.time()

        storage = {prior: [] for prior in stochastic_model.priors.keys()}
        storage_model = {}
        pbar = tqdm.tqdm(total=n_samples[epoch])

        while n_accepted[epoch] < n_samples[epoch]:
            # sample
            surfpts_samples, orients_samples = stochastic_model.sample()
            # modify
            stochastic_model.modify(surfpts_samples, orients_samples)
            # simulate & compute summary statistics (topology)
            edges, centroids = simulate(stochastic_model, settings)
            # distance
            if distance == "jaccard":
                d = distance_jaccard(edges, edges_init)
            elif distance == "matrix":
                matrix = tp.get_adjacency_matrix(geo_model, edges, centroids)
                d = distance_adj_matrix(matrix, matrix_init, normalize=False)
            logging.debug(f"Distance: {d}")
            # acceptance / rejection
            if d < threshold:
                logging.debug(f"Sample accepted.")
                n_accepted[epoch] += 1
                pbar.update(1)
                for k, v in surfpts_samples.items():
                    storage[k].append(v)
                store_sample(geo_model, storage_model)
            n_iter[epoch] += 1
        # epoch over -> adjust priors with accepted samples
        stochastic_model.reset()  # ? necessary ?
        for name, values in storage.items():
            kde = scipy.stats.kde.gaussian_kde(values)
            stochastic_model.priors[name]["dist"] = kde

        acceptance_rate = n_accepted[epoch] / n_iter[epoch]
        logging.info(f"Acceptance rate: {acceptance_rate:.05f}")
        t1 = time.time()
        epoch_time = t1 - t0
        logging.info(f"Time: {int(epoch_time)} (s)")
        # fp = Path(simulation_name)
        fp = "./" + simulation_name + "/" + str(threshold)
        os.mkdir(fp)
        save_samples(fp, storage)
        save_samples(fp, storage_model)
        


if __name__ == "__main__":
    main()