import sys
sys.path.append("../../../gempy")
import gempy as gp
import numpy as np
import pandas as pd


def create_geomodel(resolution=(30, 30, 30)):
    ve = 3
    extent = [451e3, 456e3, 6.7820e6, 6.7840e6, -2309 * ve, -1651 * ve]

    geo_model = gp.create_model('Topology-Gullfaks')

    gp.init_data(geo_model, extent, resolution,
          path_o = "data/filtered_orientations.csv",
          path_i = "data/filtered_surface_points.csv", default_values=True)

    series_distribution = {
        "fault3": "fault3",
        "fault4": "fault4",
        "unconformity": "BCU",
        "sediments": ("tarbert", "ness", "etive"),
    }

    gp.map_series_to_surfaces(geo_model, 
                              series_distribution, 
                              remove_unused_series=True)

    geo_model.reorder_series(["unconformity", "fault3", "fault4",
                              "sediments", "Basement"])
    
    geo_model.set_is_fault(["fault3"])
    geo_model.set_is_fault(["fault4"])

    rel_matrix = np.array([[0,0,0,0,0],
                           [0,0,0,1,1],
                           [0,0,0,1,1],
                           [0,0,0,0,0],
                           [0,0,0,0,0]])

    geo_model.set_fault_relation(rel_matrix)
    
    surf_groups = pd.read_csv("data/filtered_surface_points.csv").group
    geo_model.surface_points.df["group"] = surf_groups
    orient_groups = pd.read_csv("data/filtered_orientations.csv").group
    geo_model.orientations.df["group"] = orient_groups
    
    geo_model.surface_points.df.reset_index(inplace=True, drop=True)
    geo_model.orientations.df.reset_index(inplace=True, drop=True)
    
    return geo_model
