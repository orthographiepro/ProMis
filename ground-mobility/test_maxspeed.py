from promis import ProMis, StaRMap
from promis.geo import PolarLocation, CartesianMap, CartesianLocation, CartesianRasterBand, CartesianCollection
from promis.loaders import OsmLoader
from numpy import eye
import matplotlib.pyplot as plt

from promis.geo.polygon import Polygon, CartesianPolygon
from promis.geo.route import Route, CartesianRoute
import numpy as np
from pathlib import Path

# The features we will load from OpenStreetMap
# The dictionary key will be stored as the respective features location_type
# The dictionary value will be used to query the relevant geometry via Overpass
feature_description = {
    # "park": "['leisure' = 'park']",
    #"primary": "['highway' = 'primary']",
    "secondary": "['highway' = 'secondary']",
    #"tertiary": "['highway' = 'tertiary']",
    "service": "['highway' = 'service']",
    "residential": "['highway' = 'residential']",
    "crossing": "['footway' = 'crossing']",
    "living_street": "[highway=living_street]",
    #"unclassified": "[highway=unclassified]",
    # "bay": "['natural' = 'bay']",
    #"rail": "['railway' = 'rail']",
    # oneway?
    # give way, stop?
    "signal": "['highway' = 'traffic_signals']",
}

# Covariance matrices for some of the features
# Used to draw random translations representing uncertainty for the respective features
covariance = {
    "secondary": 3 * eye(2),
    "residential": 2 * eye(2),
    # cov of traffic_signal?
}

# The probabilistic, logical constraints to fulfill during a mission

logic = """
    on_road(X) :-
        over(X, secondary);
        over(X, residential);
        over(X, service).
    
    % Definition of a valid mission
    landscape(X) :-  
        on_road(X),
        maxspeed(X, secondary) > 40,
        maxspeed(X, residential) > 40.
"""

origin = PolarLocation(latitude=49.876882, longitude=8.650317)
width, height = 80.0, 80.0
number_of_random_maps = 25
support = CartesianRasterBand(origin, (80, 80), width, height)  # This is the set of points that will be directly computed through sampling (expensive)
target = CartesianRasterBand(origin, (250, 250), width, height)  # This is the set of points that will be interpolated from the support set (cheap)

uam = OsmLoader(origin, (width, height), feature_description, polygonize_routes=True).to_cartesian_map()
uam.apply_covariance(covariance)

star_map = StaRMap(target, uam)
star_map.initialize(support, number_of_random_maps, logic)  # This estimates all spatial relations that are relevant to the given logic

promis = ProMis(star_map)
landscape = promis.solve(support, logic, n_jobs=4, batch_size=1, print_first=True, show_progress=True)

axes = plt.subplot()
landscape.scatter(s=0.4, plot_basemap=True, rasterized=True, cmap="coolwarm_r", alpha=0.2, ax=axes)
support.scatter(s=0.4, ax=axes, plot_basemap=False)
plt.show()

rel, loc_type = "maxspeed", "residential"
params = star_map.get(rel, loc_type).parameters
image = params.scatter(value_index=0, s=0.4, plot_basemap=True, alpha=0.5, cmap="coolwarm_r")
cbar = plt.colorbar(image, aspect=30, pad=0.02)
cbar.solids.set(alpha=0.8)
plt.title(f"{rel}(X, {loc_type})")
plt.show()
