import numpy as np
from os import mkdir, path
from promis.geo import DeltaGrid, PolarLocation
from promis.loaders import OsmLoader
from promis import DeltaStaRMap, ProMis

name = "maxspeed_discrete"
fname = f"data/{name}"
if not path.exists(fname):
    mkdir(fname)

origin = PolarLocation(latitude=49.876882, longitude=8.650317)
width, height = 80.0, 80.0

dg = DeltaGrid(origin, (30, 30), width, height, bearing_res=12, speed_res=4, speed_bounds=(30, 70))

print(dg.data["speed"].unique())
print(dg.data["bearing"].unique())

feature_description = {
    "secondary": "['highway' = 'secondary']",
    "service": "['highway' = 'service']",
    "residential": "['highway' = 'residential']",
    "crossing": "['footway' = 'crossing']",
    "living_street": "[highway=living_street]",
    "signal": "['highway' = 'traffic_signals']",
}

covariance = {
    "secondary": 3 * np.eye(2),
    "residential": 2 * np.eye(2),
    "crossing": 1.5 * np.eye(2),
}

# 0.98::on_road(X) :-
#         state_speed(X, s),
#         over(X, secondary), 
#         50 >= s;
#         state_speed(X, s),
#         over(X, residential), 
#         30 >= s;
#         over(X, service).
    
#     respects_walkers(X) :- 
#         \+ crosses(X, crossing).

logic = """
    on_road(X) :-
        over(X, service);
        on_residential(X);
        on_secondary(X).

    
    on_residential(X) :-
        over(X, residential),
        state_speed(X, S),
        maxspeed(X, residential, MS),
        MS >= S.

    on_secondary(X) :-
        over(X, secondary),
        state_speed(X, S),
        maxspeed(X, secondary, MS),
        MS >= S.

    respects_walkers(X) :- 
        \+ crosses(X, crossing).

    % Definition of a valid mission
    landscape(X) :-
        on_road(X),
        respects_walkers(X).
"""

print("build uam")
uam = OsmLoader(origin, (width, height), feature_description, polygonize_routes=True).to_cartesian_map()
uam.apply_covariance(covariance)
uam.save(f"{fname}/{name}_uam.pkl")

print("build dsm")
dsm = DeltaStaRMap(uam)
dsm.initialize(dg, 15, logic)
dsm.save(f"{fname}/{name}_star.pkl")

print(dsm.get("maxspeed", "secondary").parameters.data)


print("solve promis")
promis = ProMis(dsm)
promis.solve(dg, logic, print_first=True, show_progress=True)
dg.save(f"{fname}/{name}_landscape.pkl")

print(dg.data["speed"].unique())
print(dg.data["bearing"].unique())
