import numpy as np
import matplotlib.pyplot as plt
from promis.geo import CartesianDeltaCollection, PolarDeltaCollection, CartesianCollection, PolarLocation, CartesianRasterBand, PolarRasterBand

origin = PolarLocation(latitude=49.876882, longitude=8.650317)
width, height = 80.0, 80.0
number_of_random_maps = 25


coll = PolarDeltaCollection(origin, 1)
bear_grid = np.linspace(0, 300, 6)
speed_grid = np.linspace(20, 60, 5)

support = PolarRasterBand(origin, (5, 5), width, height)  # This is the set of points that will be directly computed through sampling (expensive)
for s in speed_grid:
    for b in bear_grid:
        coll.append(support.coordinates(), np.random.uniform(0,1, 25), np.repeat(b, 25), np.repeat(s, 25))
        # print(b, s)

print(coll.values_for(bearing=300, speed=40).shape)

coll.scatter(bearing=300, speed=40, plot_basemap=True)



