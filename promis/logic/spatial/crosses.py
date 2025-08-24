"""This module implements a distributional predicate of distances to sets of map features."""

#
# Copyright (c) Simon Kohaut, Honda Research Institute Europe GmbH
#
# This file is part of ProMis and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with ProMis.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# Geometry
from shapely.strtree import STRtree
from shapely import LineString, Geometry

import numpy as np

# ProMis
from promis.geo import CartesianLocation, CartesianMap

from .delta_relation import DeltaRelation


class Crosses(DeltaRelation):
    def index_to_distributional_clause(self, index: int) -> str:
        return f"{self.parameters.data['v0'][index]}::crosses(x_{index}, {self.location_type}).\n"

    @staticmethod
    def compute_relation(
        location: CartesianLocation, bearing: float, speed: float, r_tree: STRtree, original_geometries: CartesianMap
    ) -> float:
        velocity = speed / 3.6 * np.array([np.sin(bearing, ), np.cos(bearing)]).reshape((2,1))  # m/s, 1s naive prognosis
        trajectory = LineString([location.geometry, (location + velocity).geometry])
        geometry = r_tree.geometries.take(r_tree.nearest(location.geometry)) 
        # no need to check for loc type as all geometries we get are previously filtered
        return trajectory.crosses(geometry)

    @staticmethod
    def empty_map_parameters() -> list[float]:
        return [0.0, 0.0]

    @staticmethod
    def arity() -> int:
        return 2
