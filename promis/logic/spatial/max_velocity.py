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

# ProMis
from promis.geo import CartesianLocation, CartesianMap, CartesianCollection
from promis.geo.route import Route
from promis.geo.polygon import Polygon

from .relation import ScalarRelation


# TODO only nearest neighbor 
class MaxVelocity(ScalarRelation):
    def __init__(self, parameters: CartesianCollection, location_type: str) -> None:
        super().__init__(parameters, location_type, problog_name="maxspeed")

    @staticmethod
    def compute_relation(
        location: CartesianLocation, r_tree: STRtree, original_geometries: CartesianMap, **kwargs
    ) -> float:
        index = r_tree.nearest(location.geometry)
        original_geo = original_geometries.features[index]
        if "maxspeed" not in original_geo.tags or not location.geometry.within(r_tree.geometries.take(index)):
            return -1
        return int(original_geo.tags["maxspeed"])

    @staticmethod
    def empty_map_parameters() -> list[float]:
        return [0.0, 0.25]

    @staticmethod
    def arity() -> int:
        return 2
