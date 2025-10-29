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

from numpy import max, sum
from pandas import isna

# ProMis
from promis.geo import CartesianLocation, CartesianMap

from .relation import DiscreteRelation

LIMIT = -1

# TODO only nearest neighbor 
class MaxVelocity(DiscreteRelation):
    def index_to_distributional_clause(self, index: int) -> str:
        data = self.parameters.data

        return "\n".join([
            f"{data['p'+str(case)][index]}::maxspeed(x_{index}, {self.location_type}, {case})."
            for case in self.cases if not isna(data['p'+str(case)][index])
            ])+"\n"

    @staticmethod
    def compute_relation(
        location: CartesianLocation, r_tree: STRtree, original_geometries: CartesianMap, **kwargs
    ) -> float:
        index = r_tree.nearest(location.geometry)
        original_geo = original_geometries.features[index]
        if "maxspeed" not in original_geo.tags or not location.geometry.within(r_tree.geometries.take(index)):
            return LIMIT
        return int(original_geo.tags["maxspeed"])
    
    # unneeded
    @classmethod
    def _moment_functions(cls):
        """note that these are not technically moments, especially not mean / variance"""
        return max, cls._prob_max
    #unneeded
    @staticmethod
    def _prob_max(data, axis):
        """Count, how often the maximum value appears in data"""
        return sum(data == max(data, axis=axis), axis=axis) / len(data)

    @staticmethod
    def empty_map_parameters() -> list[float]:
        return [LIMIT, 1]

    @staticmethod
    def arity() -> int:
        return 3
