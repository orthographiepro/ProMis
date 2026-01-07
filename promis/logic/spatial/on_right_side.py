"""This module implements a distributional predicate to highlight right sides of roads."""

#
# Copyright (c) Simon Kohaut, Honda Research Institute Europe GmbH
#
# This file is part of ProMis and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with ProMis.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# Geometry
from shapely.strtree import STRtree
from shapely import LineString
from shapely.ops import split

from numpy import array, sin, cos, pi

# ProMis
from promis.geo import CartesianLocation, CartesianMap

from .relation import Relation


class OnRightSide(Relation):
    def index_to_distributional_clause(self, index: int) -> str:
        return f"{self.parameters.data['v0'][index]}::on_right_side(x_{index}, {self.location_type}) :- over(x_{index}, {self.location_type}).\n"

    @staticmethod
    def compute_relation(
        location: CartesianLocation, r_tree: STRtree, original_geometries: CartesianMap, **kwargs
    ) -> float:
        point = location.geometry
        geometry_index = r_tree.nearest(point)
        geometry = r_tree.geometries[geometry_index]
        ogeometry = original_geometries.features[geometry_index]
        # if not point.within(geometry): 
        #     return False

        # if we are on a one way street, the predicate becomes useless
        if ogeometry.tags["oneway"] == "yes":
            return True

        street_line = ogeometry.tags["line"]
        street_line = LineString([loc.to_cartesian(original_geometries.origin).to_numpy() for loc in street_line.locations])
        distances = [
            LineString([street_line.coords[i], street_line.coords[i+1]]).distance(point)
            for i in range(len(street_line.coords) - 1)
        ]
        nearest_index = min(enumerate(distances), key=lambda x: x[1])[0]
        p1 = street_line.coords[nearest_index]
        p2 = street_line.coords[nearest_index + 1]
        cross_product = (p2[0] - p1[0]) * (point.y - p1[1]) - \
            (p2[1] - p1[1]) * (point.x - p1[0])

        return (cross_product > 1e-9)

    @staticmethod
    def empty_map_parameters() -> list[float]:
        return [0.0, 0.0]

    @staticmethod
    def arity() -> int:
        return 2
