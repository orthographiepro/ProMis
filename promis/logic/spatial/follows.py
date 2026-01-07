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

from numpy import array, sin, cos, pi

# ProMis
from promis.geo import CartesianLocation, CartesianMap

from .relation import DeltaRelation


class Follows(DeltaRelation):
    def index_to_distributional_clause(self, index: int) -> str:
        return f"{self.parameters.data['v0'][index]}::follows(x_{index}, {self.location_type}) :- over(x_{index}, {self.location_type}).\n"

    @staticmethod
    def compute_relation(
        location: CartesianLocation, r_tree: STRtree, original_geometries: CartesianMap, **kwargs
    ) -> float:
        if not "speed" in kwargs and not "bearing" in kwargs:
            raise KeyError(f"compute_relation called with insufficient kwargs. Expected 'bearing' and 'speed', got {kwargs}")
        speed = kwargs["speed"]
        bearing = kwargs["bearing"] / 180 * pi

        # m/s, 1s naive prognosis
        velocity = speed / 3.6 * array([sin(bearing), cos(bearing)])
        point = location.geometry
        geometry_index = r_tree.nearest(point)
        ogeometry = original_geometries.features[geometry_index]

        t_vector = (velocity[0], velocity[1])

        street_line = ogeometry.tags["line"]
        street_line = LineString([loc.to_cartesian(original_geometries.origin).to_numpy() for loc in street_line.locations])

        # Project the point onto the line to get the distance along the line
        distance_on_line = street_line.project(point)

        # Find the segment index that contains the projected point
        nearest_index = -1
        cumulative_distance = 0.0
        coords = street_line.coords
        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i+1]
            segment_length = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
            if cumulative_distance <= distance_on_line <= cumulative_distance + segment_length:
                nearest_index = i
                break
            cumulative_distance += segment_length
        if nearest_index == -1:
            nearest_index = len(coords) - 2

        s_vector = [
            street_line.coords[nearest_index + 1][0] - street_line.coords[nearest_index][0],
            street_line.coords[nearest_index + 1][1] - street_line.coords[nearest_index][1],
        ]

        dot_product = sum([a*b for a, b in zip(s_vector, t_vector)])
        return (dot_product > 1e-9)

    @staticmethod
    def empty_map_parameters() -> list[float]:
        return [0.0, 0.0]

    @staticmethod
    def arity() -> int:
        return 2
