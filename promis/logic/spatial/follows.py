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
        return f"{self.parameters.data['v0'][index]}::follows(x_{index}, {self.location_type}).\n"

    @staticmethod
    def compute_relation(
        location: CartesianLocation, r_tree: STRtree, original_geometries: CartesianMap, **kwargs
    ) -> float:
        if not "speed" in kwargs and not "bearing" in kwargs:
            raise KeyError(f"compute_relation called with insufficient kwargs. Expected 'bearing' and 'speed', got {kwargs}")
        speed = kwargs["speed"]
        bearing = kwargs["bearing"] / 180 * pi

        velocity = speed / 3.6 * array([sin(bearing), cos(bearing)]).reshape((2,1))  # m/s, 1s naive prognosis
        point = location.geometry
        geometry_index = r_tree.nearest(point)
        geometry = r_tree.geometries[geometry_index]
        ogeometry = original_geometries.features[geometry_index]
        if not point.within(geometry): 
            return False

        trajectory = LineString([point, (location + velocity).geometry]).coords
        t_vector = (
            trajectory[1][0] - trajectory[0][0],
            trajectory[1][1] - trajectory[0][1],
        )

        street_line: LineString = ogeometry.tags["line"]
        distances = [
            LineString([street_line.coords[i], street_line.coords[i+1]]).distance(point)
            for i in range(len(street_line.coords) - 1)
        ]
        nearest_index = min(enumerate(distances), key=lambda x: x[1])[0]
        s_vector = [
            street_line.coords[nearest_index + 1][0] - street_line.coords[nearest_index][0],
            street_line.coords[nearest_index + 1][1] - street_line.coords[nearest_index][1],
        ]

        dot_product = sum([a*b for a, b in zip(s_vector, t_vector)])
        return dot_product > 1e-9

    @staticmethod
    def empty_map_parameters() -> list[float]:
        return [0.0, 0.0]

    @staticmethod
    def arity() -> int:
        return 2
