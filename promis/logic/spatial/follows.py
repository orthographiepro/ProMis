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
from promis.geo import CartesianLocation, CartesianMap, CartesianPolyLine, PolarPolyLine

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

        # m/s, 1s naive prognosis
        velocity = speed / 3.6 * array([sin(bearing), cos(bearing)])
        point = location.geometry
        geometry_index = r_tree.nearest(point)
        ogeometry = original_geometries.features[geometry_index]

        t_vector = (velocity[0], velocity[1])

        street_line = ogeometry.tags["line"]
        street_line = LineString([loc.to_cartesian(original_geometries.origin).to_numpy() for loc in street_line.locations])

        point = location
        geom_index = r_tree.nearest(point.geometry)
        ogeometry = original_geometries.features[geom_index]

        polar_line: PolarPolyLine = ogeometry.tags["line"]
        street_line: CartesianPolyLine = polar_line.to_cartesian(original_geometries.origin)
        coords = street_line.geometry.coords

        if "tree" not in ogeometry.tags.keys():
            segments = [LineString(coords[i:i+2]) for i in range(len(coords)-1)]
            segments = STRtree(segments, 4)
            ogeometry.tags["tree"] = segments
        else: 
            segments = ogeometry.tags["tree"]
        
        part_index = segments.nearest(point.geometry)
        a, b = coords[part_index:part_index+2]
        s_vector = array((b[0] - a[0], b[1] - a[1]))

        dot_product = sum([v1*v2 for v1, v2 in zip(s_vector, t_vector)])
        return (dot_product > 1e-9)

    @staticmethod
    def empty_map_parameters() -> list[float]:
        return [0.0, 0.0]

    @staticmethod
    def arity() -> int:
        return 2
