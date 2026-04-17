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

from numpy import array, arccos, rad2deg
from numpy.linalg import norm

# ProMis
from promis.geo import CartesianLocation, CartesianMap, PolarPolyLine, CartesianPolyLine, CartesianCollection

from .relation import ScalarRelation

DEFAULT_UNIFORM_VARIANCE = 0.25

class Angle(ScalarRelation):

    def __init__(self, parameters: CartesianCollection, location_type: str) -> None:
        super().__init__(parameters, location_type, problog_name="angle")

    @staticmethod
    def compute_relation(
        location: CartesianLocation, r_tree: STRtree, original_geometries: CartesianMap, **kwargs
    ) -> float:
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
        vec = array((b[0] - a[0], b[1] - a[1]))

        return rad2deg(arccos( vec[1] / norm(vec)))


    @staticmethod
    def empty_map_parameters() -> list[float]:
        return [0.0, DEFAULT_UNIFORM_VARIANCE]

    @staticmethod
    def arity() -> int:
        return 2
