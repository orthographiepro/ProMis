"""This module defines a class for handling regular spaced data in 4 dimensions, akin to RasterBand."""

#
# Copyright (c) Simon Kohaut, Honda Research Institute Europe GmbH, Felix Divo, and contributors
#
# This file is part of ProMis and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with ProMis.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

from numpy import vstack, ravel, meshgrid, ndarray, linspace, concatenate, zeros
from pandas import DataFrame
from scipy.interpolate import RegularGridInterpolator

from promis.geo import RasterBand, CartesianDeltaCollection


class DeltaGrid(RasterBand, CartesianDeltaCollection): 
    def __init__(self, origin, resolution, width, height, number_of_values = 1, 
                    bearing_res=12, 
                    bearing_bounds=(0, 360),
                    speed_res=5,
                    speed_bounds=(30, 70),
                ):
        RasterBand.__init__(self, resolution, width, height)
        CartesianDeltaCollection.__init__(self, origin, number_of_values)

        self.bearing_res = bearing_res
        self.bearing_bounds = bearing_bounds
        self.speed_res = speed_res
        self.speed_bounds = speed_bounds

        # Compute coordinates from spatial dimensions and resolution
        raster_coordinates = vstack(
            list(map(ravel, meshgrid(
                self._speed_coordinates,
                self._bearing_coordinates,
                self._y_coordinates,
                self._x_coordinates,
                indexing="ij",
            )))[::-1]
        ).T

        # Put coordinates and default value 0 together into matrix and set DataFrame
        raster_entries = concatenate(
            (raster_coordinates, zeros((raster_coordinates.shape[0], number_of_values))), axis=1
        )
        self.data = DataFrame(raster_entries, columns=self.data.columns)

    @property
    def _x_coordinates(self) -> ndarray:
        return linspace(-self.width / 2, self.width / 2, self.resolution[0])

    @property
    def _y_coordinates(self) -> ndarray:
        return linspace(-self.height / 2, self.height / 2, self.resolution[1])

    @property
    def _bearing_coordinates(self) -> ndarray:
        return linspace(*self.bearing_bounds, self.bearing_res)

    @property
    def _speed_coordinates(self) -> ndarray:
        return linspace(*self.speed_bounds, self.speed_res)

    def get_interpolator(self, method = "linear"):
        grid_axes = [self._x_coordinates, self._y_coordinates, self._bearing_coordinates, self._speed_coordinates]
        values = self.data[self.data.columns[4:]].to_numpy().reshape(
                [len(grid_axes[i]) for i in range(len(grid_axes))] + [self.number_of_values],
                order="F",
        )
        return RegularGridInterpolator(
            grid_axes,
            values,
            method=method,
            bounds_error=False,
            fill_value=None,
        )
