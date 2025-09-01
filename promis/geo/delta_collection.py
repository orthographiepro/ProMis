"""This module contains a class for handling a collection of spatially referenced data."""

#
# Copyright (c) Simon Kohaut, Honda Research Institute Europe GmbH
#
# This file is part of ProMis and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with ProMis.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# Standard Library
from abc import ABC
from pickle import dump, load
from typing import Any

import smopy
from matplotlib import pyplot as plt

# Third Party
from numpy import hstack, zeros
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.interpolate import RegularGridInterpolator

# ProMis
from promis.geo.location import CartesianLocation, PolarLocation
from promis.geo.collection import Collection, CartesianCollection, PolarCollection


class DeltaCollection(Collection):
    """A collection of values over a polar or Cartesian space.

    Locations are stored as Cartesian coordinates, but data can be unpacked into both
    polar and Cartesian frames.

    Args:
        origin: The polar coordinates of this collection's Cartesian frame's center
        data: A list of Cartesian location and value pairs
    """

    def __init__(
        self, columns: list[str], origin: PolarLocation, number_of_values: int = 1
    ) -> None:
        # Attributes setup
        self.number_of_values = number_of_values
        self.origin = origin
        self.basemap = None

        # Initialize the data frame
        self.data = DataFrame(columns=columns)


    def values(self) -> NDArray[Any]:
        """Unpack the location values as numpy array.

        Returns:
            The values of this Collection as numpy array
        """

        value_columns = self.data.columns[4:]
        return self.data[value_columns].to_numpy()
    
    def values_for(self, bearing, speed) -> NDArray[Any]:
        """Unpack the location values for a certain bearing and speed as numpy array.

        Returns:
            The values of this Collection as numpy array
        """

        value_columns = self.data.columns[4:]
        return self.data[(self.data['bearing']==bearing) & (self.data['speed']==speed)][value_columns].to_numpy()
    
    def coordinates_for(self, bearing, speed) -> NDArray[Any]:
        location_columns = self.data.columns[:2]
        return self.data[(self.data['bearing']==bearing) & (self.data['speed']==speed)][location_columns].to_numpy()

    def coordinates(self) -> NDArray[Any]:
        """Unpack the location coordinates in combination to bearing and speed as numpy array.

        Returns:
            The indices of this Collection as numpy array
        """

        location_columns = self.data.columns[:4]
        return self.data[location_columns].to_numpy()

    def append(
        self,
        coordinates: NDArray[Any] | list[PolarLocation | CartesianLocation],
        values: NDArray[Any],
    ):
        """Append location and associated value vectors to collection. Coordinates should either be 2 or 4 dimensional,
        if 2D, bearing and speed are assumed to be 0.

        Args:
            coordinates: A list of locations to append or matrix of coordinates.
            values: The associated values as 2D matrix, each row belongs to a single location
        """

        if isinstance(coordinates, list):
            coordinates = hstack([c.to_numpy() for c in coordinates]).T
        if coordinates.shape[1] == 2:
            coordinates = hstack((coordinates, zeros(coordinates.shape)))
    
        Collection.append(self, coordinates, values)

    def scatter(
        self, value_index: int = 0, bearing: float = None, speed: float = None,  plot_basemap=True, ax=None, zoom=16, **kwargs
    ):
        """Create a scatterplot of this Collection.

        Args:
            value_index: Which value of the
            plot_basemap: Whether an OpenStreetMap tile shall be rendered below
            ax: The axis to plot to, default pyplot context if None
            zoom: The zoom level of the OSM basemap, default 16
            **kwargs: Args passed to the matplotlib scatter function
        """

        # Would cause circular import if done at module scope
        from promis.loaders import SpatialLoader

        # Either render with given axis or default context
        if ax is None:
            ax = plt.gca()

        # Render base map
        if plot_basemap:
            if self.basemap is None:
                self.basemap = self.get_basemap(zoom)
            ax.imshow(self.basemap, extent=self.extent())

        # Scatter collection data
        coordinates = self.coordinates_for(bearing, speed)
        colors = self.values_for(bearing, speed)[:, value_index].ravel()
        return ax.scatter(coordinates[:, 0], coordinates[:, 1], c=colors, **kwargs)
    

    def get_bearings(self) -> NDArray[Any]:
        return self.data["bearing"].to_numpy()
    
    def get_speeds(self) -> NDArray[Any]:
        return self.data["speed"].to_numpy()


class CartesianDeltaCollection(DeltaCollection, CartesianCollection):
    def __init__(self, origin: PolarLocation, number_of_values: int = 1):
        super().__init__(CartesianDeltaCollection._columns(number_of_values), origin, number_of_values)

    @staticmethod
    def _columns(number_of_values: int) -> list[str]:
        return ["east", "north", "bearing", "speed"] + [f"v{i}" for i in range(number_of_values)]

    @property
    def dimensions(self) -> tuple[float, float]:
        """Get the dimensions of this Collection in meters.

        Returns:
            The dimensions of this Collection in meters as ``(width, height)``.
        """

        west, east, south, north = self.extent()

        return east - west, north - south

    def to_cartesian_locations(self) -> tuple[list[CartesianLocation], list[float], list[float]]:
        """Returns the cartesian locations, as well as bearings and speed

        Returns:
            list[CartesianLocation]: A list of the cartesian locations
            list[float]: A list of corresponding bearings
            list[float]: A list of corresponding speeds
        """
        coordinates = self.coordinates()

        locations = []
        for i in range(coordinates.shape[0]):
            locations.append(CartesianLocation(east=coordinates[i, 0], north=coordinates[i, 1]))

        return locations

    def to_polar(self) -> "PolarDeltaCollection":
        # Apply the inverse projection of the origin location
        longitudes, latitudes = self.origin.projection(
            self.data["east"].to_numpy(), self.data["north"].to_numpy(), inverse=True
        )

        # Create the new collection in polar coordinates
        polar_collection = PolarDeltaCollection(self.origin, self.number_of_values)
        polar_collection.data["longitude"] = longitudes
        polar_collection.data["latitude"] = latitudes
        polar_collection.data["bearing"] = self.data["bearing"]
        polar_collection.data["speed"] = self.data["speed"]

        # Copy over the values
        for i in range(self.number_of_values):
            polar_collection.data[f"v{i}"] = self.data[f"v{i}"]

        return polar_collection

    def get_interpolator(self, method: str = "linear") -> Any:
        """Get an interpolator for the data. Assumes a grid structure of the data.

        Args:
            method: The interpolation method to use

        Returns:
            A callable interpolator function
        """

        print(self.data["v0"].unique())
        # TODO link to grid class / remove assumption
        grid_axes = [self.data[self.data.columns[i]].unique() for i in range(4)]
        return RegularGridInterpolator(
            grid_axes,
            self.data[self.data.columns[4:]].to_numpy().reshape(
                [len(grid_axes[i]) for i in range(len(grid_axes))] + [self.number_of_values]
            ),
            method=method,
        )


class PolarDeltaCollection(DeltaCollection, PolarCollection):
    def __init__(self, origin: PolarLocation, number_of_values: int = 1):
        super().__init__(PolarDeltaCollection._columns(number_of_values), origin, number_of_values)

    @staticmethod
    def _columns(number_of_values: int) -> list[str]:
        return ["longitude", "latitude", "bearing", "speed"] + [f"v{i}" for i in range(number_of_values)]

    @property
    def dimensions(self) -> tuple[float, float]:
        """Get the dimensions of this Collection in meters.

        Returns:
            The dimensions of this Collection in meters as ``(width, height)``.
        """

        return self.to_cartesian().dimensions

    def to_polar_locations(self) -> tuple[list[PolarLocation], list[float], list[float]]:
        """Returns the cartesian locations, as well as bearings and speed

        Returns:
            list[CartesianLocation]: A list of the polar locations
            list[float]: A list of corresponding bearings
            list[float]: A list of corresponding speeds
        """
        coordinates = self.coordinates()

        locations = []
        for i in range(coordinates.shape[0]):
            locations.append(PolarLocation(longitude=coordinates[i, 0], latitude=coordinates[i, 1]))

        return locations, coordinates[i, 2], coordinates[i, 3]

    def to_cartesian(self) -> CartesianDeltaCollection:
        # Apply the inverse projection of the origin location
        easts, norths = self.origin.projection(
            self.data["longitude"].to_numpy(), self.data["latitude"].to_numpy()
        )

        # Create the new collection in polar coordinates
        cartesian_collection = CartesianDeltaCollection(self.origin, self.number_of_values)
        cartesian_collection.data["east"] = easts
        cartesian_collection.data["north"] = norths
        cartesian_collection.data["bearing"] = self.data["bearing"]
        cartesian_collection.data["speed"] = self.data["speed"]

        # Copy over the values
        for i in range(self.number_of_values):
            cartesian_collection.data[f"v{i}"] = self.data[f"v{i}"]

        return cartesian_collection
