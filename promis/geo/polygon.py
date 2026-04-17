"""This module implements abstractions for geospatial, polygonal shapes in WGS84 and local cartesian
coordinates using shapely."""

#
# Copyright (c) Simon Kohaut, Honda Research Institute Europe GmbH, Felix Divo, and contributors
#
# This file is part of ProMis and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with ProMis.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# Standard Library
from math import degrees, radians
from typing import Any, TypeVar

# Third Party
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from numpy import array, asarray, isfinite, ndarray, vstack
from shapely.geometry import Polygon as ShapelyPolygon, LineString, MultiPolygon, Point
from shapely.ops import split

import numpy as np

# ProMis
from promis.geo.geospatial import Geospatial
from promis.geo.helpers import meters_to_radians, radians_to_meters, calculate_street_width
from promis.geo.location import CartesianLocation, PolarLocation
from promis.geo.polyline import CartesianPolyLine, PolarPolyLine, PolyLine
from promis.models import Gaussian

#: Helper to define <Polar|Cartesian>Location operatios within base class
DerivedPolygon = TypeVar("DerivedPolygon", bound="Polygon")
DerivedBufferedPolygon = TypeVar("DerivedPolygon", bound="BufferedPolygon")


class Polygon(Geospatial):
    def __init__(
        self,
        locations: list[PolarLocation | CartesianLocation],
        holes: list[list[PolarLocation | CartesianLocation]] | None = None,
        location_type: str | None = None,
        name: str | None = None,
        identifier: int | None = None,
        covariance: ndarray | None = None,
        tags: dict[str, str | None] | None = None, 
    ) -> None:
        # Assertions on the given locations
        assert len(locations) >= 3, "A polygon must contain at least three points!"

        # Setup attributes
        self.locations = locations
        self.holes = holes if holes else []

        # Closes the ring by connecting last and first position
        if locations[0].x != locations[-1].x or locations[0].y != locations[-1].y:
            self.locations.append(locations[0])
        for hole in self.holes:
            if hole[0].x != hole[-1].x or hole[0].y != hole[-1].y:
                hole.append(hole[0])

        # Setup Gaussian distribution for sampling the polygon
        self.covariance = covariance

        # Setup tags
        self.tags = tags if tags is not None else {}
        self.tags.setdefault("lanes", "1")
        self.tags.setdefault("maxspeed", "-1")
        self.tags.setdefault("oneway", "no")
        
        # Setup Geospatial
        super().__init__(location_type=location_type, name=name, identifier=identifier)

    @property
    def covariance(self) -> ndarray:
        return self.distribution.covariance if self.distribution is not None else None

    @covariance.setter
    def covariance(self, value: ndarray | None) -> None:
        self.distribution = Gaussian(vstack([0.0, 0.0]), value) if value is not None else None

    @property
    def __geo_interface__(self) -> dict[str, Any]:
        exterior = [(location.x, location.y) for location in self.locations]
        interiors = [[(location.x, location.y) for location in hole] for hole in self.holes]
        coordinates = [exterior] + interiors

        return {
            "type": "Polygon",
            "coordinates": coordinates,
        }

    def sample(self, number_of_samples: int = 1) -> list[DerivedPolygon]:
        """Sample a polar polygon given this polygon's uncertainty.

        Args:
            number_of_samples: How many samples to draw

        Returns:
            The set of sampled polygons, each with same name, identifier etc.
        """

        # Check if a distribution is given
        if self.distribution is None:
            return [
                type(self)(
                    self.locations,
                    self.holes,
                    self.location_type,
                    self.name,
                    self.identifier,
                    self.covariance,
                    tags=self.tags,
                )
            ] * number_of_samples

        # Gather all the sampled polygons
        sampled_polygons = []
        for sample in self.distribution.sample(number_of_samples).T:
            # Translate each border location by sampled translation
            sampled_locations = [location + vstack(sample) for location in self.locations]

            # Translate each internal border location by sampled translation
            sampled_holes = [
                [location + vstack(sample) for location in hole] for hole in self.holes
            ]

            sampled_polygons.append(
                type(self)(
                    sampled_locations,
                    sampled_holes,
                    self.location_type,
                    self.name,
                    self.identifier,
                    self.covariance,
                    tags=self.tags.copy(),
                )
            )
        
        
        return sampled_polygons


class PolarPolygon(Polygon):
    """A polygon based on WGS84 coordinates.

    An object with only a single point may be represented by
    a polygon with three times the same location.

    Examples:
        Lets first create the perimeter of the polygon
        as a list of PolarLocation objects.

        >>> locations = [PolarLocation(-1.0, 1.0), PolarLocation(1.0, 1.0),
        ...  PolarLocation(1.0, -1.0), PolarLocation(-1.0, -1.0)]

        Now, we can build a polygon from these locations like so:

        >>> polygon = PolarPolygon(locations)

        Given another list of locations, e.g.,

        >>> holes = [[PolarLocation(-0.5, 0.5), PolarLocation(0.5, 0.5),
        ...  PolarLocation(0.5, -0.5), PolarLocation(-0.5, -0.5)]]

        we can also build a polygon that contains holes:

        >>> polygon = PolarPolygon(locations, holes)

        Given a covariance of this polygons translation, random samples can be drawn.

        >>> from numpy import eye
        >>> polygon = PolarPolygon(locations, covariance=eye(2))
        >>> random_samples = polygon.sample(10)

        PolarPolygons can also be created from numpy data.

        >>> from numpy import array
        >>> locations = array([[-1.0, 1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0]])
        >>> polygon = PolarPolygon.from_numpy(locations)

        Finally, a PolarPolygon can be turned into a CartesianPolygon
        given a reference origin in polar coordinates.

        >>> origin = PolarLocation(0.0, 0.0)
        >>> cartesian_polygon = polygon.to_cartesian(origin)

    Args:
        locations: The points that make up this polygon; see :attr:`~.locations`
        holes: The points that make up holes in this polygon
        location_type: The type of this polygon
        name: An optional name of this polygon
        identifier: The polygon's optional unique identifier, in :math:`[0, 2**63)`
        covariance: An optional matrix representing the variance of this polygon's
            latitude and longitude respectively
    """

    def __init__(
        self,
        locations: list[PolarLocation],
        holes: list[list[PolarLocation]] | None = None,
        location_type: str | None = None,
        name: str | None = None,
        identifier: int | None = None,
        covariance: ndarray | None = None,
        tags: dict[str, str | None] | None = None,
    ) -> None:
        # Setup Polygon
        super().__init__(locations, holes, location_type, name, identifier, covariance, tags)

    @classmethod
    def from_numpy(cls, data: ndarray, *args, **kwargs) -> "PolarPolygon":
        """Create a polygon from a numpy representation.

        Args:
            data: An array with shape ``(2, number_of_locations)``
            args: Positional arguments to be passed to the new polygon
            kwargs: Keyword arguments to be passed to the new polygon

        Returns:
            The polygon created from the given coordinates and other parameters

        Raises:
            :class:`AssertionError`: If the shape or content of ``data`` is invalid

        See Also:
            :meth:`~to_numpy`
        """

        # Assert that data is a vertical stack of two finite values
        assert len(data.shape) == 2
        assert data.shape[0] == 2
        assert isfinite(data).all()

        # Return appropriate polygon type
        return cls(
            [PolarLocation(x, y) for (x, y) in data.T],
            *args,
            **kwargs,
        )

    def to_cartesian(self, origin: PolarLocation) -> "CartesianPolygon":
        """Projects the polygon to a Cartesian one according to a given global reference.

        Args:
            origin: The reference point by which to project onto the local tangent plane

        Returns:
            The cartesian representation of this polygon with the given reference point being set
        """

        return CartesianPolygon(
            [location.to_cartesian(origin) for location in self.locations],
            [[location.to_cartesian(origin) for location in hole] for hole in self.holes],
            location_type=self.location_type,
            name=self.name,
            identifier=self.identifier,
            covariance=radians_to_meters(
                array(
                    [radians(degree) for degree in self.distribution.covariance.reshape(4)]
                ).reshape(2, 2)
            )
            if self.distribution is not None
            else None,
            origin=origin,
            tags=self.tags,
        )

    def __repr__(self) -> str:
        locations = ", ".join(str(loc) for loc in self.locations)

        return f"PolarPolygon(locations=[{locations}]{self._repr_extras})"


class CartesianPolygon(Polygon):
    """A cartesian polygon based on local coordinates with an optional global reference.

    Examples:
        Lets first create the perimeter of the polygon as a list of CartesianLocation objects.

        >>> locations = [CartesianLocation(-1.0, 1.0), CartesianLocation(1.0, 1.0),
        ...  CartesianLocation(1.0, -1.0), CartesianLocation(-1.0, -1.0)]

        Now, we can build a polygon from these locations like so:

        >>> polygon = CartesianPolygon(locations)

        Given another list of locations, e.g.,

        >>> holes = [[CartesianLocation(-0.5, 0.5), CartesianLocation(0.5, 0.5),
        ...  CartesianLocation(0.5, -0.5), CartesianLocation(-0.5, -0.5)]]

        we can also build a polygon that contains holes:

        >>> polygon = CartesianPolygon(locations, holes)

        Given a covariance of this polygons translation, random samples can be drawn.

        >>> from numpy import eye
        >>> polygon = CartesianPolygon(locations, covariance=eye(2))
        >>> random_samples = polygon.sample(10)

        CartesianPolygons can also be created from numpy data.

        >>> from numpy import array
        >>> locations = array([[-1.0, 1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0]])
        >>> polygon = CartesianPolygon.from_numpy(locations)

        Finally, a CartesianPolygon can be turned into a PolarPolygon
        given a reference origin in polar coordinates.

        >>> origin = PolarLocation(49.873163174, 8.653830718)
        >>> polar_polygon = polygon.to_polar(origin)

    Args:
        locations: The list of locations that this shape consists of; see :attr:`~.locations`
        holes: The points that make up holes in this polygon
        location_type: The type of this polygon
        name: The name of this polygon
        identifier: The polygon's optional unique identifier, in :math:`[0, 2**63)`
        covariance: An optional matrix representing the variance of this polygon's
            latitude and longitude respectively
        origin: A reference that can be used to project this cartesian representation (back)
            into a polar one
    """

    # Shapely Polygon has some abstract methods we do not override here

    def __init__(
        self,
        locations: list[CartesianLocation],
        holes: list[list[CartesianLocation]] | None = None,
        location_type: str | None = None,
        name: str | None = None,
        identifier: int | None = None,
        covariance: ndarray | None = None,
        origin: PolarLocation | None = None,
        tags: dict[str, str | None] | None = None,
    ) -> None:
        # Setup attributes
        self.origin = origin

        if holes is not None:
            self.geometry = ShapelyPolygon(
                [location.geometry.coords[0] for location in locations],
                [[location.geometry.coords[0] for location in hole] for hole in holes],
            )
        else:
            self.geometry = ShapelyPolygon([location.geometry.coords[0] for location in locations])

        # Setup Polygon
        Polygon.__init__(self, locations, holes, location_type, name, identifier, covariance, tags)

    @classmethod
    def from_numpy(cls, data: ndarray, *args, **kwargs) -> "CartesianPolygon":
        """Create a polygon from a numpy representation.

        Args:
            data: An array with shape ``(2, number_of_locations)``
            args: Positional arguments to be passed to the new polygon
            kwargs: Keyword arguments to be passed to the new polygon

        Returns:
            The polygon created from the given coordinates and other parameters

        Raises:
            :class:`AssertionError`: If the shape or content of ``data`` is invalid

        See Also:
            :meth:`~to_numpy`
        """

        # Assert that data is a vertical stack of two finite values
        assert len(data.shape) == 2
        assert data.shape[0] == 2
        assert isfinite(data).all()

        # Transform the holes too, if given
        if args and isinstance(args[0], list):
            args[0] = [[CartesianLocation(x, y) for x, y in hole.T] for hole in args[0]]
        elif "holes" in kwargs:
            kwargs["holes"] = [
                [CartesianLocation(x, y) for x, y in hole.T] for hole in kwargs["holes"]
            ]

        # Return appropriate polygon type
        return cls(
            [CartesianLocation(x, y) for (x, y) in data.T],
            *args,
            **kwargs,
        )

    def to_polar(self, origin: PolarLocation | None = None) -> PolarPolygon:
        """Computes the polar representation of this shape.

        Args:
            origin: The global reference to be used for back-projection, must be set if and only if
                :attr:`~promis.geo.CartesianPolygon.origin` is ``None``

        Returns:
            The global, polar representation of this geometry
        """

        # Check that one origin point is given
        if origin is None:
            if self.origin is None:
                raise ValueError(
                    "You need to provide an explicit origin when the instance does not have one"
                )
            origin = self.origin
        elif self.origin is not None and origin is not self.origin:
            raise ValueError(
                "You provided an explicit origin while the instance already has a different one"
            )

        # Hole locations to polar
        holes = [[location.to_polar(origin) for location in hole] for hole in self.holes]

        # Return appropriate Polygon in polar coordinates
        return PolarPolygon(
            [location.to_polar(origin) for location in self.locations],
            holes,
            location_type=self.location_type,
            name=self.name,
            identifier=self.identifier,
            covariance=array(
                [degrees(rad) for rad in meters_to_radians(self.distribution.covariance).reshape(4)]
            ).reshape(2, 2)
            if self.distribution is not None
            else None,
            tags=self.tags,
        )

    def plot(self, axis, **kwargs) -> None:
        """Plots this polygon using Matplotlib.

        Args:
            axis: The axis object to use for plotting
            kwargs: Keyword arguments to pass to Matplotlib
        """

        path = Path.make_compound_path(
            Path(asarray(self.geometry.exterior.coords)[:, :2]),
            *[Path(asarray(ring.coords)[:, :2]) for ring in self.geometry.interiors],
        )

        patch = PathPatch(path, **kwargs)
        collection = PatchCollection([patch], **kwargs)

        axis.add_collection(collection, autolim=True)
        axis.autoscale_view()

    def distance(self, other: Any) -> float:
        return self.geometry.distance(other.geometry)

    def send_to_gui(self, ip: str, port: int, timeout: float):
        raise NotImplementedError("Cartesian Polygon does not have geospatial feature to send to gui!")

    def __repr__(self) -> str:
        origin = f", origin={self.origin}" if self.origin is not None else ""
        locations = ", ".join(f"({x}, {y})" for x, y in self.geometry.exterior.coords)

        return f"CartesianPolygon(locations=[{locations}]{origin}{self._repr_extras})"

    def __str__(self) -> str:
        # This is required to override shapely.geometry.Polygon.__str__()
        return self.__repr__()

    @classmethod
    def make_centered_box(
        cls,
        width: float,
        height: float,
        offset: CartesianLocation = CartesianLocation(0, 0),
        **kwargs,
    ) -> "CartesianPolygon":
        """Generates a box centered around a given offset.

        Args:
            width: The width of the map in meters
            height: The height of the map in meters
            kwargs: Additional keyword arguments to pass to the polygon, such as an origin

        Returns:
            The box as a polygon
        """

        return cls(
            [
                # clockwise: top-left, ...
                CartesianLocation(east=offset.east - width / 2, north=offset.north + height / 2),
                CartesianLocation(east=offset.east + width / 2, north=offset.north + height / 2),
                CartesianLocation(east=offset.east + width / 2, north=offset.north - height / 2),
                CartesianLocation(east=offset.east - width / 2, north=offset.north - height / 2),
            ],
            **kwargs,
        )
    
class BufferedPolygon(Polygon):
    def __init__(self, polyline: CartesianPolyLine | PolarPolyLine, locations, holes = None, location_type = None, name = None, identifier = None, covariance = None, tags = {}):
        super().__init__(locations, holes, location_type, name, identifier, covariance, tags)
        self.polyline = polyline

    @staticmethod
    def buffer_polyline(polyline: PolyLine) -> DerivedBufferedPolygon:
        """Turn a PolarPolyLine into a BufferedPolygon based on usual street widths. 
        Keep tags and reference to original PolyLine.

        Args:
            polyline (PolarPolyLine): The PolyLine you want to buffer

        Returns:
            PolarPolygon: The buffered PolarPolygon.
        """
        pass

    def sample(self, number_of_samples: int = 1) -> list[DerivedBufferedPolygon]:
        """Sample a BufferedPolygon given this polygon's uncertainty.
        As opposed to other Polygons, here the underlying PolyLine is sampled and then buffered again
        to create the sampled Polygons.

        Args:
            number_of_samples: How many samples to draw

        Returns:
            The set of sampled polygons, each with same name, identifier etc.
        """
        self.polyline.covariance = self.covariance
        self.polyline.distribution = self.distribution
        samples = [self.buffer_polyline(sampled_polyline) for sampled_polyline in self.polyline.sample(number_of_samples)]
        return samples
    
class BufferedPolarPolygon(BufferedPolygon, PolarPolygon):
    def __init__(self, polyline: PolarPolyLine, locations, holes = None, location_type = None, name = None, identifier = None, covariance = None, tags = None):
        BufferedPolygon.__init__(self, polyline, locations, holes, location_type, name, identifier, covariance, tags)
        PolarPolygon.__init__(self, locations, holes, location_type, name, identifier, covariance, tags)

    def __repr__(self) -> str:
        locations = ", ".join(str(loc) for loc in self.locations)
        return f"BufferedPolarPolygon(locations=[{locations}], polyline={repr(self.polyline)} {self._repr_extras})"

    def to_cartesian(self, origin):
        if origin is None:
            origin = self.locations[0]
        cartesian_pg = super().to_cartesian(origin)
        cartesian_pl = self.polyline.to_cartesian(origin)

        return BufferedCartesianPolygon(
            polyline=cartesian_pl,
            locations=cartesian_pg.locations,
            holes=cartesian_pg.holes,
            location_type=self.location_type,
            name=self.name,
            identifier=self.identifier,
            covariance=cartesian_pg.covariance,
            origin=origin,
            tags=self.tags,
        )
    
    def split_sides(self) -> list[PolarPolygon]:
        cartesian = self.to_cartesian(self.locations[0])
        return [cartesian_part.to_polar(self.locations[0]) for cartesian_part in cartesian.split_sides()]
    

    @staticmethod
    def buffer_polyline(polyline: PolarPolyLine) -> "BufferedPolarPolygon":
        """Turn a PolarPolyLine into a BufferedPolarPolygon based on usual street widths. 
        Keep tags and reference to original PolyLine, which is used for sampling.

        Args:
            polyline (PolarPolyLine): The PolyLine you want to buffer

        Returns:
            BufferedPolarPolygon: The buffered PolarPolygon.
        """
        width = calculate_street_width(polyline)
        location_type = polyline.location_type
        origin = polyline.locations[0]
        tags = polyline.tags | {"line": polyline}

        line = LineString([loc.to_cartesian(origin).to_numpy() for loc in polyline.locations])
        buffer = line.buffer(width / 2)
        exterior = buffer.geoms[0].exterior if isinstance(buffer, MultiPolygon) else buffer.exterior
        return BufferedPolarPolygon(
                polyline=polyline,
                locations=[
                    CartesianLocation(*point, location_type=location_type)
                    .to_polar(origin)
                    for point in exterior.coords
                ],
                location_type=location_type,
                name=polyline.name, 
                identifier=polyline.identifier, # TODO: do they need to be unique?
                covariance=polyline.covariance,
                tags=tags,
            )

class BufferedCartesianPolygon(BufferedPolygon, CartesianPolygon):
    def __init__(self, polyline: CartesianPolyLine, locations, holes = None, location_type = None, name = None, identifier = None, covariance = None, origin = None, tags = {}):
        BufferedPolygon.__init__(self, polyline, locations, holes, location_type, name, identifier, covariance, tags)
        CartesianPolygon.__init__(self, locations, holes, location_type, name, identifier, covariance, origin, tags)

    def __repr__(self) -> str:
        locations = ", ".join(str(loc) for loc in self.locations)
        return f"BufferedCartesianPolygon(locations=[{locations}], polyline={repr(self.polyline)} {self._repr_extras})"
    
    def to_polar(self, origin = None):
        polar_pg = super().to_polar(origin)
        polar_pl = self.polyline.to_polar(origin)

        return BufferedPolarPolygon(
            polyline=polar_pl,
            locations=polar_pg.locations,
            holes=polar_pg.holes,
            location_type=self.location_type,
            name=self.name,
            identifier=self.identifier,
            covariance=polar_pg.covariance,
            tags=self.tags,
        )

    def split_sides(self) -> list[CartesianPolygon]:
        last_diff: CartesianLocation = self.polyline.locations[-1] - self.polyline.locations[-2].to_numpy()
        last_diff = last_diff.to_numpy()
        last_diff *= 20 / self.polyline.locations[-1].distance(self.polyline.locations[-2])
        last = self.polyline.locations[-1] + last_diff

        first_diff:CartesianLocation = self.polyline.locations[0] - self.polyline.locations[1].to_numpy()
        first_diff = first_diff.to_numpy()
        first_diff *= 20 / self.polyline.locations[0].distance(self.polyline.locations[1])
        first = self.polyline.locations[0] + first_diff

        helper = CartesianPolyLine([first, *self.polyline.locations[1:-1] ,last])

        collection = split(self.geometry, helper.geometry)
        
        point = np.array(collection.geoms[0].exterior.xy).mean(axis=1)
        street_line = helper.geometry
        distances = [
            LineString([street_line.coords[i], street_line.coords[i+1]]).distance(Point(point))
            for i in range(len(street_line.coords) - 1)
        ]
        nearest_index = min(enumerate(distances), key=lambda x: x[1])[0]
        p1 = street_line.coords[nearest_index]
        p2 = street_line.coords[nearest_index + 1]
        cross_product = (p2[0] - p1[0]) * (point[1] - p1[1]) - \
            (p2[1] - p1[1]) * (point[0] - p1[0])
        
        left_first = cross_product > 0
        labels = ["left", "right"]
        labels = labels if left_first else labels[::-1]
        
        return [CartesianPolygon.from_numpy(
            np.stack(geo.exterior.xy), 
            location_type=f"{side}_{self.location_type}",
            name=self.name,
            identifier=self.identifier,
            covariance=self.covariance,
            tags=self.tags,
        ) for geo, side in zip(collection.geoms, labels)]

    @staticmethod
    def buffer_polyline(polyline: CartesianPolyLine) -> "BufferedPolarPolygon":
        """Turn a PolarPolyLine into a BufferedPolarPolygon based on usual street widths. 
        Keep tags and reference to original PolyLine, which is used for sampling.

        Args:
            polyline (PolarPolyLine): The PolyLine you want to buffer

        Returns:
            BufferedPolarPolygon: The buffered PolarPolygon.
        """
        width = calculate_street_width(polyline)
        location_type = polyline.location_type
        tags = polyline.tags | {"line": polyline.to_polar(polyline.origin)}
        
        line = LineString([loc.to_numpy() for loc in polyline.locations])
        buffer = line.buffer(width / 2)
        exterior = buffer.geoms[0].exterior if isinstance(buffer, MultiPolygon) else buffer.exterior
        return BufferedCartesianPolygon(
                polyline=polyline,
                locations=[
                    CartesianLocation(*point, location_type=location_type)
                    for point in exterior.coords
                ],
                location_type=location_type,
                origin=polyline.origin,
                tags=tags,
            )

