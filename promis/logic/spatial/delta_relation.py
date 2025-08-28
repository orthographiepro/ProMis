# Standard Library
from abc import ABC, abstractmethod
from pathlib import Path
from pickle import dump, load
from typing import TypeVar
from warnings import warn

# Third Party
from numpy import ndarray, array, clip, mean, sqrt, var, vstack
from scipy.stats import norm
from shapely.strtree import STRtree

# ProMis
from promis.geo import CartesianDeltaCollection, CartesianLocation, CartesianMap, CartesianRasterBand

#: Helper to define derived relations within base class
DerivedRelation = TypeVar("DerivedRelation", bound="DeltaRelation")


class DeltaRelation(ABC):
    """A spatial relation between points in space, bearing and speed and typed map features.

    Args:
        parameters: A CartesianCollection relating points with parameters
            of the relation's distribution, e.g., mean and variance
        location_type: The type of location this relates to, e.g., buildings or roads
    """

    def __init__(self, parameters: CartesianDeltaCollection, location_type: str) -> None:
        # Setup attributes
        self.parameters = parameters
        self.location_type = location_type

    @staticmethod
    def load(path: str) -> DerivedRelation:
        """Load the relation from a .pkl file.

        Args:
            path: The path to the file including its name and file extension

        Returns:
            The loaded Relation instance
        """

        with open(path, "rb") as file:
            return load(file)

    def save(self, path: Path):
        """Save the relation to a .pkl file.

        Args:
            path: The path to the file including its name and file extension
        """

        with open(path, "wb") as file:
            dump(self, file)

    def save_as_plp(self, path: Path):
        """Save the relation as a text file containing distributional clauses.

        Args:
            path: The path to the file including its name and file extension
        """

        with open(path, "w") as plp_file:
            plp_file.write(self.to_distributional_clauses().join(""))

    def to_distributional_clauses(self) -> list[str]:
        """Express the Relation as distributional clause.

        Returns:
            The distributional clauses representing according to this Relation
        """

        return [
            self.index_to_distributional_clause(index) for index in range(len(self.parameters.data))
        ]

    @staticmethod
    @abstractmethod
    def empty_map_parameters() -> list[float]:
        """Create the default parameters for an empty map."""

    @abstractmethod
    def index_to_distributional_clause(self, index: int) -> str:
        """Express a single index of this Relation as a distributional clause.

        Returns:
            The distributional clause representing the respective entry of this Relation
        """

    @staticmethod
    @abstractmethod
    def compute_relation(
        location: CartesianLocation, bearing: float, speed: float, r_tree: STRtree, original_geometries: CartesianMap
    ) -> float:
        """Compute the value of this Relation type for a specific location and map.

        Args:
            location: The location to evaluate in Cartesian coordinates
            r_tree: The map represented as r-tree
            original_geometries: The geometries indexed by the STRtree

        Returns:
            The value of this Relation for the given location and map
        """

    @staticmethod
    @abstractmethod
    def arity() -> int:
        """Return the arity of the relation."""

    @classmethod
    def compute_parameters(
        cls,
        location: CartesianLocation,
        bearing: float,
        speed: float,
        r_trees: list[STRtree],
        original_geometries: list[CartesianMap],
    ) -> ndarray:
        """Compute the parameters of this Relation type for a specific location and set of maps.

        Args:
            location: The location to evaluate in Cartesian coordinates
            r_trees: The set of generated maps represented as r-tree
            original_geometries: The geometries indexed by the STRtrees

        Returns:
            The parameters of this Relation for the given location and maps
        """

        relation_data = [
            cls.compute_relation(location, bearing, speed, r_tree, geometries)
            for r_tree, geometries in zip(r_trees, original_geometries)
        ]

        return array([mean(relation_data, axis=0), var(relation_data, axis=0)]).T

    @classmethod
    def from_r_trees(
        cls,
        support: CartesianDeltaCollection,
        r_trees: list[STRtree],
        location_type: str,
        original_geometries: list[CartesianMap],
    ) -> DerivedRelation:
        """Compute relation for a Cartesian collection of points and a set of R-trees.

        Args:
            support: The collection of Cartesian points to compute Over for
            r_trees: Random variations of the features of a map indexible by an STRtree each
            location_type: The type of features this relates to
            original_geometries: The geometries indexed by the STRtrees

        Returns:
            The computed relation
        """

        # TODO if `support` is a RasterBand, we could make parameters a RasterBand as well
        # to maintain the efficient raster representation

        

        # Compute Over over support points
        locations, bearings, speeds = support.to_cartesian_locations()
        statistical_moments = vstack(
            [
                cls.compute_parameters(location, bearing, speed, r_trees, original_geometries)
                for location, bearing, speed in zip(locations, bearings, speeds)
            ]
        )

        # Setup parameter collection and return relation
        parameters = CartesianDeltaCollection(
            support.origin, number_of_values=statistical_moments.shape[1]
        )
        parameters.append(locations, statistical_moments, bearings, speeds)

        return cls(parameters, location_type)
