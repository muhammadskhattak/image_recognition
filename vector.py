""" Muhammad Khattak
    2018-04-12
    Version 1.0
"""
from typing import List, Union
from math import acos

class DimensionError(Exception):
    """ An error raised when vectors of impatible dimensions are compared."""
    pass

class Vector:
    values: Tuple[List[Union[float, int]]]
    dimension: int

    """ A mathematical vector"""
    def __init__(self, *values: List[Union[int, float]]) -> None:
        """ Initialize a new vector with the desired values.

        Precondition: values is only a list of values or multiple arguments but
            not both

        >>> vec = Vector(1, 2, 3, 4, 5)
        >>> vec.values
        (1, 2, 3, 4, 5)
        >>> vec.dimension
        5
        >>> vec2 = Vector([1, 2, 3, 4, 5])
        >>> vec2.values
        (1, 2, 3, 4, 5)
        """
        if isinstance(values[0], list):
            self.values = tuple(values[0])
            self.dimension = len(values[0])
        else:
            self.values = values
            self.dimension = len(values)

    def __str__(self) -> str:
        """ Return a string representation of this vector"""
        return 'Vector: {}'.format(self.values)

    def __repr__(self) -> str:
        """ Return a string representation of this vector."""
        return  'Vector: {}'.format(self.values)

    def __add__(self, other: "Vector") -> "Vector":
        """ Return the sum of this vector and other vector.

        >>> vec = Vector(1, 2, 3, 4, 5)
        >>> vec2 = Vector(1, 2, 3, 4, 5)
        >>> vec + vec2
        Vector: (2, 4, 6, 8, 10)
        """
        if self.dimension != other.dimension:
            raise DimensionError
        val = []
        for x in range(self.dimension):
            val.append(self.values[x] + other.values[x])
        return Vector(val)

    def __sub__(self, other: "Vector") -> "Vector":
        """ Return the difference of this vector and other vector.

        >>> vec = Vector(1, 2, 3, 4, 5)
        >>> vec2 = Vector(1, 2, 3, 4, 5)
        >>> vec + vec2
        Vector: (0, 0, 0, 0, 0)
        """
        if self.dimension != other.dimension:
            raise DimensionError
        val = []
        for x in range(self.dimension):
            val.append(self.values[x] - other.values[x])
        return Vector(val)

    def __mul__(self, other: Union[int, float, "Vector"]) -> Union[int, float, "Vector"]:
        """ Return the dot product of this vector and other if other is a vector
        otherwise return a vector scaled.

        >>> vec = Vector(1, 2, 3)
        >>> vec2 = Vector(1, 2, 3)
        >>> vec1 * vec2
        14
        """
        if isinstance(other, Vector) and self.dimension != other.dimension:
            raise DimensionError
        elif isinstance(other, Vector):
            total = sum([self.values[i] * other.values[i] for i in range(self.dimension)])
        else:
            total = []
            for x in range(self.dimension):
                total.append(other * self.values[x])
        return Vector(total) if isinstance(total, list) else total

    def __rmul__(self, other: Union[int, float, "Vector"]) -> Union[int, float, "Vector"]:
        """ Return the dot product of this vector and other if other is a vector
        otherwise return a vector scaled.

        >>> vec = Vector(1, 2, 3)
        >>> vec2 = Vector(1, 2, 3)
        >>> vec1 * vec2
        14
        """
        if isinstance(other, Vector) and self.dimension != other.dimension:
            raise DimensionError
        elif isinstance(other, Vector):
            total = sum([self.values[i] * other.values[i] for i in range(self.dimension)])
        else:
            total = []
            for x in range(self.dimension):
                total.append(other * self.values[x])
        return Vector(total) if isinstance(total, list) else total

    def magnitiude(self) -> float:
        """ Return the magnitude of this vector.

        >>> vec = Vector(3, 4)
        >>> magnitiude(vec)
        5.0
        """
        return (self * self) ** 0.5

    def angle_between(self, other: "Vector") -> float:
        """ Return the angle between this vector and other vector.

        >>> vec = Vector(1, 0)
        >>> vec2 = Vector(0, 1)
        >>> vec.angle_between(vec2)
        1.5707963267948966
        """
        return acos(self * other / (self.magnitiude() * other.magnitiude()))

    def unit_vector(self) -> "Vector":
        """ Return a unit vector in the same direction as this vector.

        >>> vec = Vector(1, 2)
        >>> vec.unit_vector
        Vector: (0.4472135954999579, 0.8944271909999159)
        """
        return (1/self.magnitiude()) * self

    def project(self, other: "Vector") -> "Vector":
        """ Project this vector on other vector.

        >>> vec = Vector(1, 0)
        >>> vec2 = Vector(0, 1)
        >>> vec.project(vec2)
        Vector: (0.0, 0.0)
        """
        return (self * other / other.magnitiude()) * other
