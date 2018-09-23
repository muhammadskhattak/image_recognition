""" Muhammad Khattak
    2018-04-12
    Version 1.0
"""
from typing import List, Union, Tuple
from vector import Vector, DimensionError

class Matrix:
    """ A collection of vectors. """
    def __init__(self, columns: List[Vector]) -> None:
        """ Create a new matrix with the specified vectors. """
        if len(columns) == 0:
            self.dimension_row = 0
            self.dimension_col = 0
            self.columns = ((),)
            self.rows = ((),)

        else:
            self.dimension_row = len(columns[0].values)
            self.dimension_col = len(columns)
            self.columns = tuple(columns)
            self.rows = tuple([Vector([vec.values[i] for vec in columns]) for i in range(self.dimension_row)])

    def __str__(self) -> str:
        """ Return a str representation of this matrix. """
        return_str = ''
        for row in self.rows:
            return_str = return_str + row.__str__() + '\n'
        return return_str

    def __repr__(self) -> str:
        """ Return a string representation of this matrix. """
        return_str = ''
        for row in self.rows:
            return_str = return_str + row.__str__() + '\n'
        return return_str

    def __add__(self, other: "Matrix") -> "Matrix":
        """ Add the two matrices. """
        if self.dimension_row == other.dimension_row and \
                self.dimension_col == other.dimension_col:
            new_vectors = [a + b for a, b in zip(self.columns, other.columns)]
            sum_matrix = Matrix(new_vectors)
            return sum_matrix
        raise DimensionError("Matrices of incompatible sizes")

    def __sub__(self, other: "Matrix") -> "Matrix":
        """ Subtract the two matrices. """
        if self.dimension_row == other.dimension_row and \
                self.dimension_col == other.dimension_col:
            new_vectors = [a - b for a, b in zip(self.columns, other.columns)]
            sum_matrix = Matrix(new_vectors)
            return sum_matrix
        raise DimensionError("Matrices of incompatible sizes")

    def __mul__(self, other: "Matrix") -> "Matrix":
        """ Multiply the two matrices. """
        if isinstance(other, Matrix) and self.dimension_col == other.dimension_row:
            new_vectors = []
            for column in other.columns:
                val = []
                for row in self.rows:
                    val.append(row * column)
                new_vectors.append(Vector(val))
            new_matrix = Matrix(new_vectors)
            return Matrix(new_vectors)
        elif isinstance(other, Matrix):
            raise DimensionError("Incompatible multiplcation sizes.")
        else:
            new_matrix = Matrix([col * other for col in self.columns])
            return new_matrix

    def __rmul__(self, other: "Matrix") -> "Matrix":
        """ Multiply the two matrices. """
        if isinstance(other, Matrix) and self.dimension_row == other.dimension_col:
            new_vectors = []
            for column in other.columns:
                val = []
                for row in self.rows:
                    val.append(row * column)
                new_vectors.append(Vector(val))
            new_matrix = Matrix(new_vectors)
            return Matrix(new_vectors)
        elif isinstance(other, Matrix):
            raise DimensionError("Incompatible multiplcation sizes.")
        else:
            new_matrix = Matrix([col * other for col in self.columns])
            return new_matrix

    def transpose(self) -> "Matrix":
        """ Return the transpose of this matrix. """
        new_matrix = Matrix(self.rows)
        return new_matrix
