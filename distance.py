import numpy as np
from typing import List


class Distance:
    """
    A class to calculate the distance between two vectors.
    """

    def __init__(self, space: str = "cosine"):
        """
        Initialize the distance object.
        :param space: The space in which the distance will be calculated.
        """
        self.space = space
        if self.space not in ["cosine", "euclidean"]:
            raise ValueError(
                "Invalid distance space. Please use 'cosine' or 'euclidean'."
            )

    def distance(self, nodeA, nodeB) -> float:
        """
        Calculate the distance between two vectors.
        :param nodeA: The first vector.
        :param nodeB: The second vector.
        :return: The distance between the two vectors.
        """
        if self.space == "cosine":
            return self.__cosine_distance(nodeA, nodeB)
        elif self.space == "euclidean":
            return self.__euclidean_distance(nodeA.vector, nodeB.vector)

    def __cosine_distance(self, nodeA, nodeB) -> float:
        """
        Calculate the cosine distance between two vectors.
        Represented as: 1 - cosine_similarity
        :param nodeA: The first vector.
        :param nodeB: The second vector.
        :return: The cosine distance between the two vectors.
        """
        if nodeA.magnitude == 0 or nodeB.magnitude == 0:
            return 1.0
        cosine_similarity = np.dot(nodeA.vector, nodeB.vector) / (
            nodeA.magnitude * nodeB.magnitude
        )
        cosine_distance = 1 - cosine_similarity
        return cosine_distance

    def __euclidean_distance(self, vector_A: np.array, vector_B: np.array) -> float:
        """
        Calculate the euclidean distance between two vectors.
        :param vector_A: The first vector.
        :param vector_B: The second vector.
        :return: The euclidean distance between the two vectors.
        """
        return np.linalg.norm(vector_A - vector_B)
