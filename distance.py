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

    def distance(self, vector_A: List[float], vector_B: List[float]) -> float:
        """
        Calculate the distance between two vectors.
        :param vector_A: The first vector.
        :param vector_B: The second vector.
        :return: The distance between the two vectors.
        """
        if self.space == "cosine":
            return self.__private_cosine_distance(vector_A, vector_B)
        elif self.space == "euclidean": 
            return self.__private_euclidean_distance(vector_A, vector_B)
        else:
            raise ValueError("Invalid distance space. Please use 'cosine' or 'euclidean'.")

    def __private_cosine_distance(vector_A, vector_B):
        """
        Calculate the cosine distance between two vectors.
        Represented as: 1 - cosine_similarity
        :param vector_A: The first vector.
        :param vector_B: The second vector.
        :return: The cosine distance between the two vectors.
        """
        cosine_similarity = np.dot(vector_A, vector_B) / (np.linalg.norm(vector_A) * np.linalg.norm(vector_B))
        cosine_distance = 1 - cosine_similarity
        return cosine_distance

    def __private_euclidean_distance(vector_A, vector_B):
        """
        Calculate the euclidean distance between two vectors.
        :param vector_A: The first vector.
        :param vector_B: The second vector.
        :return: The euclidean distance between the two vectors.
        """
        return np.linalg.norm(vector_A - vector_B)