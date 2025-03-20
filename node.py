from typing import List
from distance import Distance

class Node:
    def __init__(self, vector: List[float], level: int, neighbours = None):
        self.vector = vector
        self.level = level
        self.neighbors = neighbours if neighbours else []
        self.is_deleted = False

    def distance(self, query: List[float]):
        distance_obj = Distance("cosine")
        if isinstance(query, Node):
            return distance_obj.distance(query.vector, self.vector)
        return distance_obj.distance(query, self.vector)