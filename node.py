from typing import List, Union
from distance import Distance

class Node:
    def __init__(self, vector: List[float], level: int, neighbours = None):
        self.vector = vector
        self.level = level
        self.neighbors = neighbours if neighbours else []
        self.is_deleted = False

    def get_nearest_elements(self, query: List[float], candidates: List['Node'], top_n: int):
        distance_obj = Distance("cosine")
        distances = []
        for c in candidates:
            dst = distance_obj.distance(query, c)
            distances.append((c, dst))
        
        distances.sort(key=lambda x: x[1])
        nearest_neighbors = distances[:top_n]
        return nearest_neighbors

    def distance(self, query: List[float] | 'Node') -> float:
        distance_obj = Distance("cosine")
        if isinstance(query, Node):
            return distance_obj.distance(query.vector, self.vector)
        return distance_obj.distance(query, self.vector)