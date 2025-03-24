import numpy as np

from typing import List, Dict
from .distance import Distance

class Node:
    def __init__(self, vector: List[float], level: int, neighbours = None):
        self.vector = np.array(vector)
        self.level = level
        self.neighbors: Dict[int, List["Node"]] = {i: [] for i in range(level + 1)}
        self.is_deleted = False
        # we use magnitude just as comparator in case of ties
        self.magnitude = np.linalg.norm(self.vector)

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
        else:
            query = np.array(query)
            return distance_obj.distance(query, self.vector)
        
    def __lt__(self, other: 'Node'):
        return self.magnitude < other.magnitude
    
    def __gt__(self, other: 'Node'):
        return self.magnitude > other.magnitude
