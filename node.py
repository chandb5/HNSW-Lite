import numpy as np

from typing import List, Dict
from .distance import Distance


class Node:
    _global_id = 0

    def __init__(self, vector: List[float], level: int, metadata=None, neighbours=None):
        self.id = Node._global_id
        Node._global_id += 1

        self.vector = np.array(vector)
        self.level = level
        self.metadata = metadata
        self.neighbors: Dict[int, List["Node"]] = {i: [] for i in range(level + 1)}
        self.is_deleted = False
        # we use magnitude just as comparator in case of ties
        self.magnitude = np.linalg.norm(self.vector)

    def distance(self, query: "Node", space: str = "cosine") -> float:
        distance_obj = Distance(space)
        return distance_obj.distance(self, query)

    def __lt__(self, other: "Node"):
        return self.magnitude < other.magnitude

    def __gt__(self, other: "Node"):
        return self.magnitude > other.magnitude

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id
