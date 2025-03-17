from typing import List

class Node:
    def __init__(self, vector: List[float], level: int, neighbours = None):
        self.vector = vector
        self.level = level
        self.neighbors = neighbours if neighbours else []
