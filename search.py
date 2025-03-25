from .distance import Distance
from .node import Node
from typing import List
from numpy.typing import NDArray
import heapq
import numpy as np


class Search:
    def __init__(self):
        self.distance_obj = Distance("cosine")

    def search_neighbours_simple(self, query: NDArray[np.float64], candidates: List[Node], top_n: int) -> List[Node]:
        """
        Search for the nearest neighbors of the query vector, from a list of candidate vectors.
        :param query: The query vector.
        :param candidates: The candidate vectors to search from.
        :param top_n: The number of nearest neighbors to return.
        :return: The top n nearest neighbors.
        """
        distances = []
        for c in candidates:
            dst = self.distance_obj.distance(query, c.vector)
            distances.append((c, dst))
        
        distances.sort(key=lambda x: x[1])
        nearest_neighbors = distances[:top_n]

        neighbors = [node for node, _ in nearest_neighbors]
        return neighbors
    
    def search_layer(self, query: NDArray[np.float64], entry_point: Node, top_n: int, level: int) -> List[tuple[int, Node]]:
        """
        Search for the nearest neighbors of the query vector in a specific layer.
        :param query: The query vector.
        :param node: The node containing the candidate vectors.
        :param top_n: The number of nearest neighbors to return.
        :return: The top n nearest neighbors.
        """
        visited = set()
        visited.add(entry_point)
        candidates = []
        # min heap to store the closest candidates
        heapq.heappush(candidates, (entry_point.distance(query), entry_point))
        
        # max heap to store the best neighbours
        best_neighbours = []
        heapq.heappush(best_neighbours, (-entry_point.distance(query), entry_point))

        while len(candidates) > 0:
            # closest candidate to the query vector
            distance, current = heapq.heappop(candidates)

            # get farthest possible neighbour
            farthest_best_neighbour = -best_neighbours[0][0] if len(best_neighbours) > 0 else float('inf')

            if distance > farthest_best_neighbour:
                break # all elements in candidates are farther than the best neighbour

            for node in current.neighbors[level]:
                if node not in visited and not node.is_deleted:
                    visited.add(node)
                    furthest_distance = -best_neighbours[0][0] if len(best_neighbours) > 0 else float('inf')
                    
                    if node.distance(query) < furthest_distance or len(best_neighbours) < top_n:
                        heapq.heappush(candidates, (node.distance(query), node))
                        heapq.heappush(best_neighbours, (-node.distance(query), node))
                    
                        if len(best_neighbours) > top_n:
                            heapq.heappop(best_neighbours)

        return heapq.nlargest(top_n, best_neighbours)