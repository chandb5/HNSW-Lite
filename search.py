from .distance import Distance
from .node import Node
from typing import List, Dict
import heapq


class Search:
    def __init__(self, distance_cache: Dict):
        self.distance_cache = distance_cache
        self.distance_obj = Distance("cosine")

    def search_neighbours_simple(
        self, query: Node, candidates: List[Node], top_n: int
    ) -> List[Node]:
        """
        Search for the nearest neighbors of the query vector, from a list of candidate vectors.
        :param query: The query vector.
        :param candidates: The candidate vectors to search from.
        :param top_n: The number of nearest neighbors to return.
        :return: The top n nearest neighbors. Tuple of (distance, node).
        """
        distances = []
        for c in candidates:
            dst = self.distance_obj.distance(query, c)
            distances.append((c, dst))

        distances.sort(key=lambda x: x[1])
        nearest_neighbors = distances[:top_n]

        neighbors = [node for node, _ in nearest_neighbors]
        return neighbors

    def search_layer(
        self, query_node: Node, entry_point: Node, top_n: int, level: int
    ) -> List[tuple[int, Node]]:
        """
        Search for the nearest neighbors of the query vector in a specific layer.
        :param query_node: The query vector.
        :param entry_point: The entry point for the search.
        :param top_n: The number of nearest neighbors to return.
        :param level: The layer to search in.
        :return: The top n nearest neighbors. Tuple of (distance, node).
        """
        visited = set()
        visited.add(entry_point)
        candidates = []
        entry_point_query_distance = self.distance_obj.distance(entry_point, query_node)
        # min heap to store the closest candidates
        heapq.heappush(candidates, (entry_point_query_distance, entry_point))

        # max heap to store the best neighbours
        best_neighbours = []
        heapq.heappush(best_neighbours, (-entry_point_query_distance, entry_point))

        while len(candidates) > 0:
            # closest candidate to the query vector
            distance, current = heapq.heappop(candidates)

            # get farthest possible neighbour
            farthest_best_neighbour = (
                -best_neighbours[0][0] if len(best_neighbours) > 0 else float("inf")
            )

            if distance > farthest_best_neighbour:
                break  # all elements in candidates are farther than the best neighbour

            for node in current.neighbors[level]:
                if node not in visited and not node.is_deleted:
                    visited.add(node)
                    furthest_distance = (
                        -best_neighbours[0][0]
                        if len(best_neighbours) > 0
                        else float("inf")
                    )

                    node_query_distance = self.distance_obj.distance(node, query_node)

                    if (
                        node_query_distance < furthest_distance
                        or len(best_neighbours) < top_n
                    ):
                        heapq.heappush(candidates, (node_query_distance, node))
                        if len(best_neighbours) + 1 > top_n:
                            heapq.heappushpop(
                                best_neighbours, (-node_query_distance, node)
                            )
                        else:
                            heapq.heappush(
                                best_neighbours, (-node_query_distance, node)
                            )

        return heapq.nlargest(top_n, best_neighbours)
