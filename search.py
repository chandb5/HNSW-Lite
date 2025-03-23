from hnsw.distance import Distance
from node import Node

class Search:
    def __init__(self):
        self.distance_obj = Distance("cosine")

    def search_neighbours_simple(self, query, candidates, top_n):
        """
        Search for the nearest neighbors of the query vector, from a list of candidate vectors.
        :param query: The query vector.
        :param candidates: The candidate vectors to search from.
        :param top_n: The number of nearest neighbors to return.
        :return: The top n nearest neighbors.
        """
        distances = []
        for c in candidates:
            dst = self.distance_obj.distance(query, c)
            distances.append((c, dst))
        
        distances.sort(key=lambda x: x[1])
        nearest_neighbors = distances[:top_n]
        return nearest_neighbors
    
    def search_layer(self, query, entry_point: Node, top_n: int):
        """
        Search for the nearest neighbors of the query vector in a specific layer.
        :param query: The query vector.
        :param node: The node containing the candidate vectors.
        :param top_n: The number of nearest neighbors to return.
        :return: The top n nearest neighbors.
        """
        visited = set()
        visited.add(entry_point)
        candidates = [entry_point]
        neighbours = [entry_point]
        return self.search_neighbours_simple(query, candidates, top_n)