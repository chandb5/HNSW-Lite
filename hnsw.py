import math
import numpy as np
from typing import List, Dict

from node import Node
from search import Search


class HNSW:
    def __init__(self, space: str = "cosine", M: int = 16, ef_construction: int = 200):
        self.space = space
        self.distance_cache = {}
        self.search_obj = Search(self.distance_cache)
        self.entry_point = None
        self.bottom_layer = 0
        self.top_layer = 0
        self.layered_graph: Dict[int, List[Node]] = {}

        # Setting hyperparameters
        # desired number of connections per node
        self.m = M

        # max number of connections per node for layer > 0
        self.mMax = self.m

        # max number of connections per node for layer 0 - typically 2*M
        self.m0 = 2 * self.m

        # normalizing factor for number of layers in the graph
        self.mL = 1 / np.log(self.m)

        self.ef_construction = ef_construction

    def insert(self, query: List[float], metadata: Dict = {}):
        """
        Insert a new vector into the graph.
        :param query: The vector to be inserted.
        :param metadata: Metadata associated with the vector.
        :return: None
        """

        level = self.__get_random_level()
        query_node = Node(query, level, metadata)

        if self.entry_point is None:
            self.entry_point = query_node
            self.top_layer = level

        entry_point = self.entry_point
        bottom_layer = 0
        top_layer = entry_point.level if entry_point else self.top_layer

        # Upper-layer search (zoom-out phase)
        # This loop only runs if level < top_layer.
        for layer in range(top_layer, level, -1):
            # Perform greedy search in current layer, update entry_point
            best_candidate = self.search_obj.search_layer(
                query_node, entry_point, self.ef_construction, layer
            )[0][1]
            entry_point = best_candidate

        # Lower-layer search and insertion (zoom-in phase)
        for layer in range(min(top_layer, level), bottom_layer - 1, -1):
            W_dst = self.search_obj.search_layer(
                query_node, entry_point, self.ef_construction, layer
            )
            W = [node for _, node in W_dst]
            neighbours = self.search_obj.search_neighbours_simple(query_node, W, self.m)

            for node in neighbours:
                if node not in query_node.neighbors[layer]:
                    query_node.neighbors[layer].append(node)
                if query_node not in node.neighbors[layer]:
                    node.neighbors[layer].append(query_node)

            for e in neighbours:
                eConn = e.neighbors[layer]
                if len(eConn) > self.mMax:
                    local_mMax = self.mMax if layer > 0 else self.m0
                    eNewConn = self.search_obj.search_neighbours_simple(
                        e, eConn, local_mMax
                    )
                    e.neighbors[layer] = eNewConn

            # Update entry point using best candidate from current candidate list W
            if W:
                entry_point = W[0]

        # Update the layered graph
        for l in range(level + 1):
            if l not in self.layered_graph:
                self.layered_graph[l] = []
            self.layered_graph[l].append(query_node)

        # Update the overall entry point and top layer if the new node has higher level
        if level > self.top_layer:
            self.top_layer = level
            self.entry_point = query_node

    def knn_search(self, query: Node, top_n: int):
        """
        Search for the k-nearest neighbors of the query vector.
        :param query: The query vector.
        :param top_n: The number of nearest neighbors to return.
        :return: The top n nearest neighbors.
        """
        entry_point = self.entry_point
        top_layer = self.top_layer

        for layer in range(top_layer, -1, -1):
            best_candidate = self.search_obj.search_layer(query, entry_point, 1, layer)[
                0
            ][1]
            entry_point = best_candidate

        neighbours = self.search_obj.search_layer(query, entry_point, top_n, 0)
        return neighbours

    def __get_random_level(self):
        """
        Get a random level for the new node.
        :return: The random level.
        """
        return math.floor(-math.log(np.random.uniform(0, 1)) * self.mL)
