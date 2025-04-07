"""
Unit tests for the Search class.
"""

import unittest
import numpy as np

from hnsw.node import Node
from hnsw.search import Search


class TestSearch(unittest.TestCase):
    """Test cases for the Search class."""

    def setUp(self):
        """Set up test fixtures."""
        self.distance_cache = {}
        self.search = Search(self.distance_cache)

        self.query_node = Node([0.0, 0.0], 0)
        self.node1 = Node([1.0, 0.0], 1)
        self.node2 = Node([0.0, 1.0], 1)
        self.node3 = Node([0.5, 0.5], 1)
        self.node4 = Node([2.0, 2.0], 1)
        self.node5 = Node([3.0, 0.0], 1)

        self.node1.neighbors = {0: [self.node2, self.node3], 1: [self.node2]}
        self.node2.neighbors = {
            0: [self.node1, self.node3, self.node4],
            1: [self.node1],
        }
        self.node3.neighbors = {0: [self.node1, self.node2], 1: []}
        self.node4.neighbors = {0: [self.node2, self.node5], 1: []}
        self.node5.neighbors = {0: [self.node4], 1: []}

    def test_search_neighbours_simple(self):
        """Test the simple neighbors search functionality."""
        candidates = [self.node1, self.node2, self.node3, self.node4, self.node5]

        nearest = self.search.search_neighbours_simple(self.query_node, candidates, 3)

        self.assertEqual(len(nearest), 3)
        self.assertIn(self.node1, nearest)
        self.assertIn(self.node2, nearest)
        self.assertIn(self.node3, nearest)

    def test_search_layer_single_node(self):
        """Test the layer search with a single entry point."""
        top_n = 1
        level = 0
        results = self.search.search_layer(self.query_node, self.node1, top_n, level)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], self.node1)

    def test_search_layer_exploration(self):
        """Test the layer search with graph exploration."""
        top_n = 3
        level = 0
        results = self.search.search_layer(self.query_node, self.node1, top_n, level)

        self.assertEqual(len(results), 3)
        result_nodes = [node for _, node in results]

        self.assertIn(self.node3, result_nodes)
        self.assertNotIn(self.node5, result_nodes)

    def test_search_layer_with_deleted_nodes(self):
        """Test the layer search with deleted nodes in the graph."""
        self.node3.is_deleted = True

        top_n = 3
        level = 0
        results = self.search.search_layer(self.query_node, self.node1, top_n, level)

        result_nodes = [node for _, node in results]
        self.assertNotIn(self.node3, result_nodes)
        self.node3.is_deleted = False


if __name__ == "__main__":
    unittest.main()
