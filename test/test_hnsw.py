"""
Unit tests for the HNSW class.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from hnsw.hnsw import HNSW
from hnsw.node import Node


class TestHNSW(unittest.TestCase):
    """Test cases for the HNSW class."""

    def setUp(self):
        """Set up test fixtures."""
        self.space = "cosine"
        self.M = 5
        self.ef_construction = 50
        self.hnsw = HNSW(self.space, self.M, self.ef_construction)

        self.vectors = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.8, 0.3],
            [-0.2, 0.5],
        ]

        # Insert test vectors
        for i, v in enumerate(self.vectors):
            self.hnsw.insert(v, {"id": i})

    def test_initialization(self):
        """Test HNSW initialization."""
        hnsw = HNSW(self.space, self.M, self.ef_construction)

        self.assertEqual(hnsw.space, self.space)
        self.assertEqual(hnsw.m, self.M)
        self.assertEqual(hnsw.mMax, self.M)
        self.assertEqual(hnsw.m0, 2 * self.M)
        self.assertEqual(hnsw.ef_construction, self.ef_construction)
        self.assertIsNone(hnsw.entry_point)
        self.assertEqual(hnsw.bottom_layer, 0)
        self.assertEqual(hnsw.top_layer, 0)
        self.assertEqual(hnsw.layered_graph, {})

    def test_insert_first_point(self):
        """Test inserting the first point into an empty HNSW index."""
        hnsw = HNSW(self.space, self.M, self.ef_construction)
        vector = [1.0, 2.0]
        metadata = {"id": 0}

        hnsw.insert(vector, metadata)

        # The entry point should be set
        self.assertIsNotNone(hnsw.entry_point)
        self.assertTrue(np.array_equal(hnsw.entry_point.vector, np.array(vector)))
        self.assertEqual(hnsw.entry_point.metadata, metadata)

        # The layered graph should contain the node at its level
        level = hnsw.entry_point.level
        self.assertIn(level, hnsw.layered_graph)
        self.assertIn(hnsw.entry_point, hnsw.layered_graph[level])

    def test_insert_multiple_points(self):
        """Test inserting multiple points into HNSW index."""
        all_nodes = self.hnsw.layered_graph[0]

        self.assertEqual(len(all_nodes), len(self.vectors))

        vectors_in_graph = [node.vector.tolist() for node in all_nodes]
        for v in self.vectors:
            self.assertIn(v, vectors_in_graph)

    def test_k_nearest_neighbors_search(self):
        """Test k-nearest neighbors search."""
        query_vector = [0.1, 0.1]
        query_node = Node(query_vector, 0)
        top_n = 3

        results = self.hnsw.knn_search(query_node, top_n)
        # Should return top_n results
        self.assertEqual(len(results), top_n)

        distances = [-dist for dist, _ in results]
        self.assertEqual(distances, sorted(distances))
        closest_distance = results[0][0]

        for v in self.vectors:
            v_distance = np.dot(query_vector, v) / (
                np.linalg.norm(query_vector) * np.linalg.norm(v)
            )
            self.assertGreaterEqual(v_distance, closest_distance)

    @patch("numpy.random.uniform")
    def test_random_level_generation(self, mock_uniform):
        """Test the random level generation."""
        mock_uniform.return_value = 0.5

        hnsw = HNSW(self.space, self.M, self.ef_construction)
        level = hnsw._HNSW__get_random_level()

        expected_level = 0
        self.assertEqual(level, expected_level)

        mock_uniform.return_value = 0.01
        level = hnsw._HNSW__get_random_level()
        self.assertGreater(level, 0)

    def test_node_connections_limit(self):
        """Test that nodes don't exceed their connection limits."""
        for layer, nodes in self.hnsw.layered_graph.items():
            for node in nodes:
                if layer == 0:
                    self.assertLessEqual(
                        len(node.neighbors.get(layer, [])), self.hnsw.m0
                    )
                else:
                    self.assertLessEqual(
                        len(node.neighbors.get(layer, [])), self.hnsw.mMax
                    )

    def test_bidirectional_connections(self):
        """Test that connections are bidirectional."""
        for layer, nodes in self.hnsw.layered_graph.items():
            for node in nodes:
                for neighbor in node.neighbors.get(layer, []):
                    self.assertIn(node, neighbor.neighbors.get(layer, []))


if __name__ == "__main__":
    unittest.main()
