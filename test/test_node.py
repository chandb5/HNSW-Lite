"""
Unit tests for the Node class.
"""

import unittest
import numpy as np

from hnsw.node import Node
from hnsw.distance import Distance


class TestNode(unittest.TestCase):
    """Test cases for the Node class."""

    def setUp(self):
        """Set up test fixtures."""
        Node._global_id = 0
        self.vector1 = [1.0, 0.0]
        self.vector2 = [0.0, 1.0]
        self.level = 2
        self.node1 = Node(self.vector1, self.level)
        self.node2 = Node(self.vector2, self.level)

    def test_initialization(self):
        """Test that a node is initialized correctly."""
        self.assertEqual(self.node1.id, 0)
        self.assertEqual(self.node2.id, 1)
        np.testing.assert_array_equal(self.node1.vector, np.array(self.vector1))
        self.assertEqual(self.node1.level, self.level)
        self.assertFalse(self.node1.is_deleted)
        self.assertEqual(self.node1.magnitude, 1.0)

    def test_neighbor_initialization(self):
        """Test that neighbors are initialized correctly."""
        self.assertEqual(len(self.node1.neighbors), self.level + 1)
        for i in range(self.level + 1):
            self.assertEqual(len(self.node1.neighbors[i]), 0)

    def test_distance_calculation(self):
        """Test distance calculation between nodes."""
        self.assertAlmostEqual(self.node1.distance(self.node2), 1.0)
        self.assertAlmostEqual(self.node1.distance(self.node1), 0.0)

    def test_metadata(self):
        """Test metadata handling."""
        metadata = {"key": "value"}
        node = Node(self.vector1, self.level, metadata=metadata)
        self.assertEqual(node.metadata, metadata)

    def test_comparison_operators(self):
        """Test comparison operators."""
        node_small = Node([0.5, 0.0], self.level)
        node_large = Node([2.0, 0.0], self.level)

        self.assertTrue(node_small < node_large)
        self.assertTrue(node_large > node_small)
        self.assertFalse(node_small > node_large)
        self.assertFalse(node_large < node_small)

    def test_hash_and_equality(self):
        """Test hash and equality operators."""
        node1_copy = self.node1
        self.assertEqual(hash(self.node1), hash(node1_copy))
        self.assertEqual(self.node1, node1_copy)

        self.assertNotEqual(hash(self.node1), hash(self.node2))
        self.assertNotEqual(self.node1, self.node2)

        self.assertNotEqual(self.node1, "not a node")


if __name__ == "__main__":
    unittest.main()
