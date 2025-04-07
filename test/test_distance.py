"""
Unit tests for the Distance class.
"""

import unittest
import numpy as np

from hnsw.distance import Distance
from hnsw.node import Node


class TestDistance(unittest.TestCase):
    """Test cases for the Distance class."""

    def setUp(self):
        """Set up test fixtures."""
        self.distance_cosine = Distance("cosine")
        self.distance_euclidean = Distance("euclidean")

        # Create test nodes
        self.node1 = Node([1.0, 0.0], 0)
        self.node2 = Node([0.0, 1.0], 0)
        self.node3 = Node([1.0, 1.0], 0)
        self.node4 = Node([0.5, 0.0], 0)

    def test_initialization(self):
        """Test that distance objects are initialized correctly."""
        self.assertEqual(self.distance_cosine.space, "cosine")
        self.assertEqual(self.distance_euclidean.space, "euclidean")
        with self.assertRaises(ValueError):
            Distance("invalid_space")

    def test_cosine_distance(self):
        """Test cosine distance calculation."""
        self.assertAlmostEqual(
            self.distance_cosine.distance(self.node1, self.node2), 1.0
        )

        self.assertAlmostEqual(
            self.distance_cosine.distance(self.node1, self.node1), 0.0
        )

        self.assertAlmostEqual(
            self.distance_cosine.distance(self.node1, self.node4), 0.0
        )
        expected_distance = 1.0 - (1.0 / np.sqrt(2))
        self.assertAlmostEqual(
            self.distance_cosine.distance(self.node1, self.node3),
            expected_distance,
            places=4,
        )

    def test_euclidean_distance(self):
        """Test euclidean distance calculation."""
        self.assertAlmostEqual(
            self.distance_euclidean._Distance__euclidean_distance(
                self.node1.vector, self.node2.vector
            ),
            np.sqrt(2),
            places=4,
        )

        self.assertAlmostEqual(
            self.distance_euclidean._Distance__euclidean_distance(
                self.node1.vector, self.node1.vector
            ),
            0.0,
            places=4,
        )

        self.assertAlmostEqual(
            self.distance_euclidean._Distance__euclidean_distance(
                self.node1.vector, self.node4.vector
            ),
            0.5,
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
