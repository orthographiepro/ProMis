from unittest import TestCase, main

import numpy as np

from promis.logic.spatial import MaxVelocity

class TestMaxVelocity(TestCase):
    def test_prob_max(self):
        data = np.array(
            [[0,  2, 4, -5, 23, 23, 1],
             [0,  2, 3, -1, 17, 24, 5],
             [0, -1, 4, -1, 46, 23, 0]]
        )
        expected =  np.array([3, 2, 2, 2, 1, 1, 1]) / 3
        results = MaxVelocity._prob_max(data, 0)

        self.assertTrue(np.equal(results, expected).all(), f"expected {expected}, got {results}")

if __name__ == "__main__":
    main()
