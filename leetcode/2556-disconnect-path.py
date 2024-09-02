"""
https://leetcode.com/problems/disconnect-path-in-a-binary-matrix-by-at-most-one-flip/description/
"""
from typing import List, Tuple
import unittest

def find_path(
        matrix: List[List[int]],
        dir1: Tuple[int, int],
        dir2: Tuple[int, int]
        ) -> List[Tuple[int, int]]:
    m = len(matrix)  # i
    n = len(matrix[0])  # j
    # items: (i, j, path to (i, j), direction to grow
    q = [(((0, 0),), dir2), (((0, 0),), dir1)]
    while len(q):
        (path, direction) = q.pop()
        (i, j) = path[-1]
        i_next = i + direction[0]
        if i_next >= m:
            continue
        j_next = j + direction[1]
        if j_next >= n:
            continue
        if not matrix[i_next][j_next]:
            continue
        path_next = (*path, (i_next, j_next))
        if (i_next == m - 1) and (j_next == n - 1):
            return path_next                               # return
        q.append((path_next, dir2))
        q.append((path_next, dir1))


class Solution:
    def isPossibleToCutPath(self, grid: List[List[int]]) -> bool:
        if len(grid) == 1 and len(grid[0]) == 1:
            return False
        path1 = find_path(grid, (1, 0), (0, 1))
        if not path1:
            return True
        path2 = find_path(grid, (0, 1), (1, 0))
        if not path2:
            return True
        shared = set(path1).intersection(path2)
        return len(shared) > 2


class Test(unittest.TestCase):
    def testFindPath(self):
        """
        1 1 1
        0 1 1
        1 1 1
        """
        matrix = [[1, 1, 1], [0, 1, 1], [1, 1, 1]]

        path1 = find_path(matrix, (1, 0), (0, 1))
        self.assertEqual(path1, ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)))

        path2 = find_path(matrix, (0, 1), (1, 0))
        self.assertEqual(path2, ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2)))

    sol = Solution()

    def testExample1(self):
        grid = [[1,1,1],[1,0,0],[1,1,1]]
        is_possible = self.sol.isPossibleToCutPath(grid)
        self.assertTrue(is_possible)

    def testExample2(self):
        grid = [[1,1,1],[1,0,1],[1,1,1]]
        is_possible = self.sol.isPossibleToCutPath(grid)
        self.assertFalse(is_possible)

    def testOne(self):
        grid = [[1]]
        is_possible = self.sol.isPossibleToCutPath(grid)
        self.assertFalse(is_possible)


if __name__ == '__main__':
    unittest.main()
