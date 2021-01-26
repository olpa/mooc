"""
https://leetcode.com/explore/challenge/card/january-leetcoding-challenge-2021/582/week-4-january-22nd-january-28th/3614/

A matrix diagonal is a diagonal line of cells starting from some
cell in either the topmost row or leftmost column and going in
the bottom-right direction until reaching the matrix's end.
For example, the matrix diagonal starting from mat[2][0],
where mat is a 6 x 3 matrix, includes cells mat[2][0], mat[3][1],
and mat[4][2].

Given an m x n matrix mat of integers, sort each matrix diagonal
in ascending order and return the resulting matrix.
"""

import itertools
import unittest
from typing import List, Tuple


def generate_diagonal_indices(
        i: int, j: int, m: int, n: int
) -> List[Tuple[int, int]]:
    """
    Generate indices of a diagonal. For example, for i=2, j=0, m=7, n=3,
    the list is [(2, 0), (3, 1), (4, 2)]

    0 <= i < m
    0 <= j < n
    One of `i` or `j` is `0`
    """
    ls = []
    while i < m and j < n:
        ls.append((i, j))
        i += 1
        j += 1
    return ls


class Solution:
    def diagonalSort(self, mat: List[List[int]]) -> List[List[int]]:
        """ doesn't corrupt the input `mat` """
        m = len(mat)
        if not m:
            return mat
        n = len(mat[0])
        # create NxM matrix, temporary filled with zeros
        out_mat = [[0] * n for __ in range(m)]

        # calculate start indices of diagonals
        start_idx_ls = list(zip(range(m), itertools.repeat(0)))
        # start range from 1, as (0, 0) already in `start_idx_row
        start_idx_col = zip(itertools.repeat(0), range(1, n))
        start_idx_ls.extend(start_idx_col)

        for start_idx in start_idx_ls:
            idc = generate_diagonal_indices(start_idx[0], start_idx[1], m, n)
            diagonal = [mat[i][j] for (i, j) in idc]
            sorted_ = sorted(diagonal)
            for ((i, j), v) in zip(idc, sorted_):
                out_mat[i][j] = v
        return out_mat


class SolutionTest(unittest.TestCase):
    fixture = (
        {
            "mat": [],
            "expected": [],
            "msg": "empty matrix",
        },
        {
            "mat": [[1, 4, 2]],
            "expected": [[1, 4, 2]],
            "msg": "one row",
        },
        {
            "mat": [[1], [4], [2]],
            "expected": [[1], [4], [2]],
            "msg": "one column",
        },
        {
            "mat": [[3, 3, 1, 1], [2, 2, 1, 2], [1, 1, 1, 2]],
            "expected": [[1, 1, 1, 1], [1, 2, 2, 2], [1, 2, 3, 3]],
            "msg": "leetcode"
        },
    )

    def test(self):
        for item in self.fixture:
            with self.subTest(**item):
                sorted_ = Solution().diagonalSort(item["mat"])
                self.assertEqual(sorted_, item["expected"])


unittest.main()
