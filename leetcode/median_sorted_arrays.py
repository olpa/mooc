"""
https://leetcode.com/problems/median-of-two-sorted-arrays/
Given two sorted arrays nums1 and nums2 of size m and n respectively,
return the median of the two sorted arrays.

Follow up: The overall run time complexity should be O(log (m+n)).
"""
import itertools
import unittest
from typing import List, Generator


def generator_for_sorted(
        arr1: List[int], arr2: List[int]
) -> Generator[int, None, None]:
    pos1 = 0
    pos2 = 0
    while True:
        take_from1 = pos1 < len(arr1)
        take_from2 = pos2 < len(arr2)
        if take_from1 and take_from2:
            if arr1[pos1] < arr2[pos2]:
                take_from2 = False
            else:
                take_from1 = False
        if not (take_from1 or take_from2):
            break
        if take_from1:
            value = arr1[pos1]
            pos1 += 1
        else:
            value = arr2[pos2]
            pos2 += 1
        yield value


class Solution:
    def findMedianSortedArrays(
            self,
            nums1: List[int],
            nums2: List[int]
    ) -> float:
        ordered_seq = generator_for_sorted(nums1, nums2)
        median_pos = (len(nums1) + len(nums2) - 1) / 2.0
        median_seq = itertools.islice(
            ordered_seq,
            int(median_pos),
            int(median_pos) + 2)
        pre_median = next(median_seq)
        if median_pos == int(median_pos):
            return pre_median
        return (pre_median + next(median_seq)) / 2.0


class SolutionTest(unittest.TestCase):
    def testGeneratorFirstEmpty(self):
        merged = list(generator_for_sorted([], [1, 2, 3]))
        self.assertEqual(merged, [1, 2, 3])

    def testGeneratorSecondEmpty(self):
        merged = list(generator_for_sorted([1, 2, 3], []))
        self.assertEqual(merged, [1, 2, 3])

    def testGeneratorWithDuplicated(self):
        merged = list(generator_for_sorted([1, 2, 2, 3], [1, 2, 3]))
        self.assertEqual(merged, [1, 1, 2, 2, 2, 3, 3])

    def testGeneratorSmokeTest(self):
        merged = list(generator_for_sorted([1, 3, 9], [2, 4, 6]))
        self.assertEqual(merged, [1, 2, 3, 4, 6, 9])

    def testLeetcode(self):
        fixture = [
            {
                "arr1": [1, 3],
                "arr2": [2],
                "expected": 2.0,
            },
            {
                "arr1": [1, 2],
                "arr2": [3, 4],
                "expected": 2.5,
            },
            {
                "arr1": [0, 0],
                "arr2": [0, 0],
                "expected": 0.0,
            },
            {
                "arr1": [],
                "arr2": [1],
                "expected": 1.0,
            },
            {
                "arr1": [2],
                "arr2": [],
                "expected": 2.0,
            },
        ]
        for case in fixture:
            with self.subTest(**case):
                merged = Solution().findMedianSortedArrays(
                    case["arr1"],
                    case["arr2"])
                self.assertEqual(merged, case["expected"])


unittest.main()
