import unittest


class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        count = sum(cur == c for cur in s)
        return count * (count + 1) // 2


class Test(unittest.TestCase):
    cs = Solution()

    def test_ex1(self):
        back = self.cs.countSubstrings("abada", "a")
        self.assertEqual(back, 6)


if __name__ == '__main__':
    unittest.main()
