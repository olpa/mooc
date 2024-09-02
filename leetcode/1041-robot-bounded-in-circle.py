"""
https://leetcode.com/problems/robot-bounded-in-circle/
"""

import unittest
from collections import namedtuple
from enum import Enum

Direction = Enum('Direction', ['EAST', 'NORTH', 'WEST', 'SOUTH'])

dir_to_dcoord = {
        Direction.EAST: (1, 0),
        Direction.NORTH: (0, 1),
        Direction.WEST: (-1, 0),
        Direction.SOUTH: (0, -1),
        }
from_dir_to_left = {
        Direction.EAST: Direction.NORTH,
        Direction.NORTH: Direction.WEST,
        Direction.WEST: Direction.SOUTH,
        Direction.SOUTH: Direction.EAST,
        }
from_dir_to_right = {
        Direction.EAST: Direction.SOUTH,
        Direction.NORTH: Direction.EAST,
        Direction.WEST: Direction.NORTH,
        Direction.SOUTH: Direction.WEST,
        }

State = namedtuple('State', 'x y dir')


def go_forward(state: State) -> State:
    (dx, dy) = dir_to_dcoord[state.dir]
    return State(
            x=state.x + dx,
            y=state.y + dy,
            dir=state.dir)


def walk(state: State, path: str) -> State:
    for ch in path:
        if ch == 'L':
            state = State(
                    x=state.x,
                    y=state.y,
                    dir=from_dir_to_left[state.dir])
        elif ch == 'R':
            state = State(
                    x=state.x,
                    y=state.y,
                    dir=from_dir_to_right[state.dir])
        elif ch == 'G':
            state = go_forward(state)
        else:
            assert False, f'Unknown code for direction: {ch}'
    return state


class Solution:
    def isRobotBounded(self, instructions: str) -> bool:
        state = State(x=0, y=0, dir=Direction.NORTH)
        seen = [state]
        for _ in range(4):
            state = walk(state, instructions)
            if state in seen:
                return True
            seen.append(state)
        return False


class RobotTest(unittest.TestCase):
    def testStepEast(self):
        state = State(x=3, y=5, dir=Direction.EAST)

        next_state = go_forward(state)

        expected_state = State(x=4, y=5, dir=Direction.EAST)
        self.assertEqual(next_state, expected_state)

    def testStepNorth(self):
        state = State(x=3, y=5, dir=Direction.NORTH)

        next_state = go_forward(state)

        expected_state = State(x=3, y=6, dir=Direction.NORTH)
        self.assertEqual(next_state, expected_state)

    def testStepWest(self):
        state = State(x=3, y=5, dir=Direction.WEST)

        next_state = go_forward(state)

        expected_state = State(x=2, y=5, dir=Direction.WEST)
        self.assertEqual(next_state, expected_state)

    def testStepSouth(self):
        state = State(x=3, y=5, dir=Direction.SOUTH)

        next_state = go_forward(state)

        expected_state = State(x=3, y=4, dir=Direction.SOUTH)
        self.assertEqual(next_state, expected_state)

    path1 = 'GGLLGG'
    path2 = 'GG'
    path3 = 'GR'
    start_pos = State(x=0, y=0, dir=Direction.NORTH)

    def testLocationForPath1(self):
        state = walk(self.start_pos, self.path1)
        expected_state = State(x=0, y=0, dir=Direction.SOUTH)
        self.assertEqual(state, expected_state)

    def testLocationForPath2(self):
        state = walk(self.start_pos, self.path2)
        expected_state = State(x=0, y=2, dir=Direction.NORTH)
        self.assertEqual(state, expected_state)

    def testLocationForPath3(self):
        state = walk(self.start_pos, self.path3)
        expected_state = State(x=0, y=1, dir=Direction.EAST)
        self.assertEqual(state, expected_state)


    is_bounded_func = Solution().isRobotBounded

    def testBoundedForPath1(self):
        is_bounded = self.is_bounded_func(self.path1)
        self.assertTrue(is_bounded)

    def testBoundedForPath2(self):
        is_bounded = self.is_bounded_func(self.path2)
        self.assertFalse(is_bounded)

    def testBoundedForPath3(self):
        is_bounded = self.is_bounded_func(self.path3)
        self.assertTrue(is_bounded)


if __name__ == '__main__':
    unittest.main()
