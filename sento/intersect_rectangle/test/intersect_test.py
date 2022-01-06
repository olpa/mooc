import unittest
from hamcrest import assert_that, is_not, described_as, calling, raises

from lib import has_rectangle_intersection_jsonstr_matcher, \
        has_interval_intersection_matcher, pairwise

from intersect import has_rectangle_intersection_jsonstr, \
        has_rectangle_intersection


class HasIntersectionJsonstrTest(unittest.TestCase):

    def test_has_intersection(self):
        s_json = '''
        {
          "rectangle1": {
            "x1": -2.3, "y1": -4.5,
            "x2": 10.5, "y2": 12.2
          },
          "rectangle2": {
            "x1": -2.3, "y1": -4.5,
            "x2": 10.5, "y2": 12.2
          }
        }
        '''

        assert_that(s_json, has_rectangle_intersection_jsonstr_matcher())

    def test_has_no_intersection(self):
        s_json = '''
        {
          "rectangle1": {
            "x1": -2.3, "y1": -4.5,
            "x2": -1.5, "y2": -2.2
          },
          "rectangle2": {
            "x1": 12.3, "y1": 14.5,
            "x2": 10.5, "y2": 12.2
          }
        }
        '''

        assert_that(s_json,
                    is_not(has_rectangle_intersection_jsonstr_matcher()))

    def test_fail_on_unparsable_json(self):
        s_json = '''{ ###'''

        assert_that(
                calling(has_rectangle_intersection_jsonstr).with_args(s_json),
                raises(ValueError))


class HasIntersectionTest(unittest.TestCase):

    def test_validate_input(self):
        valid_rect = {
                "x1": 1, "y1": 2, "x2": 1.2, "y2": 1.3
                }
        fixture_set = [
                "dummy", "Not a dict",
                {}, "Not a dict: 'rectangle1' (missed)",
                {
                    "rectangle1": valid_rect,
                    "rectangle2": []
                }, "Not a dict: 'rectangle2'",
                {
                    "rectangle1": {**valid_rect, "x1": None},
                    "rectangle2": valid_rect
                }, "Bad field on 'rectangle1' ('x1')",
                {
                    "rectangle1": valid_rect,
                    "rectangle2": {**valid_rect, "y1": "not-a-number"}
                }, "Bad field on 'rectangle2' ('y1')",
                {
                    "rectangle1": valid_rect,
                    "rectangle2": {**valid_rect, "x2": 4+2j}
                }, "Bad field 'x2' (not a real number)",
                {
                    "rectangle1": {**valid_rect, "y2": "2"},
                    "rectangle2": valid_rect
                }, "Bad field 'y2'",
                ]

        for (fixture, describe) in pairwise(fixture_set):
            assert_that(
                    calling(has_rectangle_intersection).with_args(fixture),
                    described_as(describe,
                                 raises(ValueError)))

    def test_intersect_coordinate(self):
        # interval1.1, interval1.2, interval2.1, interval2.2
        fixture_set = [
                (1, 2, 3, 4), False, (1, 2, 4, 3), False,
                (1, 3, 2, 4), True,  (1, 3, 4, 2), True,
                (1, 4, 2, 3), True,  (1, 4, 3, 2), True,
                (2, 1, 3, 4), False, (2, 1, 4, 3), False,
                (2, 3, 1, 4), True,  (2, 3, 4, 1), True,
                (2, 4, 1, 3), True,  (2, 4, 3, 1), True,
                (3, 1, 2, 4), True,  (3, 1, 4, 2), True,
                (3, 2, 1, 4), True,  (3, 2, 4, 1), True,
                (3, 4, 1, 2), False, (3, 4, 2, 1), False,
                (4, 1, 2, 3), True,  (4, 1, 3, 2), True,
                (4, 2, 1, 3), True,  (4, 2, 3, 1), True,
                (4, 3, 1, 2), False, (4, 3, 2, 1), False]
        for (fixture, expected) in pairwise(fixture_set):
            matcher = has_interval_intersection_matcher()
            if not expected:
                matcher = is_not(matcher)
            assert_that(fixture, matcher)


if '__main__' == __name__:
    unittest.main()
