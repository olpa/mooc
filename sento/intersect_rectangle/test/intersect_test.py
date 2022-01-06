import unittest
from hamcrest import assert_that, calling, raises

from lib import has_rectangle_intersection_jsonstr_matcher

from intersect import has_rectangle_intersection_jsonstr


class HasIntersectionJsonstrTest(unittest.TestCase):

    def test_with_json_as_string(self):
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

    def test_fail_on_unparsable_json(self):
        s_json = '''{ ###'''

        assert_that(
                calling(has_rectangle_intersection_jsonstr).with_args(s_json),
                raises(ValueError))


if '__main__' == __name__:
    unittest.main()
