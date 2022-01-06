import unittest
from hamcrest import assert_that

from lib import has_intersection_jsonstr


class HasIntersectionTest(unittest.TestCase):

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
          },
        }
        '''

        assert_that(s_json, has_intersection_jsonstr())


if '__main__' == __name__:
    unittest.main()
