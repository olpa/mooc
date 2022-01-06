from hamcrest.core.base_matcher import BaseMatcher

from intersect import has_rectangle_intersection_jsonstr, \
        has_interval_intersection


class HasRectangleIntersectionJsonstrMatcher(BaseMatcher):

    def _matches(self, s_json):
        return has_rectangle_intersection_jsonstr(s_json)

    def describe_to(self, description):
        description.append_text('rectangles should intersect')


def has_rectangle_intersection_jsonstr_matcher():
    return HasRectangleIntersectionJsonstrMatcher()


class HasIntervalIntersectionMatcher(BaseMatcher):

    def _matches(self, coordinates):
        return has_interval_intersection(*coordinates)

    def describe_to(self, description):
        description.append_text('intervals should intersect')


def has_interval_intersection_matcher():
    return HasIntervalIntersectionMatcher()


def pairwise(t):
    it = iter(t)
    return zip(it, it)
