from hamcrest.core.base_matcher import BaseMatcher

from intersect import has_rectangle_intersection_jsonstr


class HasRectangleIntersectionJsonstrMatcher(BaseMatcher):

    def _matches(self, s_json):
        return has_rectangle_intersection_jsonstr(s_json)

    def describe_to(self, description):
        description.append_text('rectangles should intersect')


def has_rectangle_intersection_jsonstr_matcher():
    return HasRectangleIntersectionJsonstrMatcher()
