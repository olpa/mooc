"""
Having the coordinates of two rectangles,
determine if they intersect or not.
Public functions of the module:

    - `has_rectangle_intersection(rect_pair)`
    - `has_rectangle_intersection_jsonstr(s_json: str)`

An rectangle is a mapping with fields: `x1`, `y1`, `x2`, `y2`.
The point (`x1`, `y1`) is one corner, the point (`x2`, `y2`)
is the diagonally opposite corner.

There are no implications on the location of the base points,
it can be any of left-bottom, left-top, right-top or right-bottom.

A rectangle pair is a mapping with two fields, `rectangle1`
and `rectangle2`. The function that gets a string json as input,
parses the json to a python mapping.

The functions validate the input and throw `ValueError` if
the input doesn't match expectations.

The border of a rectangle belongs to the rectangle. It means
that if two rectangle touch each other, then we consider them
to intersect as well. In particular, degenerative rectangles
(of zero width and zero height) _do_ intersect if they are at
the same location.
"""

import json
import collections.abc
import numbers


def validate_rect(rect) -> None:
    if not isinstance(rect, collections.abc.Mapping):
        raise ValueError("Rectange should be a mapping")
    for field in ("x1", "y1", "x2", "y2"):
        val = rect.get(field, None)
        if not isinstance(val, numbers.Real):
            raise ValueError(f"The field '{field}' should be a real number")


def validate_rect_pair(rect_pair) -> None:
    if not isinstance(rect_pair, collections.abc.Mapping):
        raise ValueError("Rectange pair should be a mapping")
    validate_rect(rect_pair.get("rectangle1", None))
    validate_rect(rect_pair.get("rectangle2", None))


def has_interval_intersection(
        i11: numbers.Real,
        i12: numbers.Real,
        i21: numbers.Real,
        i22: numbers.Real
        ) -> bool:
    # intervals intersect if:
    # an end of the second interval is in the first interval, or ...
    i1min = min(i11, i12)
    i1max = max(i11, i12)
    if (i1min <= i21 <= i1max) or (i1min <= i22 <= i1max):
        return True
    # ... an end of the first interval is in the second interval
    i2min = min(i21, i22)
    i2max = max(i21, i22)
    return (i2min <= i11 <= i2max) or (i2min <= i12 <= i2max)


def has_rectangle_intersection(rect_pair) -> bool:
    """ See the docstring of the module """
    validate_rect_pair(rect_pair)
    r1 = rect_pair["rectangle1"]
    r2 = rect_pair["rectangle2"]
    has_x = has_interval_intersection(
            r1["x1"], r1["x2"], r2["x1"], r2["x2"])
    has_y = has_interval_intersection(
            r1["y1"], r1["y2"], r2["y1"], r2["y2"])
    return has_x and has_y


def has_rectangle_intersection_jsonstr(s_json: str) -> bool:
    """ See the docstring of the module """
    rect_pair = json.loads(s_json)
    return has_rectangle_intersection(rect_pair)
