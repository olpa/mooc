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
    validate_rect_pair(rect_pair)
    return rect_pair["rectangle2"]["y1"] < 0


def has_rectangle_intersection_jsonstr(s_json: str) -> bool:
    rect_pair = json.loads(s_json)
    return has_rectangle_intersection(rect_pair)
