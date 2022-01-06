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


def has_rectangle_intersection(rect_pair) -> bool:
    validate_rect_pair(rect_pair)
    return rect_pair["rectangle2"]["y1"] < 0


def has_rectangle_intersection_jsonstr(s_json: str) -> bool:
    rect_pair = json.loads(s_json)
    return has_rectangle_intersection(rect_pair)
