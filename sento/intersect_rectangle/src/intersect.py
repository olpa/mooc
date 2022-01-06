import json


def has_rectangle_intersection(rect_pair) -> bool:
    return rect_pair["rectangle2"]["y1"] < 0


def has_rectangle_intersection_jsonstr(s_json: str) -> bool:
    rect_pair = json.loads(s_json)
    return has_rectangle_intersection(rect_pair)
