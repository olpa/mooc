import json


def has_rectangle_intersection_jsonstr(s_json: str) -> bool:
    rect_pair = json.loads(s_json)
    return bool(rect_pair)
