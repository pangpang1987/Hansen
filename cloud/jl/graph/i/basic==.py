import os
import sys

sys.path.append(os.getcwd())  # 父级目录

def get_dict_keys(obj: dict):
    """返回字典的键"""
    return list(obj.keys()) if obj else []


def get_keys_values_of_dicts_without_default(obj: dict,
                                             keys = []) -> list:
    """根据键值返回值"""
    obj_keys = get_dict_keys(obj)
    return [obj.get(key) for key in keys
            if key in obj_keys] if (keys and obj) else []


def get_no_repeat_values_set_of_dicts_in_list(arr, keys = []) -> list:
    """根据键值顺序返回值，并去除重复值"""
    field_set = set()
    for obj in arr:
        field_set.update(get_keys_values_of_dicts_without_default(obj, keys))
    return list(field_set)
