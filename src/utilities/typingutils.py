import numpy as np


def is_typed_list(lst: object, obj_type: type, allow_nones: bool = False) -> bool:
    """
    Check if a variable is a list that contains objects of specific type.

    :param lst: The variable/list to check
    :param obj_type: The type of objects that the list should contain (for the check to return true)
    :param allow_nones: When set to true, each item of the list can also be None (use for List[Optional[obj_type]])

    :return: Whether lst is a list that contains only objects of type obj_type
    """
    assert isinstance(lst, object)
    assert isinstance(obj_type, type)
    assert isinstance(allow_nones, bool)

    if not isinstance(lst, list):
        return False

    for obj in lst:
        b1: bool = isinstance(obj, obj_type)
        b2: bool = allow_nones and obj is None
        if not (b1 or b2):
            return False

    return True


def is_typed_tuple(tpl: object, obj_type: type, allow_none: bool = False, allow_empty: bool = True) -> bool:
    """
    Check if a variable is a tuple that contains objects of specific type.

    :param tpl: The variable/list to check
    :param obj_type: The type of objects that the tuple should contain (for the check to return true)
    :param allow_none: When set to true, return true if tpl is None
    :param allow_empty: When set to true, return true if tpl is empty
    :return: Whether tpl is a tuple that contains only objects of type obj_type
    """
    assert isinstance(tpl, object)
    assert isinstance(obj_type, type)
    assert isinstance(allow_none, bool)
    assert isinstance(allow_empty, bool)

    if allow_none and tpl is None:
        return True

    if not isinstance(tpl, tuple):
        return False

    if not allow_empty and len(tpl) == 0:
        return False

    for obj in tpl:
        if not isinstance(obj, obj_type):
            return False

    return True


def is_int(x, allow_numpy: bool = True):
    assert isinstance(allow_numpy, bool)

    types = (int,)
    if allow_numpy:
        types += (np.int8, np.int16, np.int32, np.int64)

    return isinstance(x, types)


def is_tuple(x, length: int = None):
    assert isinstance(length, int) or length is None

    if not isinstance(x, tuple):
        return False

    if length is None:
        return True
    else:
        return len(x) == length
