def str2bool(s: str) -> bool:
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError('Cannot convert to bool: ' + str(s))
