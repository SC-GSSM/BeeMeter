import warnings


def warn(msg, **kwargs):
    warnings.warn(msg, category=Warning)


def error(msg, cls):
    raise cls(msg)


def check_instance(obj, cls, allow_none=False):
    if allow_none:
        if obj is None:
            return obj
    if not isinstance(obj, cls):
        raise TypeError(
            "Given object should be an instance of class '{0}' but type {1} is given.".format(cls.__name__, type(obj)))
    return obj


def check_bool(obj, use_warn=False):
    answer = warn if use_warn else error
    if type(obj) is not bool:
        answer(msg="Attribute should be boolean but {0} is given.".format(type(obj)), cls=TypeError)
    return obj


def check_number(obj, cls=None, value_min=None, value_max=None, one_of=None, use_warn=False):
    answer = warn if use_warn else error
    if cls is not None:
        if not type(obj) == cls:
            answer(
                msg="Given number should be of type '{0}' but instance of {1} is used.".format(cls.__name__, type(obj)),
                cls=TypeError)
    if value_min is not None:
        if obj < value_min:
            answer(msg="Number {0} is too small. Should be grater than {1}.".format(obj, value_min), cls=ValueError)
    if value_max is not None:
        if obj > value_max:
            answer(msg="Number {0} is too big. Should be less than {1}.".format(obj, value_max), cls=ValueError)
    if one_of is not None:
        if obj not in one_of:
            answer(msg="Given object {0} need to be one of {1}".format(obj, one_of), cls=ValueError)
    return obj


def check_array(obj, expected_shape=None, expected_dtype=None):
    if expected_shape is not None:
        if not obj.ndim == len(expected_shape):
            raise NumpyDimensionError(expected_dim=len(expected_shape), actual_dim=obj.ndim)
        for axis in range(len(expected_shape)):
            if expected_shape[axis] is not None:
                if not obj.shape[axis] == expected_shape[axis]:
                    raise NumpyShapeError(position=axis, expected_shape=expected_shape[axis],
                                          actual_shape=obj.shape[axis])

    if expected_dtype is not None:
        if obj.dtype == expected_dtype:
            raise TypeError(
                "The type of the array have to be '{0}' but is actually '{1}'.".format(expected_dtype, obj.dtype))

    return obj


class BoundingBoxError(Exception):
    pass


class InconsistentNumberDefaultBoxError(Exception):
    def __init__(self, number_default, actual_number, **kwargs):
        message = 'The number of predicted boxes must match the number of boxes determined by the configuration. The' \
                  'handler calculates {0} default boxes but the prediction has {1}.'.format(number_default,
                                                                                            actual_number)
        super().__init__(message)


class NumpyDimensionError(Exception):
    def __init__(self, expected_dim, actual_dim):
        message = 'The dimension of the array have to be {0} but is actually {1}.'.format(expected_dim, actual_dim)
        super().__init__(message)


class NumpyShapeError(Exception):
    def __init__(self, position, expected_shape, actual_shape):
        message = 'The shape on axis {0} have to be {1} but {2} is given.'.format(position, expected_shape,
                                                                                  actual_shape)
        super().__init__(message)
