import pickle

MODEL_REGISTRY = {}
BASE_NET_REGISTRY = {}


def register_model(cls):
    if cls.__name__ in MODEL_REGISTRY.keys():
        raise ValueError("Duplicate model registry detected!")
    MODEL_REGISTRY[cls.__name__] = cls
    return cls


def register_base_net(cls):
    if cls.__name__ in BASE_NET_REGISTRY.keys():
        raise ValueError("Duplicate base net registry detected!")
    BASE_NET_REGISTRY[cls.__name__] = cls
    return cls


def load_model(config_path, weights_path=None, new=False):
    if not new and weights_path is None:
        raise ValueError("If your are loading a model and not recreating it, you need to specify a weights path.")

    with open(config_path, 'rb') as file:
        args = pickle.load(file)

    if args['class_name'] not in MODEL_REGISTRY.keys():
        raise ValueError('Unknown name of class {1}. Registered classes are: {0}'.format(MODEL_REGISTRY.keys(),
                                                                                         args['class_name']))

    cls_name = args['class_name']
    if new:
        return MODEL_REGISTRY[cls_name].from_config_new(config=args, weights_path=weights_path)
    else:
        return MODEL_REGISTRY[cls_name].from_config(config=args, weights_path=weights_path, training=False)


def get_model(model_name):
    return MODEL_REGISTRY[model_name]


def load_base_net(name):
    if name not in BASE_NET_REGISTRY:
        raise ValueError('Unknown name of class. Registered classes are: {0}'.format(BASE_NET_REGISTRY.keys()))
    return BASE_NET_REGISTRY[name]
