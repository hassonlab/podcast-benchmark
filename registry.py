# Registry of functions for creating decoding models.
model_constructor_registry = {}
# Registry of functions for preprocessing data.
data_preprocessor_registry = {}
# Registry of functions for altering config values after data outputs.
config_setter_registry = {}

def register_model_constructor(name=None):
    """TODO: add doc here"""
    def decorator(fn):
        model_name = name or fn.__name__
        model_constructor_registry[model_name] = fn
        return fn
    return decorator


def register_data_preprocessor(name=None):
    """TODO: add doc here"""
    def decorator(fn):
        preprocessor_name = name or fn.__name__
        data_preprocessor_registry[preprocessor_name] = fn
        return fn
    return decorator


def register_config_setter(name=None):
    def decorator(fn):
        config_setter_name = name or fn.__name__
        config_setter_registry[config_setter_name] = fn
        return fn
    return decorator
