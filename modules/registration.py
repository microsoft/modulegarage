import logging

from registry.registry import registry

from .flashattention import _try_register_flashattention_modules


def make_module(id_module, **kwargs):
    logging.info("initializing module {}".format(id_module))
    _module = registry.get_module(id_module)
    assert _module is not None, "Could not find module with name {}".format(id_module)
    return _module(**kwargs)


_try_register_flashattention_modules()
