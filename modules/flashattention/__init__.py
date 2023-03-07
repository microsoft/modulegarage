import traceback

import torch.nn as nn

from registry.registry import registry


def _try_register_flashattention_modules():
    # try:
    #     from .lightningmodules import CharDataModule, minGPT
    #     from .minGPT.mingpt.fmodel import FGPT, fBlock
    #     from .minGPT.mingpt.model import GPT  # noqa: F401

    #     print("import minGPT modules success")
    # except ImportError as e:
    #     flashattention_import_error = e

    #     print("import minGPT module failed: ")
    #     print(traceback.format_exc())

    #     @registry.register_module(name="FlashAttentionErr")
    #     class minGPTImportError(nn.Module):
    #         def __init__(self, *args, **kwargs):
    #             raise flashattention_import_error

    # from .lightningmodules import CharDataModule, minGPT
    # from .minGPT.mingpt.fusedmodel import FGPT
    # from .minGPT.mingpt.model import GPT  # noqa: F401
    from .lightningmodules import CharDataModule, minGPT
    from .minGPT.mingpt.fmodel import FGPT, fBlock
    from .minGPT.mingpt.model import GPT  # noqa: F401