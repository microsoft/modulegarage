from typing import Optional, Type

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torchlightning_utils.base_registry import BaseRegistry


class Registry(BaseRegistry):
    """Registry for various entities
    Args:
        BaseRegistry (_type_): _description_
    Returns:
        _type_: _description_
    """

    @classmethod
    def register_module(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a torch module to registry with key :p:`name`
        :param name: Key with which the module will be registered.
        """

        return cls._register_impl("module", to_register, name, assert_type=torch.nn.Module)

    @classmethod
    def register_attentionblock(cls, to_register=None, *, name:Optional[str]=None):
        r""" Register an attention block (torch.nn.Module) to registry with key :p:`name`
        :param name: Key with which the block will be registered.
        """
        return cls._register_impl("attentionblock", to_register, name, assert_type=torch.nn.Module)

    @classmethod
    def register_lightningmodule(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a lightning module to registry with key :p:`name`
        :param name: Key with which the lightning module will be registered.
        """

        return cls._register_impl("lightningmodule", to_register, name, assert_type=LightningModule)

    @classmethod
    def register_datamodule(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a lightning data module to registry with key :p:`name`
        :param name: Key with which the lightning datamodule will be registered.
        """

        return cls._register_impl("datamodule", to_register, name, assert_type=LightningDataModule)

    @classmethod
    def get_module(cls, name: str) -> Type[torch.nn.Module]:
        return cls._get_impl("module", name)

    @classmethod
    def get_attentionblock(cls, name:str) -> Type[torch.nn.Module]:
        return cls._get_impl("attentionblock", name)

    @classmethod
    def get_lightningmodule(cls, name: str) -> Type[LightningModule]:
        return cls._get_impl("lightningmodule", name)

    @classmethod
    def get_datamodule(cls, name: str) -> Type[LightningDataModule]:
        return cls._get_impl("datamodule", name)


registry = Registry()
