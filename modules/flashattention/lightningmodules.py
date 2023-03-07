from functools import partial
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from deepspeed.ops.adam import FusedAdam
from pytorch_lightning.utilities import rank_zero_info
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchlightning_utils.lr_scheduler import LinearWarmupCosineAnnealingLR

from registry.registry import registry

from .minGPT.mingpt.fmodel import fBlock


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        rank_zero_info("data has %d characters, %d unique." % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, i):
        chunk = self.data[i : i + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]

        # src and target are off by one, we want the model to predict the next word
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

    def to_tokens(self, message, device):
        return torch.tensor([self.stoi[s] for s in message], dtype=torch.long)[None, ...].to(device)

    def from_tokens(self, tokens):
        return "".join([self.itos[int(i)] for i in tokens])


@registry.register_datamodule(name="char")
class CharDataModule(pl.LightningDataModule):
    def __init__(self, dataset_config: Dict[str, Any], dataloader_config: Dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.dataset_config = dataset_config
        self.dataloader_config = dataloader_config

        text = open(self.dataset_config["input_path"], "r").read()
        self._train_dataset = CharDataset(
            text, self.dataset_config["block_size"]
        )  # one line of poem is roughly 50 characters

    # def setup(self, stage=None) -> None:
    #     self.random_sampler = RandomSampler(self._train_dataset)
    #     return

    @property
    def train_dataset(self):
        return self._train_dataset

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            # sampler=self.random_sampler,
            shuffle=False,
            batch_size=self.dataloader_config["batch_size"],
            num_workers=self.dataloader_config["num_workers"],
            pin_memory=True,
        )


@registry.register_lightningmodule(name="minGPT")
class minGPT(pl.LightningModule):
    def __init__(
        self, model_config: Dict[str, Any], optimizer_config: Dict[str, Any], scheduler_config: Dict[str, Any]
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model_class = registry.get_module(name=model_config["model_name"])
        self.model_config = self.model_class.get_default_config()
        self.model_config.model_type = "gpt2"
        self.model_config.attention = "scaled_dot_product"
        self._tokens_seen = 0

    def post_init(self, train_dataset: CharDataset, epochs: int) -> None:

        self.model_config.vocab_size = train_dataset.vocab_size
        self.model_config.block_size = train_dataset.block_size
        self.model_config.final_tokens = epochs * len(train_dataset) * train_dataset.block_size

        # self.model = self.model_class(config=self.model_config)
        # self.model.checkpoint_wrap()

    def training_step(self, batch, _):
        src, targets = batch
        # Update the tokens we've seen (tracked for LR scheduling)
        self._tokens_seen += (src >= 0).numel()

        # same action as inference
        logits, loss = self.model(src, targets)

        self.logger.log_metrics(
            {
                "train_loss": loss.mean(),
                "learning_rate": self.lr_schedulers().get_last_lr()[0],
            },
            step=self.trainer.global_step,
        )

        return loss

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        
        optim_groups = self.model.configure_optimizers(self.hparams.optimizer_config) 
        optimizer = FusedAdam(params=optim_groups, 
            lr=self.hparams.optimizer_config["lr"],
            betas=self.hparams.optimizer_config["betas"],
            adam_w_mode=True)
        # # optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        # optimizer = torch.optim.AdamW(
        #     params=optim_groups,
        #     lr=self.hparams.optimizer_config["lr"],
        #     betas=self.hparams.optimizer_config["betas"],
        # )
        # weight decay scheduler and parameter group between: model and balancer
        # optimizer = torch.optim.AdamW(
        #     params=self.trainer.model.parameters(),
        #     lr=self.hparams.optimizer_config["lr"],
        #     weight_decay=self.hparams.optimizer_config["weight_decay"],
        #     betas=self.hparams.optimizer_config["betas"],
        # )
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.scheduler_config["warmup_epochs"],
            self.hparams.scheduler_config["max_epochs"],
            self.hparams.scheduler_config["start_lr"],
            self.hparams.scheduler_config["min_lr"],
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def apply_activation_checkpoint(self):
        non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        check_fn = lambda submodule: isinstance(submodule, fBlock)
        apply_activation_checkpointing(
        self.model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )

    def configure_sharded_model(self) -> None:
        # Meant for deepspeed, Created within sharded model context, modules are instantly sharded across processes
        # as soon as they are made
        self.model = self.model_class(config=self.model_config)
        return 

    @torch.no_grad()
    def sample(x, steps, temperature=1.0, sample=False, top_k=None):
        """
        take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
        the sequence, feeding the predictions back into the model each time. Clearly the sampling
        has quadratic complexity unlike an RNN that is only linear, and has a finite context window
        of block_size, unlike an RNN that has an infinite context window.
        """
        block_size = self.model.get_block_size()
        self.model.eval()

        # CREDITS: https://github.com/karpathy/minGPT/blob/master/mingpt/utils.py
        def top_k_logits(logits, k):
            v, _ = torch.topk(logits, k)
            out = logits.clone()
            out[out < v[:, [-1]]] = -float("Inf")
            return out

        for _ in range(steps):
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
            logits = self.model(x_cond)

            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature

            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = top_k_logits(logits, top_k)

            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)

            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1)

        return x[0]  # escape the batch dimension
