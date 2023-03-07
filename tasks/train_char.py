import sys

import torch
from pytorch_lightning.strategies import (
    DDPFullyShardedNativeStrategy,
    DDPFullyShardedStrategy,
)
from torchlightning_utils import cli, pl_instantiate

import modules
from registry.registry import registry


def main():
    # skip the program name in sys.argv
    cfg = cli.parse(sys.argv[1:])
    trainer = pl_instantiate.instantiate_trainer(
        cfg["trainer"],
        cfg["callbacks"],
        cfg["logger"],
        cfg.get("seed_everything", None),
    )
    module = registry.get_lightningmodule(cfg["model_name"])(**cfg["model"])
    datamodule = registry.get_datamodule(cfg["data_name"])(**cfg["data"])
    train_dataset = datamodule.train_dataset
    module.post_init(
        train_dataset,
        epochs=trainer.max_epochs,
    )
    # module.apply_activation_checkpoint()
    trainer.fit(module, datamodule)

    # print memory usage stats, ref: https://www.youtube.com/watch?v=NfZeeR7bISk
    print("batch size: ", cfg["data"]["dataloader_config"]["batch_size"])

    # pay attention to cudaMalloc retries, it should be 0 for efficiency. Decrease your batch size if not.
    print(torch.cuda.memory_summary())
    
    # concise version of above code
    cuda_info = torch.cuda.memory_stats()
    num_retries = cuda_info.get("num_alloc_retries", 0)
    cuda_oom = cuda_info.get("num_ooms", 0)
    print(f"cudaMallock retries = {num_retries}")
    print(f"cuda OOM = {cuda_oom}\n")

if __name__ == "__main__":
    main()
