# Dev and Tracking

## Transformer and attention blocks

### stats

Train dataset: shakespear dialogue.

| model | time per epoch | precision| num GPU| num cores | batch size | (peak) memory|
| standard GPT| 2h13m | f16 | 1 (V100)| 6 | 64 | 54% |
| standard GPT with memory efficient attention (no attention dropout) | 55 m| fp16| 1 (V100) | 6| 64 | 54% |

multi-gpu strategy

on a 2-GPU machine with 12 cores, base model is minGPT.
|strategy| model | time per epoch | precision|  batch size | (peak) memory utils|
| vanlia ddp | standard GPT with memory efficient attention (no attention dropout) | 30 m| fp16|  64 | 54% |
| vanlia ddp | standard GPT with memory efficient attention (no attention dropout) no param_group| 25 m| fp16|  128 | 97% |
| fsdp|  standard GPT with memory efficient attention (no attention dropout) no param_group | 19 m| fp16|  128 | 68% |
| fsdp offload2cpu|  standard GPT with memory efficient attention (no attention dropout) no param_group | 48 m| fp16|  128 | NA |
| fsdp auto wrap each layer + activation checkpointing|standard GPT with memory efficient attention (no attention dropout) no param_group | 1h  22m| fp16|  512 | NA |
| fsd_native + activation checkpointing|standard GPT with memory efficient attention (no attention dropout) no param_group | 20m| fp16|  512 | 50% | 
| fsd_native + activation checkpointing|standard GPT with memory efficient attention (no attention dropout) no param_group | 19m| fp16|  1024 | 75% |
|deepspeed zero stage 3 param_group | 30m| fp16| 128| 69% |
|deepspeed zero stage 3 param_group with activation checkpoint| 23m| fp16| 512| 50% |
|deepspeed zero stage 3 param_group with activation checkpoint| 22m| fp16| 1024| 50% |
#### 12/25/2022

issue: xformer with fused modules complain `Runtime error: Triton requires cuda 11.4+`.

fix: install(update) locally cuda 11.6 following [install cuda](https://medium.com/analytics-vidhya/installing-any-version-of-cuda-on-ubuntu-and-using-tensorflow-and-torch-on-gpu-b1a954500786).

issue: then the program gives me ` 2928 segmentation fault (core dumped)  python tasks/train_char.py base=configs/train_char.yaml`.

using gdp python, it shows the segmentation fault happens in triton's implementation of dropout.

remove fused mlp, the code works.

tried fused layer normalization, also get segmentation error. I guess I am giving up on triton.


###### memory efficient attention

no operators found for q, k, v with dropout and casual mask. Removing attention dropout makes the code work and one epoch only takes 55 minutes (two time faster).

Notes: the `memory_efficient_attention` is already trying out different approaches (including flash attention) in its dispatcher.

#### 12/28/2022

do not install fairscale from conda-forge, it will replace pytorch with cpu-only version -> use `pip install fairscale`.

with reinstalled cuda and nvidia-driver (newer than 11.4), xformer's triton still complains `RuntimeError: Triton requires CUDA 11.4+` error. 


#### 1/3/2022

fsdp occupies gpu memory after interrupt. Use `sudo fuser -v /dev/nvidia*` and `sudo kill -9 <PID>` to kill the python process and free gpu memory.


#### 1/6/2023

manually wrapping is worse than the default wrapping policy provided by fsdp

#### 1/8/2023

argument activation_checkpointing from fsdp_native is pytorch is missing in real torch lightning code: https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.strategies.DDPFullyShardedNativeStrategy.html#pytorch_lightning.strategies.DDPFullyShardedNativeStrategy (it's only meant for 1.9.0)

#### 1/9/2023

updated pytorch to 1.13.1 and xformer correspondingly to use activation checkpoint in fsdp_native.

transfomer_auto_wrap_policy does not work, backward prefetch does not seem to make a difference


##### Deepspeed

error: `/usr/bin/ld: cannot find -lcudart`. Fix: Make a symbolic link to libcuda where ld is searching it.

```sudo ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib/libcudart.so```
