---
base_model: Qwen/Qwen3.5-4B
library_name: peft
pipeline_tag: text-generation
tags:
  - prime-agent
  - lora
  - coordinator
  - experimental
---

# Qwen3.5 4B Prime Agent Coordinator Step 4

This private checkpoint is the selected initialization for the next coordinator rung.
It is a PEFT LoRA adapter over `Qwen/Qwen3.5-4B`, not a standalone dense model and not
a promoted coordinator.

## Training

- Base revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- BF16 LoRA rank 16, alpha 32
- Learning rate: `1e-5`
- Selected after 4 optimizer steps
- Training mix: 32 direct-parent, 32 single-parent, 32 single-child demonstrations
- Training data SHA-256: `1ebeb965bcd46dd7866a90260fcd22623d9eba51e29da2478accde11acc32686`
- Adapter SHA-256: `8bdc69ba7bd21c2a386428fde0b5baaf7be9ce5e485cdf3bba5dd35b8bcb29a7`

## Selection result

On fresh development tasks, step 4 retained direct solving at `8/8` joint success.
Single-child answer accuracy was `2/8`, but complete answer-and-protocol success was
`0/8`. Later checkpoints did not improve those task metrics and introduced more
duplicate cells or rollout errors.

The adapter is retained because it is the safest point from which to isolate the
missing admission -> explicit child reply -> resumed parent transition. Frozen
Capacity Battery V1 was not run because this checkpoint did not pass development
promotion.

The repository's `training/` directory contains the exact training configuration,
metrics, experiment record, and raw development traces.
