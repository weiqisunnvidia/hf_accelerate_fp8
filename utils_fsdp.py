# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import time
import sys

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    AutoConfig,
)
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils.dataclasses import FP8RecipeKwargs

from accelerate import FullyShardedDataParallelPlugin as FSDPPlugin
from torch.distributed.fsdp import MixedPrecision
from functools import partial
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from te_llama import TELlamaForCausalLM, TELlamaDecoderLayer

class HyperParameters:
    def __init__(self):
        self.mixed_precision = "bf16"
        self.model_name = "/models/v2/llama-7bf-hf/" # <== Add model weight location here
        self.dataset_name = "timdettmers/openassistant-guanaco"
        self.dataset_text_field = "text"
        self.learning_rate = 1.41e-5
        self.batch_size = 8
        self.max_seq_length = 256
        self.gradient_accumulation_steps = 1
        self.num_warmup_steps = 5
        self.num_training_steps = 10
        self.enable_te_llama = True


hyperparams = HyperParameters()


def get_dataloaders(accelerator: Accelerator, hyperparams):
    dataset = load_dataset(hyperparams.dataset_name, split="train")
    tokenizer = AutoTokenizer.from_pretrained(hyperparams.model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            padding=False,
            max_length=hyperparams.max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

    with accelerator.main_process_first():
        dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    # Simply pad to the multiple of 16 for both FP8 and BF16 precision
    pad_to_multiple_of = 16
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    dataloader_params = {
        "batch_size": hyperparams.batch_size,
        "collate_fn": data_collator,
        "drop_last": True,
    }
    train_dataloader = DataLoader(dataset, **dataloader_params)
    return train_dataloader


def init_baseline_model(hyperparams):
    # Init the model
    config = AutoConfig.from_pretrained(hyperparams.model_name)
    # make sure to use flash_attention to do iso comparison with TELlamaModel
    config._attn_implementation = "flash_attention_2"
    max_memory_allocated = torch.cuda.max_memory_allocated()
    print(f"Peak GPU memory allocated by PyTorch before model: {max_memory_allocated / 1024**3:.2f} GB")
    model = AutoModelForCausalLM.from_pretrained(
        hyperparams.model_name,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    # model = model.cuda()
    # Needed for the cases when using TELlamaForCausalLM. So adding here for 1:1 comparison
    model.config.use_cache = False

    return model


def init_te_llama_model(hyperparams):
    # Init the model
    from te_llama import TELlamaForCausalLM

    config = AutoConfig.from_pretrained(hyperparams.model_name)
    config._attn_implementation = "flash_attention_2"
    model = TELlamaForCausalLM.from_pretrained_local(
        hyperparams.model_name,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    # model = model.cuda()
    # Needed for the cases when using TELlamaForCausalLM
    model.config.use_cache = False

    return model


def wrap_with_accelerator(hyperparams):
    # # Create FP8 kwarg handler if required
    # fp8_kwarg_handler = (
    #     [FP8RecipeKwargs(backend="te")] if hyperparams.mixed_precision == "fp8" else None
    # )

    # # Init HF accelerator that's used for training
    # accelerator = Accelerator(
    #     log_with="wandb",
    #     gradient_accumulation_steps=hyperparams.gradient_accumulation_steps,
    #     mixed_precision=hyperparams.mixed_precision,
    #     kwargs_handlers=fp8_kwarg_handler,
    # )
    if hyperparams.enable_te_llama:
        FSDP_WRAP_POLICY = partial(transformer_auto_wrap_policy, transformer_layer_cls={TELlamaDecoderLayer})
    else:
        FSDP_WRAP_POLICY = partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer})
    # AcceleratorState()._reset_state(True)
    fsdp_plugin = FSDPPlugin(
        auto_wrap_policy=FSDP_WRAP_POLICY,
        use_orig_params=True,
        mixed_precision_policy=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
    )
    if hyperparams.mixed_precision == "fp8":
        FP8_RECIPE_KWARGS = {"fp8_format": "HYBRID", "amax_history_len": 32, "amax_compute_algo": "max"}
        kwargs_handlers = [FP8RecipeKwargs(backend="TE", **FP8_RECIPE_KWARGS)]
    else:
        kwargs_handlers = []
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=hyperparams.gradient_accumulation_steps,
        mixed_precision=hyperparams.mixed_precision, 
        fsdp_plugin=fsdp_plugin, 
        kwargs_handlers=kwargs_handlers
    )

    if hyperparams.enable_te_llama:
        model = init_te_llama_model(hyperparams)
    else: 
        model = init_baseline_model(hyperparams)


    # accelerator.print(f'State: {accelerator.state}')
    train_dataloader = get_dataloaders(accelerator, hyperparams)

    max_memory_allocated = torch.cuda.max_memory_allocated()
    print(f"Peak GPU memory allocated by PyTorch before optimizer: {max_memory_allocated / 1024**3:.2f} GB")
    # Wrap model, optimizer/scheduler, dataloaders in accelerate
    optimizer = AdamW(params=model.parameters(), lr=hyperparams.learning_rate, fused=True)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=hyperparams.num_training_steps,
    )
    max_memory_allocated = torch.cuda.max_memory_allocated()
    print(f"Peak GPU memory allocated by PyTorch before accelerator.prepare(): {max_memory_allocated / 1024**3:.2f} GB")
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    max_memory_allocated = torch.cuda.max_memory_allocated()
    print(f"Peak GPU memory allocated by PyTorch before train: {max_memory_allocated / 1024**3:.2f} GB")
    return accelerator, model, optimizer, train_dataloader, lr_scheduler


def finetune_model(model, hyperparams, accelerator, train_dataloader, optimizer, lr_scheduler):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    train_dataloader = enumerate(train_dataloader)

    # Warmup iters
    for _ in range(hyperparams.num_warmup_steps):
        step, batch = next(train_dataloader)
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    max_memory_allocated = torch.cuda.max_memory_allocated()
    print(f"Peak GPU memory allocated by PyTorch after train: {max_memory_allocated / 1024**3:.2f} GB")
    # Get the timers ready
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    start.record()
    # Training iters
    for _ in range(hyperparams.num_training_steps):
        step, batch = next(train_dataloader)
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    torch.cuda.synchronize()
    end.record()
    accelerator.end_training()

    print(
        f"{hyperparams.num_training_steps} finetuning steps complete!\nAverage time taken per step:"
        f" {(start.elapsed_time(end)/hyperparams.num_training_steps):.0f} milliseconds"
    )
