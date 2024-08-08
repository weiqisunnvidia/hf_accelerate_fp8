# hf_accelerate_fp8
- Installation:

    ```bash
    docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:24.07-py3 bash
    pip install datasets evaluate transformers
    git clone -b muellerzr-fp8-deepspeed-support https://github.com/huggingface/accelerate.git
    cd accelerate 
    pip install -e .
    ```

- LLaMa 3 8B training with FP8 and without FSDP:
    ```bash
    python test.py
    ```
    Result:
    ```bash
    10 finetuning steps complete!
    Average time taken per step: 185 milliseconds
    ```
- LLaMa 3 8B training with FP8+FSDP on single H100
    ```bash
    accelerate launch --config_file fsdp_fp8_1gpu.yaml test.py
    ```
    Error message saved in [fsdp_fp8_1gpu.log](./fsdp_fp8_1gpu.log)
- LLaMa 3 8B training with FP8+FSDP on 2 H100
    ```bash
    accelerate launch --config_file fsdp_fp8_2gpu.yaml test.py
    ```
    Error message saved in [fsdp_fp8_2gpu.log](./fsdp_fp8_2gpu.log)