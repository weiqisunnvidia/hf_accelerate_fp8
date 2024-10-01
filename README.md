# hf_accelerate_fp8
Developed performance benchmarking script for testing Transformer Engine in HuggingFace Accelerate for LLaMa models based on https://github.com/NVIDIA/TransformerEngine/tree/main/docs/examples/te_llama and https://github.com/huggingface/accelerate/tree/main/benchmarks/fp8/transformer_engine. 

- Installation:

    ```bash
    docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:24.09-py3 bash
    pip install accelerate datasets evaluate transformers
    ```

- Running:

    Set `model_name`, `mixed_precision`, `batch_size`, and `enable_te_llama` in `tutorial_accelerate_hf_llama_with_te.py`
    ```bash
    accelerate launch --num_processes 8 tutorial_accelerate_hf_llama_with_te.py
    ```