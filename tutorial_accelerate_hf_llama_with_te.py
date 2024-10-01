# from utils import *
from utils_fsdp import *

# For Llama 2, download weights from https://huggingface.co/meta-llama/Llama-2-7b-hf (Hugging Face weight format).
# For Llama 3, download weights from https://huggingface.co/meta-llama/Meta-Llama-3-8B (Hugging Face weight format).
hyperparams.model_name = "/models/v2/llama-7bf-hf/"
# hyperparams.model_name = "/workspace/models/Meta-Llama-3-8B" # <== Add model weight location here
# hyperparams.model_name = "/models/v2/llama-v2-70b-hf/" # <== Add model weight location here
# hyperparams.mixed_precision = "bf16"
hyperparams.mixed_precision = "fp8"
hyperparams.batch_size = 1
hyperparams.enable_te_llama = True
# hyperparams.enable_te_llama = False
print(hyperparams.model_name, hyperparams.mixed_precision, hyperparams.enable_te_llama)

# Init the model and accelerator wrapper
accelerator, model, optimizer, train_dataloader, lr_scheduler = wrap_with_accelerator(hyperparams)

# Finetune the model
finetune_model(model, hyperparams, accelerator, train_dataloader, optimizer, lr_scheduler)
