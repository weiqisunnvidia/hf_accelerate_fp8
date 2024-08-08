# Import necessary packages and methods
from utils import *


# Default hyperparams, also defined in `utils.py` in class `Hyperparameters`
## !!! `model_name` attr must point to the location of the model weights !!!
# For Llama 2, download weights from https://huggingface.co/meta-llama/Llama-2-7b-hf (Hugging Face weight format).
# For Llama 3, download weights from https://huggingface.co/meta-llama/Meta-Llama-3-8B (Hugging Face weight format).
hyperparams.model_name =  # <== Add model weight location here e.g. "/path/to/downloaded/llama/weights"
hyperparams.mixed_precision = "fp8"


# Init the model and accelerator wrapper
model = init_te_llama_model(hyperparams)
accelerator, model, optimizer, train_dataloader, lr_scheduler = wrap_with_accelerator(model, hyperparams)


# Finetune the model
finetune_model(model, hyperparams, accelerator, train_dataloader, optimizer, lr_scheduler)