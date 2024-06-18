import os
from transformers import AutoModel, AutoTokenizer

targets = {
        # "Meta-Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
        # "Llama-2-7b-hf": "meta-llama/Llama-2-7b-hf",
        # "Llama-2-13b-hf": "meta-llama/Llama-2-13b-hf",
        # "Llama-2-70b-hf": "meta-llama/Llama-2-70b-hf",
        # "Mixtral-8x7B-v0.1": "mistralai/Mixtral-8x7B-v0.1",
        # "Mixtral-8x22B-v0.1": "mistralai/Mixtral-8x22B-v0.1"
}

for model_name, model_path in targets.items():
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    save_directory = f"/blob/projects/ds_rlhf/models/{model_name}"
    
    if os.path.exists(save_directory):
        continue

    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

