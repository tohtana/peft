from transformers import AutoModel, AutoTokenizer

targets = {
        "Meta-Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
        "Llama-2-7b-hf": "meta-llama/Llama-2-7b-hf",
        "Llama-2-13b-hf": "meta-llama/Llama-2-13b-hf",
        "Llama-2-70b-hf": "meta-llama/Llama-2-70b-hf",
}

for model_name, model_path in targets.items():
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    save_directory = f"/blob/projects/ds_rlhf/models/{model_name}"

    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

