from transformers import AutoModel, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-70B"  # 例としてBERTのモデルを使用
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

save_directory = "/blob/projects/ds_rlhf/models/Meta-Llama-3-70B"

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

