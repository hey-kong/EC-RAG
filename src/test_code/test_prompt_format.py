import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pruning_model = AutoModelForCausalLM.from_pretrained("/data/wk/models/Meta-Llama-3.1-8B-Instruct").to(device)
pruning_tokenizer = AutoTokenizer.from_pretrained("/data/wk/models/Meta-Llama-3.1-8B-Instruct")

message = "From a scale of 0 to 4.judge the relevance\nbetween the query and the document.\nQuery:[query)\nDocument:{document)\nOutput:"
answer = "4"
messages = []
# messages.append({"role": "system", "content": "You are an accurate and reliable AI assistant that can determine the degree of relevance between a chunk and a query."})
messages.append({"role": "user", "content": message})
messages.append({"role": "assistant", "content": answer})
prompt = pruning_tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
print(prompt)