import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pruning_model = AutoModelForCausalLM.from_pretrained("/data/wk/models/Meta-Llama-3.1-8B-Instruct").to(device)
pruning_tokenizer = AutoTokenizer.from_pretrained("/data/wk/models/Meta-Llama-3.1-8B-Instruct")

def judge_relevance_qa_prompt(chunk, query):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

Froma scale of 0 to 4.judge the relevance
between the query and the document.
Query:{query}

Document:{chunk}

Please output the score directly (0-4), without any additional text.
Output:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt

# 弃用
def judge_relevance_qa_prompt_2(chunk, query):
    prompt = f"""{chunk}

Based on the above information, score its relevance to the question: {query}

Do not output preamble or explanations.

Output the score (0-4):
"""
    return prompt


def judge_relevance(chunk, query):
    prompt = judge_relevance_qa_prompt(chunk, query)
    input_ids = pruning_tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = pruning_model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=pruning_tokenizer.eos_token_id,
        )
    generated_ids = outputs[0]
    # generated_text = pruning_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    # print(f"-------\nGenerated text:\n {generated_text}")

    new_token_ids = generated_ids[input_ids.shape[-1]:]
    new_text = pruning_tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    # print(f"Score: {new_text}")

    # 0-2认为不相关，34认为相关
    if new_text in ["0", "1", "2"]:
        return False, int(new_text)
    elif new_text in ["3", "4"]:
        return True, int(new_text)
    else:
        # exit(f"Invalid score: {new_text}")
        return False, 0


# query
def query_prompt(chunk_list, query):
    chunk_str = "\n\n".join(chunk_list)

    prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

{chunk_str}

Based on the above information, only give me the answer and do not output any other words.

Question: {query}

Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    return prompt_template

def local_generate_answer(chunk_list, query):
    prompt = query_prompt(chunk_list, query)
    input_ids = pruning_tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = pruning_model.generate(
            input_ids,
            max_new_tokens=40,
            pad_token_id=pruning_tokenizer.eos_token_id,
        )
    generated_ids = outputs[0]  # 获取生成的完整序列
    input_length = input_ids.shape[1]  # 计算原始输入的长度
    # 截取生成部分（排除输入提示）并解码
    answer = pruning_tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True)
    return answer