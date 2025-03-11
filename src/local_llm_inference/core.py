import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM

def judge_relevance_qa_prompt(chunk, query):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

From a scale of 0 to 4, judge the relevance between the query and the document.

Query:{query}

Document:{chunk}

Please output the score directly (0-4), without any additional text.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
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

# query
def query_prompt(chunk_list, query):
    chunk_str = "\n\n".join(chunk_list)

    prompt_template = f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are an assistant for giving short answers based on given context.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{chunk_str}

Based on the above information, only give me the answer and do not output any other words.

Question: {query}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    return prompt_template


class CustomModelWrapper:
    def init(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.eos_token_ids = [128001, 128009]

    def judge_relevance(self, chunk, query):
        prompt = judge_relevance_qa_prompt(chunk, query)
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.eos_token_ids,
            )
        generated_ids = outputs[0]
        # generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        # print(f"-------\nGenerated text:\n {generated_text}")

        new_token_ids = generated_ids[input_ids.shape[-1]:]
        new_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        # print(f"Score: {new_text}")

        # 0-2认为不相关，34认为相关
        if new_text in ["0", "1", "2"]:
            return False, int(new_text)
        elif new_text in ["3", "4"]:
            return True, int(new_text)
        else:
            # exit(f"Invalid score: {new_text}")
            return False, 0
    def generate_answer(self, chunk_list, query):
        prompt = query_prompt(chunk_list, query)
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.eos_token_ids,
            )
        generated_ids = outputs[0]  # 获取生成的完整序列
        input_length = input_ids.shape[1]  # 计算原始输入的长度
        # 截取生成部分（排除输入提示）并解码
        answer = self.tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True)
        return answer

local_llm = CustomModelWrapper()