import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM


# query
def query_prompt(chunk_list, query):
    chunk_str = "\n\n".join(chunk_list)

    prompt_template = f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are an AI assistant for giving short answers based on given context.<|eot_id|>
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
