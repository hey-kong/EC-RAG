import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch
import torch.serialization
from llama_index.core.schema import TextNode
from transformers.cache_utils import DynamicCache
from modelscope import AutoTokenizer, AutoModelForCausalLM

from serde import TorchDeserializer
from customed_statistic import global_statistic

torch.serialization.add_safe_globals([DynamicCache])

PROMPT_PREFIX = f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
"""


def chunk_with_prefix(chunk_text):
    chunk = PROMPT_PREFIX + f"""
{chunk_text}
"""

    return chunk


def judge_relevance_prompt(chunk, query):
    prompt_template = PROMPT_PREFIX + f"""
{chunk}

Determine whether the above information is relevant to the following question. If the information directly or indirectly helps answer the question, respond with "Yes", otherwise respond with "No".

Question: {query}

Respond with "Yes" or "No" only, do not output any other words.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    return prompt_template


def judge_complexity_prompt(query):
    prompt_template = PROMPT_PREFIX + f"""For the given question: {query}

Classify the complexity of the question as high or low.

Respond with "High" or "Low" only, do not output any other words.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    return prompt_template


def query_prompt(chunk_list, query):
    chunks = "\n\n".join(chunk_list)

    prompt_template = PROMPT_PREFIX + f"""
{chunks}

Given the above information and not prior knowledge, answer the question: {query}

Respond with a concise answer only, do not output any other words.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    return prompt_template


class KVCacheLoader:
    _preload_executor = ThreadPoolExecutor(max_workers=1)

    def __init__(self, deserializer):
        self.deserializer = deserializer
        self._preloaded_path = None
        self._file_content_future = None

    def preload_kvcache_file(self, cache_file_path: str):
        def _read_file():
            with open(cache_file_path, 'rb') as f:
                return f.read()

        self._preloaded_path = cache_file_path
        self._file_content_future = KVCacheLoader._preload_executor.submit(_read_file)

    def load_kvcache(self, cache_file_path: str = None):
        if self._file_content_future is not None and self._preloaded_path == cache_file_path:
            file_content = self._file_content_future.result()
        else:
            if cache_file_path is None:
                raise RuntimeError("Must call preload_kvcache_file before load_kvcache, or provide a cache file path.")
            with open(cache_file_path, 'rb') as file:
                file_content = file.read()

        start = time.perf_counter()
        kvcache = self.deserializer.from_bytes(file_content)
        end = time.perf_counter()
        global_statistic.add_to_list("load_kvcache_time", end - start)
        return kvcache


class CustomModelWrapper:
    def init(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kvcache_loader = KVCacheLoader(TorchDeserializer(torch.float16, self.device))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.model.config.eos_token_id

    def judge_relevance(self, node, query, use_kvcache=False, preload_node: Optional[TextNode] = None):
        start = time.perf_counter()
        prompt = judge_relevance_prompt(node.text, query)
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        kvcache = self.kvcache_loader.load_kvcache(node.metadata["kvcache_file_path"]) if use_kvcache else None
        if use_kvcache and preload_node is not None:
            self.kvcache_loader.preload_kvcache_file(preload_node.metadata["kvcache_file_path"])

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_token_id,
                past_key_values=kvcache
            )
        generated_ids = outputs[0]  # 获取生成的完整序列
        input_length = input_ids.shape[1]  # 计算原始输入的长度
        # 截取生成部分（排除输入提示）并解码
        answer = self.tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True).strip()
        end = time.perf_counter()
        global_statistic.add_to_list("judge_relevance_time", end - start)
        if answer == "No":
            return False
        return True

    def judge_complexity(self, query, preload_node: Optional[TextNode] = None):
        start = time.perf_counter()
        if preload_node is not None:
            self.kvcache_loader.preload_kvcache_file(preload_node.metadata["kvcache_file_path"])
        prompt = judge_complexity_prompt(query)
        input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device).input_ids
        with torch.no_grad():
            next_token_logits = self.model(input_ids).logits[:, -1, :]
            # next_token_id = next_token_logits.argmax(dim=-1)
            # first_token = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            # print("First Token:", first_token)
            high_id = self.tokenizer("High", add_special_tokens=False).input_ids[0]
            low_id = self.tokenizer("Low", add_special_tokens=False).input_ids[0]
            log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
            complexity_score = torch.sigmoid(log_probs[0, high_id] - log_probs[0, low_id]).item()
        end = time.perf_counter()
        global_statistic.add_to_list("judge_complexity_time", end - start)
        return complexity_score

    def generate_answer(self, query, nodes, use_kvcache=False):
        start = time.perf_counter()
        chunk_list = [node.text for node in nodes]
        prompt = query_prompt(chunk_list, query)
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        kvcache = self.kvcache_loader.load_kvcache(nodes[0].metadata["kvcache_file_path"]) if use_kvcache else None

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_token_id,
                past_key_values=kvcache
            )
        generated_ids = outputs[0]  # 获取生成的完整序列
        input_length = input_ids.shape[1]  # 计算原始输入的长度
        # 截取生成部分（排除输入提示）并解码
        answer = self.tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True).strip()
        end = time.perf_counter()
        global_statistic.add_to_list("slm_generate_time", end - start)
        return answer


slm = CustomModelWrapper()
