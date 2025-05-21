import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, Future

import torch
import torch.serialization
from llama_index.core.schema import TextNode
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from kvcache_io import read_kvcache
from customed_statistic import global_statistic

QWEN_PROMPT_PREFIX = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
"""

LLAMA_PROMPT_PREFIX = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
"""


def get_model_type(model_name):
    model_lower = model_name.lower()
    if 'qwen3' in model_lower:
        return 'qwen3'
    elif 'qwen' in model_lower:
        return 'qwen'
    elif 'llama' in model_lower:
        return 'llama'
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def chunk_with_prefix(chunk_text, model_name):
    model_type = get_model_type(model_name)
    prefix = QWEN_PROMPT_PREFIX if model_type in ('qwen', 'qwen3') else LLAMA_PROMPT_PREFIX
    return f"{prefix}{chunk_text}\n\n"


def _build_suffix(model_type):
    suffix_map = {
        'qwen3': (
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n\n"
        ),
        'qwen': (
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        'llama': (
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
    }
    return suffix_map[model_type]


def judge_relevance_prompt(chunk, query, model_name):
    model_type = get_model_type(model_name)
    prefix = QWEN_PROMPT_PREFIX if model_type in ('qwen', 'qwen3') else LLAMA_PROMPT_PREFIX

    return (
        f"{prefix}{chunk}\n\n"
        f"Determine whether the above context is relevant to the question: {query}\n"
        f"If the context directly or indirectly helps answer the question, respond with \"Yes\".\n"
        f"If the context does not contain useful information, respond with \"No\".\n\n"
        f"Respond with \"Yes\" or \"No\" only, do not output any other words."
        f"{_build_suffix(model_type)}"
    )


def judge_complexity_prompt(query, model_name):
    model_type = get_model_type(model_name)
    prefix = QWEN_PROMPT_PREFIX if model_type in ('qwen', 'qwen3') else LLAMA_PROMPT_PREFIX

    return (
        f"{prefix}For the given question: {query}\n\n"
        f"Classify the question as easy or hard to answer.\n"
        f"If the question is simple, factual, or straightforward, respond with \"Easy\".\n"
        f"If the question is complex, nuanced, requires multi-step reasoning or in-depth analysis, respond with \"Hard\".\n\n"
        f"Respond with \"Easy\" or \"Hard\" only, do not output any other words."
        f"{_build_suffix(model_type)}"
    )


def query_prompt(chunk_list, query, model_name):
    model_type = get_model_type(model_name)
    prefix = QWEN_PROMPT_PREFIX if model_type in ('qwen', 'qwen3') else LLAMA_PROMPT_PREFIX
    chunks = "\n\n".join(chunk_list)

    return (
        f"{prefix}{chunks}\n\n"
        f"Given the above context, answer the question: {query}\n\n"
        f"Only give me the answer and do not output any other words."
        f"{_build_suffix(model_type)}"
    )


class KVCacheLoader:
    _preload_executor = ThreadPoolExecutor(max_workers=1)

    def __init__(self, device: torch.device):
        self.device = device
        self._preloaded_path = None
        self._kvcache_future: Future = None

    def preload_kvcache(self, cache_file_path: str):
        self._preloaded_path = cache_file_path
        self._kvcache_future = KVCacheLoader._preload_executor.submit(
            read_kvcache, cache_file_path
        )

    def load_kvcache(self, cache_file_path: str):
        start = time.perf_counter()
        if (
            self._kvcache_future is not None
            and self._preloaded_path == cache_file_path
            and self._kvcache_future.done()
        ):
            kvcache = self._kvcache_future.result()
        else:
            kvcache = read_kvcache(cache_file_path)

        kvcache.key_cache = [t.to(self.device, non_blocking=True) for t in kvcache.key_cache]
        kvcache.value_cache = [t.to(self.device, non_blocking=True) for t in kvcache.value_cache]
        torch.cuda.synchronize()
        end = time.perf_counter()
        global_statistic.add_to_list("load_kvcache_time", end - start)
        return kvcache


class CustomModelWrapper:
    def init(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kvcache_loader = KVCacheLoader(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.model.config.eos_token_id
        self.model_type = get_model_type(model_path)

    def judge_relevance(self, node, query, use_kvcache=False, preload_node: Optional[TextNode] = None):
        start = time.perf_counter()
        prompt = judge_relevance_prompt(node.text, query, self.model_type)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        kvcache = DynamicCache()
        if use_kvcache:
            kvcache = self.kvcache_loader.load_kvcache(node.metadata["kvcache_file_path"])
        if use_kvcache and preload_node is not None:
            self.kvcache_loader.preload_kvcache(preload_node.metadata["kvcache_file_path"])

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                past_key_values=kvcache
            )
        generated_ids = outputs[0]
        input_length = inputs.input_ids.shape[1]
        answer = self.tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True).strip()
        end = time.perf_counter()
        global_statistic.add_to_list("judge_relevance_time", end - start)
        if answer == "No":
            return False
        return True

    def judge_complexity(self, query):
        start = time.perf_counter()
        prompt = judge_complexity_prompt(query, self.model_type)
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device).input_ids
        with torch.no_grad():
            next_token_logits = self.model(input_ids).logits[:, -1, :]
        # next_token_id = next_token_logits.argmax(dim=-1)
        # first_token = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
        # print("First Token:", first_token)
        easy_id = self.tokenizer("Easy", add_special_tokens=False).input_ids[0]
        hard_id = self.tokenizer("Hard", add_special_tokens=False).input_ids[0]
        log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        complexity_score = torch.sigmoid(log_probs[0, hard_id] - log_probs[0, easy_id]).item()
        end = time.perf_counter()
        global_statistic.add_to_list("judge_complexity_time", end - start)
        return complexity_score

    def generate_answer(self, query, nodes, use_kvcache=False):
        chunk_list = [node.text for node in nodes]
        prompt = query_prompt(chunk_list, query, self.model_type)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        kvcache = DynamicCache()
        if use_kvcache:
            kvcache = self.kvcache_loader.load_kvcache(nodes[0].metadata["kvcache_file_path"])

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                past_key_values=kvcache
            )
        generated_ids = outputs[0]
        input_length = inputs.input_ids.shape[1]
        answer = self.tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True).strip()
        end = time.perf_counter()
        global_statistic.add_to_list("slm_generate_time", end - start)
        return answer


slm = CustomModelWrapper()
