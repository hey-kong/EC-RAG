from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
    QueryBundle,
)
from llama_index.core.schema import NodeWithScore
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


import os
import time
from typing import List

from reranker import local_reranker
from customed_statistic import global_statistic

def direct_qa_prompt(query):
    prompt = f'''Answer the following question based on your knowledge, 
without external references:\n\nQuestion: {query}\n\nAnswer:'''
    return prompt


def query_prompt(chunk_list, query):
    chunks = "\n\n".join(chunk_list)

    prompt_template = f"""\
{chunks}

Given the above information and not prior knowledge, answer the question: {query}

Respond with a concise answer only, do not output any other words.
"""

    return prompt_template

def llm_self_feedback_prompt(prompt, ori_answer):
    prompt_template = (
        f"Original Prompt: {prompt}\n\n"
        f"Original Answer: {ori_answer}\n\n"
        "Objective (O): You are to evaluate the original answer for the original prompt "
        "on a scale of 1 to 10 based on its accuracy and reasonability. "
        "Additionally, determine if the original prompt needs more related context (1) or less context (-1).\n"
        "Style (S): Provide a clear and concise evaluation in a formal and professional style.\n\n"
        "Response (R): Ensure the output follows this format:\n"
        "Evaluation Score: [1-10]. (The answer is highly accurate if (Score >= 9))\n"
        "Context Adjustment: [1, -1].\n"
        "Context adjustment should output \"less context (-1)\" with a probability of 60%, and \"more context (1)\" with a probability of 40%.\n\n"
        "(output example):\n"
        "Evaluation Score: 8\n"
        "Context Adjustment: -1\n"
    )
    return prompt_template



class SAGEAgent:
    def __init__(
        self,
        docstore: str,
        similarity_top_k: int,
        model_name: str = "deepseek-chat",
        embed_model: str = "BAAI/bge-small-en-v1.5",
        ori_min_k: int = 7,
        g: float = 0.3,
        feedback_score_threshold: int = 9,
        max_self_feedback_iters: int = 3,
    ):
        # hyper parameters
        self.similarity_top_k = similarity_top_k
        self.ori_min_k = min(similarity_top_k, ori_min_k)
        self.g = g
        self.feedback_score_threshold = feedback_score_threshold
        self.max_self_feedback_iters = max_self_feedback_iters
        
        # Setup LLM and embedding model
        self.llm = DeepSeek(model=model_name, api_key=os.environ.get("DEEPSEEK_API_KEY"))
        Settings.llm = self.llm
        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model)

        # index: vector
        print("ChunkRAGAgent: Loading vector retriever...")
        self.storage_context = StorageContext.from_defaults(persist_dir=docstore + "_vec")
        self.vector_index = load_index_from_storage(self.storage_context)
        self.vector_retriever = self.vector_index.as_retriever(similarity_top_k=similarity_top_k)
        print("ChunkRAGAgent: Successfully loaded vector retriever.")

    def _retrieval(self, query: str) -> List[NodeWithScore]:
        return self.vector_retriever.retrieve(QueryBundle(query_str=query))

    def _gradient_based_chunk_selection(self, min_k, reranked_nodes) -> List[str]:
        if not reranked_nodes or min_k <= 0:
            return []

        selected_chunks = []
        # get min_k chunks
        min_k = min(min_k, len(reranked_nodes))
        for i in range(min_k):
            node, _ = reranked_nodes[i]
            selected_chunks.append(node.text)
        gradient_threshold = reranked_nodes[min_k - 1][1] * self.g
        for i in range(min_k, len(reranked_nodes)):
            node, score = reranked_nodes[i]
            if score >= gradient_threshold:
                selected_chunks.append(node.text)
            else:
                break
        return selected_chunks
    
    def _llm_self_feedback(self, prompt: str, response: str):
        feedback_prompt = llm_self_feedback_prompt(prompt, response)
        feedback_response = self.llm.complete(feedback_prompt)
        feedback_lines = feedback_response.text.strip().split("\n")
        
        # parse feedback
        try:
            score_line = feedback_lines[0].split(":")
            score = int(score_line[1].strip())
            context_adjustment_line = feedback_lines[1].split(":")
            context_adjustment = int(context_adjustment_line[1].strip())
            more_context = True if context_adjustment == 1 else False
        except (IndexError, ValueError):
            score = 0
            more_context = False
        
        return score, more_context
    
    def query(self, query: str):
        start = time.perf_counter()
        retrieved_nodes = self._retrieval(query)
        retrieve_end = time.perf_counter()
        global_statistic.add_to_list("retrieval_time", retrieve_end - start)
        
        if not retrieved_nodes:
            response = Settings.llm.complete(direct_qa_prompt(query))
            return response.text

        # rerank
        reranked_nodes = local_reranker.rerank_nodes(query, retrieved_nodes, top_k=len(retrieved_nodes))

        iter = 0
        cur_min_k = self.ori_min_k
        while iter < self.max_self_feedback_iters:
            selected_chunks = self._gradient_based_chunk_selection(cur_min_k, reranked_nodes)
            
            # generate response
            prompt = query_prompt(selected_chunks, query)
            response = Settings.llm.complete(prompt)

            # self-feedback
            feedback_score, more_context = self._llm_self_feedback(prompt, response.text)
            if feedback_score >= self.feedback_score_threshold:
                break

            # update min_k
            if more_context:
                cur_min_k += 1
            else:
                cur_min_k -= 1

            iter += 1
        global_statistic.add_to_list("self_feedback_iters", iter)
        return response.text