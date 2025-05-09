from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    QueryBundle,
)
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.llms.ollama import Ollama

import os
import time
import numpy as np
from typing import List
import Stemmer
from scipy import spatial

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



class ChunkRAGAgent:
    def __init__(
        self,
        docstore: str,
        vec_topk: int,
        bm25_topk: int,
        model_name: str = "deepseek-chat",
        local_model_name: str = "qwen2.5:7b",
        embed_model: str = "BAAI/bge-small-en-v1.5",
        similarity_threshold: float = 0.9,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        variance_threshold: float = 0.01,
    ):
        self.vec_topk = vec_topk
        self.bm25_topk = bm25_topk
        
        # Setup LLM and embedding model
        self.llm = DeepSeek(model=model_name, api_key=os.environ.get("DEEPSEEK_API_KEY"))
        Settings.llm = self.llm
        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model)
        self.local_llm = Ollama(model=local_model_name)

        # index: vector and bm25
        print("ChunkRAGAgent: Loading vector and bm25 retrievers...")
        self.storage_context = StorageContext.from_defaults(persist_dir=docstore + "_vec")
        self.vector_index = load_index_from_storage(self.storage_context)
        self.vector_retriever = self.vector_index.as_retriever(similarity_top_k=vec_topk)

        self.docstore = SimpleDocumentStore.from_persist_path(docstore + "_docstore.pkl")
        self.bm25_retriever = BM25Retriever.from_defaults(
            docstore=self.docstore,
            similarity_top_k=bm25_topk,
            stemmer=Stemmer.Stemmer("english"),
            language="english",
        )

        print("ChunkRAGAgent: Successfully loaded vector and bm25 retrievers.")

        # para settings
        self.similarity_threshold = similarity_threshold
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.variance_threshold = variance_threshold
    
    def _rewrite_query(self, query: str) -> str:
        prompt = (
            "You are an AI assistant that improves user "
            "queries for better search results. "
            "Rewrite the following query to be more "
            "effective for document retrieval without "
            "changing its meaning.\n"
            f"Original Query: \"{query}\"\n"
            "Rewritten Query:"
        )
        
        response = Settings.llm.complete(prompt)
        return response.text.strip()
    
    def _hybrid_retrieval(self, rewritten_query: str) -> List[NodeWithScore]:
        query_bundle = QueryBundle(query_str=rewritten_query)
        
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
        
        node_dict = {}
        
        for node in vector_nodes:
            node_id = node.node_id
            if node_id not in node_dict:
                node_dict[node_id] = {"node": node, "vector_score": node.score, "bm25_score": 0.0}

        # adding bm25 scores to the node_dict
        for node in bm25_nodes:
            node_id = node.node_id
            if node_id in node_dict:
                node_dict[node_id]["bm25_score"] = node.score
            else:
                node_dict[node_id] = {"node": node, "vector_score": 0.0, "bm25_score": node.score}
        
        # TODO(wk): if sort is redundant or not?
        # calculate combined scores
        nodes_with_score = []
        for node_data in node_dict.values():
            combined_score = (
                self.vector_weight * node_data["vector_score"] + 
                self.bm25_weight * node_data["bm25_score"]
            )
            node = node_data["node"]
            node.score = combined_score
            nodes_with_score.append(node)

        nodes_with_score.sort(key=lambda x: x.score, reverse=True)
        return nodes_with_score

    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        return 1 - spatial.distance.cosine(embedding1, embedding2)

    def _filter_redundant_chunks(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        if not nodes:
            return []
            
        filtered_nodes = []
        for node in nodes:
            # if first node, add it to filtered list
            current_embedding = node.node.embedding
            if current_embedding is None:
                node.node.embedding = Settings.embed_model.get_text_embedding(node.get_text())
                current_embedding = node.node.embedding

            if not filtered_nodes:
                filtered_nodes.append(node)
                continue

            is_redundant = False
            for filtered_node in filtered_nodes:
                filtered_embedding = filtered_node.node.embedding
                
                if filtered_embedding is not None and current_embedding is not None:
                    similarity = self._calculate_similarity(current_embedding, filtered_embedding)
                    
                    if similarity > self.similarity_threshold:
                        is_redundant = True
                        break

            if not is_redundant:
                filtered_nodes.append(node)
                
        return filtered_nodes
    
    def _get_relevance_score(self, chunk: str, query: str) -> float:
        prompt = (
            "You are an AI assistant tasked with "
            "determining the relevance of a text "
            "chunk to a user query. "
            "Analyze the provided chunk and query, then "
            "assign a relevance score between 0 and 1, where 1 means highly relevant and 0 "
            "means not relevant at all.\n"
            f"Chunk: {chunk}\n"
            f"User Query: {query}\n"
            "A single decimal number between 0 and 1, "
            "representing the final relevance score. "
            "No other text.\n"
            "Relevance Score (between 0 and 1):"
        )
        
        response = self.local_llm.complete(prompt)
        try:
            return float(response.text.strip())
        except ValueError:
            return 0.0
    
    def _refine_relevance_score(self, chunk: str, query: str, initial_score: float) -> float:
        """Reflect on and refine the relevance score"""
        prompt = (
            "You have assigned a relevance score to a "
            "text chunk based on a user query. "
            f"Your initial score was: {initial_score}\n"
            "Reflect on your scoring and adjust the score "
            "if necessary. Provide the final score.\n"
            f"Chunk: {chunk}\n"
            f"User Query: {query}\n"
            "A single decimal number between 0 and 1, "
            "representing the final relevance score. "
            "No other text.\n"
            "Final Relevance Score (between 0 and 1):"
        )
        
        response = self.local_llm.complete(prompt)
        try:
            return float(response.text.strip())
        except ValueError:
            return initial_score
    
    def _score_and_filter_chunks(
        self, nodes: List[NodeWithScore], query: str
    ) -> List[NodeWithScore]:
        if not nodes:
            return []

        scores = []
        for node in nodes:
            chunk_text = node.get_text()
            
            initial_score = self._get_relevance_score(chunk_text, query)
            final_score = self._refine_relevance_score(chunk_text, query, initial_score)
            
            node.score = final_score
            scores.append(final_score)

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        var_score = np.var(scores)

        threshold = mean_score + std_score if var_score < self.variance_threshold else mean_score
        filtered_nodes = [node for node in nodes if node.score >= threshold]
        filtered_nodes.sort(key=lambda x: x.score, reverse=True)

        return filtered_nodes
    
    def basic_query(self, query: str):
        start = time.perf_counter()
        retrieved_nodes = self._hybrid_retrieval(query)
        retrieve_end = time.perf_counter()
        global_statistic.add_to_list("hybrid_retrieval_time", retrieve_end - start)
        
        if not retrieved_nodes:
            response = Settings.llm.complete(direct_qa_prompt(query))
            return response.text
        # rerank
        reranked_nodes = local_reranker.rerank_nodes(query, retrieved_nodes)
        chunk_list = [node.text for node, _ in reranked_nodes]
        rerank_end = time.perf_counter()
        global_statistic.add_to_list("rerank_nodes_time", rerank_end - retrieve_end)
        response = Settings.llm.complete(query_prompt(chunk_list, query))
        qa_end = time.perf_counter()
        global_statistic.add_to_list("qa_time", qa_end - rerank_end)
        return response.text

    def query(self, query: str):
        # start = time.perf_counter()
        # query = self._rewrite_query(query)
        # rewrite_end = time.perf_counter()
        # global_statistic.add_to_list("rewrite_query_time", rewrite_end - start)

        start = time.perf_counter()
        retrieved_nodes = self._hybrid_retrieval(query)
        retrieve_end = time.perf_counter()
        global_statistic.add_to_list("hybrid_retrieval_time", retrieve_end - start)

        # filtered_nodes = self._filter_redundant_chunks(retrieved_nodes)
        # filter_end = time.perf_counter()
        # global_statistic.add_to_list("filter_redundant_chunks_time", filter_end - retrieve_end)

        final_nodes = self._score_and_filter_chunks(retrieved_nodes, query)
        score_end = time.perf_counter()
        global_statistic.add_to_list("score_and_filter_chunks_time", score_end - retrieve_end)

        if not final_nodes:
            response = Settings.llm.complete(direct_qa_prompt(query))
            return response.text
        
        # rerank
        reranked_nodes = local_reranker.rerank_nodes(query, final_nodes)
        chunk_list = [node.text for node, _ in reranked_nodes]
        rerank_end = time.perf_counter()
        global_statistic.add_to_list("rerank_nodes_time", rerank_end - score_end)

        response = Settings.llm.complete(query_prompt(chunk_list, query))
        qa_end = time.perf_counter()
        global_statistic.add_to_list("qa_time", qa_end - rerank_end)
        return response.text