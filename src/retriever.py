from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    QueryBundle,
    load_index_from_storage,
)
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer

import time

# custom module
from rerank import rerank_chunks
from customed_statistic import global_statistic
from pruning.core import judge_relevance

class CustomedRetriever:
    def __init__(self, args):
        self.args = args

        # build vector retriever
        self.storage_context = StorageContext.from_defaults(persist_dir=args.docstore+"_vec")
        self.vec_index = load_index_from_storage(self.storage_context)
        self.vec_retriever = self.vec_index.as_retriever(similarity_top_k=args.similarity_top_k)

        # build bm25 retriever
        self.docstore = SimpleDocumentStore.from_persist_path(args.docstore+"_docstore.pkl")
        if args.enable_bm25_retriever:
            self.bm25_retriever = BM25Retriever.from_defaults(
                docstore=self.docstore,   # 直接复用 docstore
                similarity_top_k=args.bm25_similarity_top_k,
                stemmer=Stemmer.Stemmer("english"),
                language="english",
            )
        
        # pruning strategy
        self.pruning_strategies = ['Naive']

    def retrieve(self, query_text):
        if self.args.pruning_strategy != 'None':
            return self._retrieve_pruning(query_text)
        else:
            return self._basic_retrieve(query_text)

    def _retrieve_pruning(self, query_text):
        if self.args.pruning_strategy not in self.pruning_strategies:
            exit("Invalid pruning strategy")
        
        # naive pruning: 遍历所有chunk判定相关性
        if self.args.pruning_strategy == 'Naive':
            chunk_list = self._basic_retrieve(query_text)
            pruned_chunk_list = []
            for chunk in chunk_list:
                relevance, score = judge_relevance(chunk, query_text)
                if relevance:
                    pruned_chunk_list.append(chunk)
                global_statistic.add_to_list("relevance_score", score)
            return pruned_chunk_list
        else:
            exit("Invalid pruning strategy")

    def _basic_retrieve(self, query_text):
        query_bundle = QueryBundle(query_str=query_text)
        
        chunk_list = []
        bm25_node_ids = set()   # 用于去重

        start = time.perf_counter()
        # bm25 retriever
        if self.args.enable_bm25_retriever:
            bm25_retrieved_nodes = self.bm25_retriever.retrieve(query_bundle)
            for node in bm25_retrieved_nodes:
                chunk_list.append(node.text)
                bm25_node_ids.add(node.node_id)
            global_statistic.add_to_list("bm25_retrieved_nodes", len(bm25_retrieved_nodes))
        end = time.perf_counter()
        global_statistic.add_to_list("bm25_retriever_time", end - start)

        # vector retriever
        vec_retrieved_nodes = self.vec_retriever.retrieve(query_bundle)
        for node in vec_retrieved_nodes:
            if node.node_id not in bm25_node_ids:   # 去重
                chunk_list.append(node.text)
        global_statistic.add_to_list("vec_retriever_time", time.perf_counter() - end)
        global_statistic.add_to_list("vec_retrieved_nodes", len(vec_retrieved_nodes))
        
        # check logic
        if len(chunk_list) == 0:
            exit("No chunk retrieved")
        
        # rerank
        start = time.perf_counter()
        chunk_list = rerank_chunks(query_text, chunk_list, self.args.rerank_top_k)
        global_statistic.add_to_list("rerank_time", time.perf_counter() - start)

        return chunk_list