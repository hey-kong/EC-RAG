import time

from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import (
    StorageContext,
    QueryBundle,
    load_index_from_storage,
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
import Stemmer

from slm_inference import slm
from customed_statistic import global_statistic


class Retriever:
    def __init__(self, args):
        self.args = args

        # build vector retriever
        self.storage_context = StorageContext.from_defaults(persist_dir=args.docstore + "_vec")
        self.vec_index = load_index_from_storage(self.storage_context)
        self.vec_retriever = self.vec_index.as_retriever(similarity_top_k=args.similarity_top_k)

        self.docstore = SimpleDocumentStore.from_persist_path(args.docstore + "_docstore.pkl")
        # build bm25 retriever
        self.bm25_retriever = BM25Retriever.from_defaults(
            docstore=self.docstore,  # 直接复用 docstore
            similarity_top_k=args.bm25_similarity_top_k,
            stemmer=Stemmer.Stemmer("english"),
            language="english",
        )
        # build fusion retriever
        self.fusion_retriever = QueryFusionRetriever(
            [self.vec_retriever, self.bm25_retriever],
            similarity_top_k=args.similarity_top_k + args.bm25_similarity_top_k,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
        )

        # pruning strategy
        self.pruning_strategies = ['topk', 'dynamic']

    def bm25_retrieve(self, query_text):
        start = time.perf_counter()
        nodes = self.bm25_retriever.retrieve(query_text)
        end = time.perf_counter()
        global_statistic.add_to_list("bm25_retrieval_time", end - start)
        global_statistic.add_to_list("bm25_retrieved_nodes", len(nodes))

        if len(nodes) == 0:
            exit("No chunk retrieved")
        return nodes

    def vec_retrieve(self, query_text):
        start = time.perf_counter()
        query_bundle = QueryBundle(query_str=query_text)
        nodes = self.vec_retriever.retrieve(query_bundle)
        end = time.perf_counter()
        global_statistic.add_to_list("vec_retriever_time", end - start)
        global_statistic.add_to_list("vec_retrieved_nodes", len(nodes))

        if len(nodes) == 0:
            exit("No chunk retrieved")
        return nodes

    def fusion_retrieve(self, query_text):
        start = time.perf_counter()
        nodes = self.fusion_retriever.retrieve(query_text)
        nodes = nodes[:self.args.similarity_top_k]
        end = time.perf_counter()
        global_statistic.add_to_list("retrieval_time", end - start)
        global_statistic.add_to_list("retrieved_nodes", len(nodes))

        if len(nodes) == 0:
            exit("No chunk retrieved")
        return nodes

    def dynamic_pruning(self, reranked_nodes, query_text, min_k, max_k):
        start = time.perf_counter()
        pruned_pos = self._find_pruned_pos(reranked_nodes, query_text, min_k, max_k)
        nodes = reranked_nodes[:pruned_pos]
        global_statistic.add_to_list("pruning_time", time.perf_counter() - start)
        global_statistic.add_to_list("avg_chunks", len(nodes))
        return nodes

    # def _find_pruned_pos(self, reranked_nodes, query_text, min_k, step=2):
    #     n = len(reranked_nodes)
    #     if n == 0:
    #         return 0
    #     if min_k <= 0:
    #         raise ValueError("min_k must be >= 1")
    #     if n <= min_k:
    #         return n
    #
    #     i = min_k
    #     while i < n:
    #         if not slm.judge_relevance(reranked_nodes[i][0].node, query_text, self.args.use_kvcache):
    #             break
    #         i += step
    #
    #     start = max(min_k, i - step + 1)
    #     end = min(i, n)
    #     for j in range(start, end):
    #         if not slm.judge_relevance(reranked_nodes[j][0].node, query_text, self.args.use_kvcache):
    #             return j
    #     return end

    def _find_pruned_pos(self, reranked_nodes, query_text, min_k, max_k):
        max_k = min(max_k, len(reranked_nodes))
        if max_k == 0:
            return 0
        if min_k <= 0:
            raise ValueError("min_k must be >= 1")
        if max_k <= min_k:
            return max_k

        i = min_k
        while i < max_k:
            preload_node = reranked_nodes[i + 1] if i + 1 < max_k and self.args.preload_kvcache else None
            if not slm.judge_relevance(reranked_nodes[i], query_text, self.args.use_kvcache, preload_node):
                break
            i += 1

        return i
