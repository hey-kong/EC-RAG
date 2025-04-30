import time

from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import (
    StorageContext,
    QueryBundle,
    load_index_from_storage,
)
from llama_index.retrievers.bm25 import BM25Retriever
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

        # build bm25 retriever
        self.docstore = SimpleDocumentStore.from_persist_path(args.docstore + "_docstore.pkl")
        if args.enable_bm25_retriever:
            self.bm25_retriever = BM25Retriever.from_defaults(
                docstore=self.docstore,  # 直接复用 docstore
                similarity_top_k=args.bm25_similarity_top_k,
                stemmer=Stemmer.Stemmer("english"),
                language="english",
            )

        # pruning strategy
        self.pruning_strategies = ['topk', 'dynamic']

    def bm25_retrieve(self, query_text):
        start = time.perf_counter()
        query_bundle = QueryBundle(query_str=query_text)
        nodes = self.bm25_retriever.retrieve(query_bundle)
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

    def rrf(self, bm25_nodes, vec_nodes, k=60):
        start = time.perf_counter()
        from collections import defaultdict

        score_dict = defaultdict(float)
        node_dict = dict()

        for rank, node in enumerate(bm25_nodes):
            score_dict[node.node_id] += 1 / (k + rank)
            node_dict[node.node_id] = node

        for rank, node in enumerate(vec_nodes):
            score_dict[node.node_id] += 1 / (k + rank)
            node_dict[node.node_id] = node

        fused_node_ids = sorted(score_dict.keys(), key=lambda nid: score_dict[nid], reverse=True)
        fused_nodes = [node_dict[nid] for nid in fused_node_ids]

        end = time.perf_counter()
        global_statistic.add_to_list("rrf_time", end - start)
        return fused_nodes

    def dynamic_pruning(self, reranked_nodes, query_text, min_k):
        start = time.perf_counter()
        pruned_pos = self._find_pruned_pos(reranked_nodes, query_text, min_k)
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

    def _find_pruned_pos(self, reranked_nodes, query_text, min_k):
        n = len(reranked_nodes)
        if n == 0:
            return 0
        if min_k <= 0:
            raise ValueError("min_k must be >= 1")
        if n <= min_k:
            return n

        i = min_k
        step = 2  # step by 2
        while i < n:
            preload_node = reranked_nodes[i + step] if i + step < n and self.args.preload_kvcache else None
            if not slm.judge_relevance(reranked_nodes[i], query_text, self.args.use_kvcache, preload_node):
                break
            i += step

        # If the first checked chunk is irrelevant, stop there
        if i == min_k:
            return i

        # Check the previous chunk's relevance
        j = i - 1
        preload_node = reranked_nodes[0] if self.args.preload_kvcache else None
        if j < n and slm.judge_relevance(reranked_nodes[j], query_text, self.args.use_kvcache, preload_node):
            return i
        return j
