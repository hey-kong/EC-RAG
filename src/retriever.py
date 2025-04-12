import time

from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import (
    StorageContext,
    QueryBundle,
    load_index_from_storage,
)
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer

from reranker import local_reranker
from customed_statistic import global_statistic
from slm_inference import slm
from utils import (
    rrf_fusion,
)


class CustomedRetriever:
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
        self.pruning_strategies = ['exhaustive', 'dynamic']

    def retrieve(self, query_text):
        if self.args.pruning_strategy != 'topk':
            return self._retrieve_pruning(query_text)  # 带有剪枝的retrieve
        else:
            # 默认策略：retrieve + rerank
            nodes = self._basic_retrieve(query_text)
            start = time.perf_counter()
            reranked_nodes = local_reranker.rerank_nodes(query_text, nodes, self.args.rerank_top_k)
            global_statistic.add_to_list("rerank_time", time.perf_counter() - start)
            nodes = [node for node, _ in reranked_nodes]
            return nodes

    def _retrieve_pruning(self, query_text):
        if self.args.pruning_strategy not in self.pruning_strategies:
            exit("Invalid pruning strategy")

        # exhaustive pruning: 遍历所有chunk判定相关性
        if self.args.pruning_strategy == 'exhaustive':
            # basic retrieve + rerank
            nodes = self._basic_retrieve(query_text)
            start = time.perf_counter()
            nodes = local_reranker.rerank_nodes(query_text, nodes, self.args.rerank_top_k)
            global_statistic.add_to_list("rerank_time", time.perf_counter() - start)

            pruned_nodes = []
            for node in nodes:
                relevance, score = slm.judge_relevance(node, query_text, self.args.use_kvcache)
                if relevance:
                    pruned_nodes.append(node)
                global_statistic.add_to_list("relevance_score", score)
            return pruned_nodes

        elif self.args.pruning_strategy == 'dynamic':
            nodes = self._dynamic_pruning_retrieve(query_text)
            return self._merge_first_sequence(nodes)
        else:
            exit("Invalid pruning strategy")

    def _basic_retrieve(self, query_text):
        """
        返回nodes列表
        """
        query_bundle = QueryBundle(query_str=query_text)

        nodes = []
        bm25_node_ids = set()  # 用于去重

        start = time.perf_counter()
        # bm25 retriever
        if self.args.enable_bm25_retriever:
            bm25_retrieved_nodes = self.bm25_retriever.retrieve(query_bundle)
            for node in bm25_retrieved_nodes:
                nodes.append(node)
                bm25_node_ids.add(node.node_id)
            global_statistic.add_to_list("bm25_retrieved_nodes", len(bm25_retrieved_nodes))
        end = time.perf_counter()
        global_statistic.add_to_list("bm25_retriever_time", end - start)

        # vector retriever
        vec_retrieved_nodes = self.vec_retriever.retrieve(query_bundle)
        for node in vec_retrieved_nodes:
            if node.node_id not in bm25_node_ids:  # 去重
                nodes.append(node)
        global_statistic.add_to_list("vec_retriever_time", time.perf_counter() - end)
        global_statistic.add_to_list("vec_retrieved_nodes", len(vec_retrieved_nodes))

        # check logic
        if len(nodes) == 0:
            exit("No chunk retrieved")
        return nodes

    def _dynamic_pruning_retrieve(self, query_text):
        """
        动态剪枝
        """
        # basic retrieve
        query_bundle = QueryBundle(query_str=query_text)

        nodes = []
        bm25_ranking = []

        start = time.perf_counter()
        # bm25 retriever
        bm25_retrieved_nodes = self.bm25_retriever.retrieve(query_bundle)
        for node in bm25_retrieved_nodes:
            bm25_ranking.append(node.node_id)
        nodes.extend(bm25_retrieved_nodes)
        global_statistic.add_to_list("bm25_retrieved_nodes", len(bm25_retrieved_nodes))
        end = time.perf_counter()
        global_statistic.add_to_list("bm25_retrieval_time", end - start)

        # vector retriever
        vec_ranking = []
        vec_retrieved_nodes = self.vec_retriever.retrieve(query_bundle)
        for node in vec_retrieved_nodes:
            if node.node_id not in bm25_ranking:  # 去重
                nodes.append(node)
                vec_ranking.append(node.node_id)
        global_statistic.add_to_list("vec_retriever_time", time.perf_counter() - end)
        global_statistic.add_to_list("vec_retrieval_nodes", len(vec_retrieved_nodes))

        # check logic
        if len(nodes) == 0:
            exit("No chunk retrieved")

        # rrf fusion
        rankings = [bm25_ranking, vec_ranking]
        rrf_ranking = rrf_fusion(rankings)
        node_id_to_node = {node.node_id: node for node in nodes}
        nodes = [node_id_to_node[node_id] for node_id in rrf_ranking if node_id in node_id_to_node]

        # rerank: list(node, score)
        start = time.perf_counter()
        reranked_nodes = local_reranker.rerank_nodes_with_early_stopping(query_text, nodes, self.args.max_k)
        global_statistic.add_to_list("reranking_time", time.perf_counter() - start)

        # dynamic pruning between min_k and max_k
        start = time.perf_counter()
        pruned_pos = self._find_pruned_pos(reranked_nodes, query_text, self.args.min_k, self.args.max_k)
        nodes = [node for node, _ in reranked_nodes[:pruned_pos]]
        global_statistic.add_to_list("pruning_time", time.perf_counter() - start)
        global_statistic.add_to_list("avg_chunks", len(nodes))
        return nodes

    def _find_pruned_pos(self, reranked_nodes, query_text, min_k, max_k):
        n = len(reranked_nodes)
        if n <= min_k:
            return n

        left = min_k
        right = min(max_k, len(reranked_nodes)) - 1
        last_true_index = min_k - 1
        while left <= right:
            mid = (left + right) // 2
            node = reranked_nodes[mid][0].node

            if slm.judge_relevance(node, query_text, self.args.use_kvcache):
                last_true_index = mid
                left = mid + 1
            else:
                right = mid - 1

        return last_true_index + 1

    def _merge_first_sequence(self, nodes):
        if not nodes:
            return []

        first_node = nodes[0]
        doc_id, chunk_id_str = first_node.node_id.rsplit('_', 1)
        chunk_id = int(chunk_id_str)

        merged_nodes = [first_node]
        used_indices = {0}
        next_chunk_id = chunk_id + 1

        for i in range(1, len(nodes)):
            node = nodes[i]
            node_doc_id, node_chunk_str = node.node_id.rsplit('_', 1)
            node_chunk_id = int(node_chunk_str)

            if node_doc_id == doc_id and node_chunk_id == next_chunk_id:
                merged_nodes.append(node)
                used_indices.add(i)
                next_chunk_id += 1

        remaining_nodes = [nodes[i] for i in range(len(nodes)) if i not in used_indices]
        return merged_nodes + remaining_nodes
