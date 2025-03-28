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
from reranker import local_reranker
from customed_statistic import global_statistic
from local_llm_inference.core import local_llm
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
        self.pruning_strategies = ['Naive', 'dynamic']
        if args.pruning_strategy == 'dynamic':
            if args.enable_bm25_retriever == False:
                exit("bm25 retriever should be enabled")

    def retrieve(self, query_text):
        if self.args.pruning_strategy != 'None':
            return self._retrieve_pruning(query_text)  # 带有剪枝的retrieve
        else:
            # 默认策略：retrieve + rerank
            nodes = self._basic_retrieve(query_text)
            chunk_list = [node.text for node in nodes]
            start = time.perf_counter()
            chunk_list = local_reranker.rerank_chunks(query_text, chunk_list, self.args.rerank_top_k)
            global_statistic.add_to_list("rerank_time", time.perf_counter() - start)
            return chunk_list

    def _retrieve_pruning(self, query_text):
        if self.args.pruning_strategy not in self.pruning_strategies:
            exit("Invalid pruning strategy")

        # naive pruning: 遍历所有chunk判定相关性
        if self.args.pruning_strategy == 'Naive':
            # basic retrieve + rerank
            nodes = self._basic_retrieve(query_text)
            chunk_list = [node.text for node in nodes]
            start = time.perf_counter()
            chunk_list = local_reranker.rerank_chunks(query_text, chunk_list, self.args.rerank_top_k)
            global_statistic.add_to_list("rerank_time", time.perf_counter() - start)

            # Naive pruning
            pruned_chunk_list = []
            for chunk in chunk_list:
                relevance, score = local_llm.judge_relevance(chunk, query_text)
                if relevance:
                    pruned_chunk_list.append(chunk)
                global_statistic.add_to_list("relevance_score", score)
            return pruned_chunk_list

        elif self.args.pruning_strategy == 'dynamic':
            return self._dynamic_pruning_retrieve(query_text)
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
        # check args
        if self.args.enable_bm25_retriever == False:
            exit("retriever requires bm25 retriever")

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
        global_statistic.add_to_list("bm25_retriever_time", end - start)

        # vector retriever
        vec_ranking = []
        vec_retrieved_nodes = self.vec_retriever.retrieve(query_bundle)
        for node in vec_retrieved_nodes:
            if node.node_id not in bm25_ranking:  # 去重
                nodes.append(node)
                vec_ranking.append(node.node_id)
        global_statistic.add_to_list("vec_retriever_time", time.perf_counter() - end)
        global_statistic.add_to_list("vec_retrieved_nodes", len(vec_retrieved_nodes))

        # check logic
        if len(nodes) == 0:
            exit("No chunk retrieved")

        # rrf fusion
        # rankings = [bm25_ranking, vec_ranking]
        # start = time.perf_counter()
        # rrf_ranking = rrf_fusion(rankings)
        # end = time.perf_counter()
        # global_statistic.add_to_list("rrf_fusion_time", end - start)

        # rerank: list(node, score)
        start = time.perf_counter()
        reranked_nodes = local_reranker.rerank_nodes_with_scores(query_text, nodes)
        global_statistic.add_to_list("rerank_time", time.perf_counter() - start)

        # dynamic pruning
        # pruning range in reranked_nodes
        min_k = 2
        max_k = 10
        pruned_pos = self._find_pruned_pos(reranked_nodes, query_text, min_k, max_k)
        pruned_chunk_list = [node.text for node, _ in reranked_nodes[:pruned_pos]]
        global_statistic.add_to_list("dynamic_pruning_pos", len(pruned_chunk_list))
        return pruned_chunk_list

    def _find_pruned_pos(self, reranked_nodes, query_text, min_k, max_k):
        n = len(reranked_nodes)
        if n <= min_k:
            return n

        left = min_k
        right = min(max_k, n)
        last_true_index = min_k - 1

        while right > left:
            # 找到 score gap 最大的位置 i
            max_gap = 0
            split_index = left
            for i in range(left, right):
                gap = reranked_nodes[i - 1][1] - reranked_nodes[i][1]
                if gap > max_gap:
                    max_gap = gap
                    split_index = i

            chunk = reranked_nodes[split_index][0].text
            if local_llm.judge_relevance(chunk, query_text):
                last_true_index = split_index
                left = split_index + 1  # 保留当前，向下继续裁剪
            else:
                right = split_index  # 丢弃当前及之后，向上继续裁剪

        return last_true_index + 1
