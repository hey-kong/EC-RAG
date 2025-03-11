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
        self.pruning_strategies = ['Naive', 'rrf_dynamic']
        if args.pruning_strategy == 'rrf_dynamic':
            if args.enable_bm25_retriever == False:
                exit("rrf retriever requires bm25 retriever")

    def retrieve(self, query_text):
        if self.args.pruning_strategy != 'None':
            return self._retrieve_pruning(query_text)       # 带有剪枝的retrieve
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

        elif self.args.pruning_strategy == 'rrf_dynamic':
            return self._rrf_dynamic_pruning_retrieve(query_text)
        else:
            exit("Invalid pruning strategy")

    def _basic_retrieve(self, query_text):
        """
        返回nodes列表
        """
        query_bundle = QueryBundle(query_str=query_text)
        
        nodes = []
        bm25_node_ids = set()   # 用于去重

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
            if node.node_id not in bm25_node_ids:   # 去重
                nodes.append(node)
        global_statistic.add_to_list("vec_retriever_time", time.perf_counter() - end)
        global_statistic.add_to_list("vec_retrieved_nodes", len(vec_retrieved_nodes))
        
        # check logic
        if len(nodes) == 0:
            exit("No chunk retrieved")
        return nodes
    
    def _rrf_dynamic_pruning_retrieve(self, query_text):
        """
        融合 rrf + rerank + 动态剪枝
        """
        # check args
        if self.args.enable_bm25_retriever == False:
            exit("rrf retriever requires bm25 retriever")

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
            if node.node_id not in bm25_ranking:   # 去重
                nodes.append(node)
                vec_ranking.append(node.node_id)
        global_statistic.add_to_list("vec_retriever_time", time.perf_counter() - end)
        global_statistic.add_to_list("vec_retrieved_nodes", len(vec_retrieved_nodes))

        # check logic
        if len(nodes) == 0 or len(vec_ranking) == 0 or len(bm25_ranking) == 0:
            exit("No chunk retrieved")
        
        # rrf fusion
        rankings = [bm25_ranking, vec_ranking]
        start = time.perf_counter()
        rrf_ranking = rrf_fusion(rankings)
        end = time.perf_counter()
        global_statistic.add_to_list("rrf_fusion_time", end - start)

        # rerank: list(node, score)
        reranked_nodes = local_reranker.rerank_nodes_with_scores(query_text, nodes)
        global_statistic.add_to_list("rerank_time", time.perf_counter() - end)


        # dynamic pruning
        # pruning range in reranked_nodes
        min_k = 2
        max_k = 10
        # check if pruning is needed
        if len(reranked_nodes) <= max_k:
            return [node.text for node, _ in reranked_nodes]

        # step 1: 遍历reranked_nodes中的得分，计算出每个node与前一个node得分的差值并降序排序（包含node_id）
        diff_scores = []
        for i in range(min_k, min(max_k + 1, len(reranked_nodes))):
            diff_scores.append((reranked_nodes[i][1] - reranked_nodes[i-1][1], reranked_nodes[i][0].node_id, i))
        sorted_diff_scores = sorted(diff_scores, key=lambda x: x[0], reverse=True)

        # step 2: 找到裁剪点
        # 依次检查 sorted_diff_scores，如果满足: 
        #   （1）在rrf中的排名是否在更后面（不包括相等）
        #   （2）rerank_rank排名之前的chunks与在rrf中的排名之后的chunks没有交集
        # 则裁剪掉该 node 以及之后的 nodes
        pruned_pos = min(max_k, len(reranked_nodes))
        intersection_check_cnt = 0
        for _, node_id, rank in sorted_diff_scores:
            if node_id not in rrf_ranking:
                continue
            # get rank in rrf_ranking:
            rrf_rank = rrf_ranking.index(node_id)
            if rrf_rank <= rank or pruned_pos < rank:
                continue
            # check intersection
            rrf_intersection = set(rrf_ranking[rrf_rank:])
            rerank_intersection = set([node.node_id for node, _ in reranked_nodes[:rank]])

            intersection_check_cnt += 1
            global_statistic.add_to_list("rrf_dynamic_pruning_intersection_check_cnt", intersection_check_cnt)
            if len(rerank_intersection.intersection(rrf_intersection)) == 0:
                pruned_pos = min(pruned_pos, rank)

        # step 3: 裁剪
        pruned_chunk_list = [node.text for node, _ in reranked_nodes[:pruned_pos]]
        global_statistic.add_to_list("rrf_dynamic_pruning_pos", len(pruned_chunk_list))
        return pruned_chunk_list
