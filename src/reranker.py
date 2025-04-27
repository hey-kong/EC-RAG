import time
import heapq
from itertools import islice
from FlagEmbedding import FlagReranker, LayerWiseFlagLLMReranker

from customed_statistic import global_statistic


class RerankerWrapper:
    def __init__(self):
        self.args = None
        self.reranker = None
        self.is_layerwise = False

    def init(self, args):
        """初始化重排序模型

        Args:
            use_layerwise (bool): 
                True - 使用LayerWiseFlagLLMReranker
                False - 使用普通FlagReranker
        """
        if args.reranker_layerwise:
            # 初始化层间重排序模型
            model_path = 'BAAI/bge-reranker-v2-minicpm-layerwise'
            self.reranker = LayerWiseFlagLLMReranker(
                model_path,
                use_fp16=True,
                devices=["cuda:0"],
                cutoff_layers=[28]
            )
            self.is_layerwise = True
        else:
            # 初始化普通重排序模型
            model_path = 'BAAI/bge-reranker-v2-m3'
            self.reranker = FlagReranker(
                model_path,
                use_fp16=True,
                devices=["cuda:0"]
            )
        self.args = args
        print(f'use local reranker: {model_path}')

    def rerank_nodes(self, query_text, nodes):
        """重排序节点并返回带分数结果"""
        start = time.perf_counter()
        pairs = [(query_text, node.text) for node in nodes]

        if self.is_layerwise:
            scores = self.reranker.compute_score(pairs, cutoff_layers=[28])
        else:
            scores = self.reranker.compute_score(pairs)
        sorted_pairs = sorted(zip(scores, nodes), key=lambda x: x[0], reverse=True)

        end = time.perf_counter()
        global_statistic.add_to_list("reranking_time", end - start)
        return [(node, score) for score, node in sorted_pairs]

    def rerank_nodes_with_early_stopping(self, query_text, nodes, top_k=8):
        """重排序节点并返回得分最高的 top_k 结果，支持 early stopping"""
        start = time.perf_counter()
        pairs = [(query_text, node.text, node) for node in nodes]
        heap = []
        processed = 0

        for batch in self._batch_iterable(pairs, self.args.rerank_batch_size):
            texts_batch = [(q, t) for q, t, _ in batch]
            if self.is_layerwise:
                scores = self.reranker.compute_score(texts_batch, cutoff_layers=[28])
            else:
                scores = self.reranker.compute_score(texts_batch)

            for score, (_, _, node) in zip(scores, batch):
                if len(heap) < top_k:
                    heapq.heappush(heap, (score, processed, node))  # 加入 processed 作为 tie-breaker
                else:
                    if score > heap[0][0]:
                        heapq.heappushpop(heap, (score, processed, node))
                processed += 1

            if len(heap) == top_k and max(scores) <= heap[0][0]:
                # skipped = len(pairs) - processed
                # if skipped != 0:
                #     print(f"[Early Stop] Total {len(pairs)} pairs, skipped {skipped} pairs")
                break

        topk_results = sorted(heap, key=lambda x: x[0], reverse=True)[:top_k]
        end = time.perf_counter()
        global_statistic.add_to_list("reranking_time", end - start)
        return [(node, score) for score, _, node in topk_results]

    def _batch_iterable(self, iterable, batch_size):
        """将 iterable 分批"""
        it = iter(iterable)
        while True:
            batch = list(islice(it, batch_size))
            if not batch:
                break
            yield batch


local_reranker = RerankerWrapper()
