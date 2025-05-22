import time
import heapq
from itertools import islice
from FlagEmbedding import FlagReranker

from customed_statistic import global_statistic
from slm_inference import slm


class RerankerWrapper:
    def __init__(self):
        self.args = None
        self.reranker = None

    def init(self, args):
        model_path = 'BAAI/bge-reranker-v2-m3'
        self.reranker = FlagReranker(
            model_path,
            use_fp16=True,
            devices=["cuda:0"]
        )
        self.args = args
        print(f'use local reranker: {model_path}')

    def rerank_nodes(self, query_text, nodes, top_k=8):
        """重排序节点并返回带分数结果"""
        start = time.perf_counter()
        pairs = [(query_text, node.text) for node in nodes]

        scores = self.reranker.compute_score(pairs)

        topk = heapq.nlargest(top_k, zip(scores, nodes), key=lambda x: x[0])
        end = time.perf_counter()
        global_statistic.add_to_list("reranking_time", end - start)
        return [node for _, node in topk]

    def rerank_nodes_with_early_stopping(self, query_text, nodes, top_k=8):
        """重排序节点并返回得分最高的 top_k 结果，支持 early stopping"""
        start = time.perf_counter()
        heap = []
        processed = 0

        for batch in self._batch_iterable(nodes, top_k):
            pairs = [(query_text, node.text) for node in batch]

            scores = self.reranker.compute_score(pairs)
            if len(heap) == top_k and max(scores) <= heap[0][0]:
                # skipped = len(nodes) - processed - len(pairs)
                # if skipped != 0:
                #     print(f"[Early Stop] Total {len(nodes)} pairs, skipped {skipped} pairs")
                break

            for score, node in zip(scores, batch):
                if len(heap) < top_k:
                    heapq.heappush(heap, (score, -processed, node))
                else:
                    if score > heap[0][0]:
                        heapq.heappushpop(heap, (score, -processed, node))
                processed += 1

            if self.args.preload_kvcache and len(heap) == top_k:
                topk_results = sorted(heap, key=lambda x: (x[0], x[1]), reverse=True)
                preload_node = topk_results[self.args.min_k][2]
                slm.kvcache_loader.preload_kvcache(preload_node.metadata["kvcache_file_path"])

        topk_results = sorted(heap, key=lambda x: (x[0], x[1]), reverse=True)[:top_k]
        end = time.perf_counter()
        global_statistic.add_to_list("reranking_time", end - start)
        return [node for _, _, node in topk_results]

    def _batch_iterable(self, iterable, batch_size):
        it = iter(iterable)
        while True:
            batch = list(islice(it, batch_size))
            if not batch:
                break
            yield batch


local_reranker = RerankerWrapper()
