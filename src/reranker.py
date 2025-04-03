import heapq
from itertools import islice
from FlagEmbedding import FlagReranker, LayerWiseFlagLLMReranker


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

    def rerank_chunks(self, query_text, chunk_list, top_k=8):
        """重排序文本片段"""
        pairs = [(query_text, chunk) for chunk in chunk_list]

        # 根据模型类型调用不同的计算方式
        if self.is_layerwise:
            scores = self.reranker.compute_score(pairs, cutoff_layers=[28])
        else:
            scores = self.reranker.compute_score(pairs)

        scored_chunks = list(zip(scores, chunk_list))
        sorted_chunks = sorted(scored_chunks, key=lambda x: x[0], reverse=True)
        return [chunk for (_, chunk) in sorted_chunks[:top_k]]

    def ori_rerank_nodes_with_scores(self, query_text, nodes, top_k=8):
        """重排序节点并返回带分数结果"""
        pairs = [(query_text, node.text) for node in nodes]

        # 根据模型类型调用不同的计算方式
        if self.is_layerwise:
            scores = self.reranker.compute_score(pairs, cutoff_layers=[28])
        else:
            scores = self.reranker.compute_score(pairs)

        topk = heapq.nlargest(top_k, zip(scores, nodes), key=lambda x: x[0])
        return [(node, score) for score, node in topk]

    def rerank_nodes_with_scores(self, query_text, nodes, top_k=8):
        """重排序节点并返回得分最高的 top_k 结果，支持 early stopping"""
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
