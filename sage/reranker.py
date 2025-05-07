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

    def rerank_nodes(self, query_text, nodes, top_k=0):
        """重排序节点并返回带分数结果"""
        if top_k <= 0:
            top_k = self.args.rerank_top_k

        """重排序节点并返回带分数结果"""
        pairs = [(query_text, node.text) for node in nodes]

        # 根据模型类型调用不同的计算方式
        if self.is_layerwise:
            scores = self.reranker.compute_score(pairs, cutoff_layers=[28])
        else:
            scores = self.reranker.compute_score(pairs)

        topk = heapq.nlargest(top_k, zip(scores, nodes), key=lambda x: x[0])
        return [(node, score) for score, node in topk]

local_reranker = RerankerWrapper()