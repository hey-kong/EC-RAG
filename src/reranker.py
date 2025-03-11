from FlagEmbedding import FlagReranker, LayerWiseFlagLLMReranker


class RerankerWrapper:
    def __init__(self):
        self.reranker = None
        self.is_layerwise = False

    def init(self, use_layerwise: bool):
        """初始化重排序模型
        
        Args:
            use_layerwise (bool): 
                True - 使用LayerWiseFlagLLMReranker
                False - 使用普通FlagReranker
        """
        if use_layerwise:
            # 初始化层间重排序模型
            model_path = 'BAAI/bge-reranker-v2-minicpm-layerwise'
            self.reranker = LayerWiseFlagLLMReranker(
                model_path,
                use_fp16=True,
                cutoff_layers=[28]
            )
        else:
            # 初始化普通重排序模型
            model_path = 'BAAI/models/bge-reranker-v2-m3'
            self.reranker = FlagReranker(
                model_path,
                use_fp16=True,
                devices=["cuda:0"]
            )
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

    def rerank_nodes_with_scores(self, query_text, nodes):
        """重排序节点并返回带分数结果"""
        pairs = [(query_text, node.text) for node in nodes]

        # 根据模型类型调用不同的计算方式
        if self.is_layerwise:
            scores = self.reranker.compute_score(pairs, cutoff_layers=[28])
        else:
            scores = self.reranker.compute_score(pairs)

        scored_nodes = list(zip(scores, nodes))
        sorted_nodes = sorted(scored_nodes, key=lambda x: x[0], reverse=True)
        return [(node, score) for score, node in sorted_nodes]


local_reranker = RerankerWrapper()
