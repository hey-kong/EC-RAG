from FlagEmbedding import LayerWiseFlagLLMReranker

reranker = LayerWiseFlagLLMReranker('BAAI/bge-reranker-v2-minicpm-layerwise', use_fp16=True, cutoff_layers=[28])


def rerank_chunks(query_text, chunk_list, top_k=8):
    pairs = [(query_text, chunk) for chunk in chunk_list]
    scores = reranker.compute_score(pairs, cutoff_layers=[28])
    scored_chunks = list(zip(scores, chunk_list))
    sorted_chunks = sorted(scored_chunks, key=lambda x: x[0], reverse=True)
    return [chunk for (_, chunk) in sorted_chunks[:top_k]]

def rerank_nodes_with_scores(query_text, nodes):
    """
    return: a list of tuples (node, score)
    """
    pairs = [(query_text, node.text) for node in nodes]
    scores = reranker.compute_score(pairs, cutoff_layers=[28])
    scored_nodes = list(zip(scores, nodes))
    sorted_nodes = sorted(scored_nodes, key=lambda x: x[0], reverse=True)
    return [(node, score) for score, node in sorted_nodes]


