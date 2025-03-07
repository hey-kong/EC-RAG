from FlagEmbedding import FlagReranker

model = FlagReranker(
    '../models/bge-reranker-v2-m3',
    use_fp16=True,
    devices=["cuda:0"],
)

def rerank_chunks(query_text, chunk_list, top_k=3):
    pairs = [(query_text, chunk) for chunk in chunk_list]
    scores = model.compute_score(pairs)
    scored_chunks = list(zip(scores, chunk_list))
    sorted_chunks = sorted(scored_chunks, key=lambda x: x[0], reverse=True)
    return [chunk for (_, chunk) in sorted_chunks[:top_k]]

def rerank_nodes_with_scores(query_text, chunk_list, top_k=3):
    pairs = [(query_text, chunk) for chunk in chunk_list]
    scores = model.compute_score(pairs)
    scored_chunks = list(zip(scores, chunk_list))
    sorted_chunks = sorted(scored_chunks, key=lambda x: x[0], reverse=True)
    return [(chunk, score) for score, chunk in sorted_chunks[:top_k]]