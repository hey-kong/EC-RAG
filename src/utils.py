def rrf_fusion(rankings, k=60):
    """
    实现Reciprocal Rank Fusion算法
    :param rankings: 一个包含多个排序列表的列表,每个排序列表是一个文档ID的列表
    :param k: RRF算法中的常数, 默认为60
    :return: 融合后的排序结果, 按得分从高到低排序
    """
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id not in scores:
                scores[doc_id] = 0
            # calc rrf score
            scores[doc_id] += 1 / (k + rank)
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_scores]