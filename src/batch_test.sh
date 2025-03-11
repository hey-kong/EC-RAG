#!/bin/bash

# 测试范围1-10，跳过已测试的8
for k in 1 2 3 4 5 6 7 8 9 10; do
    echo "Testing with rerank_top_k=${k}"
    python3 run.py \
        --generation_file "../generations/use_local_llm_hybrid_16_16_${k}.jsonl" \
        --similarity_top_k 16 \
        --enable_bm25_retriever True \
        --bm25_similarity_top_k 16 \
        --rerank_top_k $k \
        --use_local_llm_for_query True &> "../test_logs/use_local_llm_hybrid_16_16_${k}.log"
        
    echo "Completed test with rerank_top_k=${k}"
done
