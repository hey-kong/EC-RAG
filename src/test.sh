#!/bin/bash


# chunking
python3 chunking.py \
    --chunk_size 512 \
    --persist_dir ../docs_store

# run

# basic vec search
python3 run.py \
    --generation_file ../generations/basic.jsonl \
    --similarity_top_k 20 &> ../test_logs/basic.log

# hybrid search
python3 run.py \
    --generation_file ../generations/basic_hybrid.jsonl \
    --similarity_top_k 10 \
    --enable_bm25_retriever \
    --bm25_similarity_top_k 4 &> ../test_logs/basic_hybrid.log

# pruning
python3 run.py \
    --generation_file ../generations/test_pruning.jsonl \
    --similarity_top_k 20 \
    --rerank_top_k 20 \
    --pruning_strategy Naive \
    --no_generate &> ../test_logs/test_prompt_pruning.log

# basic hybrid
python3 run.py \
    --generation_file ../generations/basic_hybrid_10_10_8.jsonl \
    --similarity_top_k 10 \
    --enable_bm25_retriever \
    --bm25_similarity_top_k 10 \
    --rerank_top_k 8 &> ../test_logs/basic_hybrid_10_10_8.log


# dynamic_pruning
python3 run.py \
    --generation_file ../generations/dynamic_pruning.jsonl \
    --similarity_top_k 10 \
    --enable_bm25_retriever \
    --bm25_similarity_top_k 10 \
    --pruning_strategy dynamic &> ../test_logs/dynamic_pruning.log

# dynamic_pruning with local llm
python3 run.py \
    --generation_file ../generations/dynamic_pruning.jsonl \
    --similarity_top_k 10 \
    --enable_bm25_retriever \
    --bm25_similarity_top_k 10 \
    --pruning_strategy dynamic \
    --use_local_llm_for_query &> ../test_logs/dynamic_pruning.log

# use local llm for query
python3 run.py \
    --generation_file ../generations/use_local_llm_hybrid_10_10_8.jsonl \
    --similarity_top_k 10 \
    --enable_bm25_retriever \
    --bm25_similarity_top_k 10 \
    --rerank_top_k 8 \
    --use_local_llm_for_query &> ../test_logs/use_local_llm_hybrid_10_10_8.log

# test choose local llm and reranker
python3 run.py \
    --generation_file ../generations/hotpotqa/test_choose_local_llm_reranker.jsonl \
    --similarity_top_k 10 \
    --enable_bm25_retriever \
    --bm25_similarity_top_k 10 \
    --rerank_top_k 8 \
    --use_local_llm_for_query &> ../test_logs/test_choose_local_llm_reranker.log