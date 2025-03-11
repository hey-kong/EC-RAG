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
    --similarity_top_k 16 \
    --enable_bm25_retriever True \
    --bm25_similarity_top_k 4 &> ../test_logs/basic_hybrid.log

# pruning
python3 run.py \
    --generation_file ../generations/test_pruning.jsonl \
    --similarity_top_k 20 \
    --num_questions 2 \
    --rerank_top_k 20 \
    --pruning_strategy Naive \
    --no_generate True &> ../test_logs/test_prompt_pruning.log

# basic hybrid
python3 run.py \
    --generation_file ../generations/basic_hybrid_16_16_8.jsonl \
    --similarity_top_k 16 \
    --enable_bm25_retriever True \
    --bm25_similarity_top_k 16 \
    --rerank_top_k 8 &> ../test_logs/basic_hybrid_16_16_8.log


# rrf_dynamic_pruning
python3 run.py \
    --generation_file ../generations/rrf_dynamic_pruning.jsonl \
    --similarity_top_k 16 \
    --enable_bm25_retriever True \
    --bm25_similarity_top_k 16 \
    --pruning_strategy rrf_dynamic \
    --detailed_logging False &> ../test_logs/rrf_dynamic_pruning.log

# use local llm for query
python3 run.py \
    --generation_file ../generations/use_local_llm_hybrid_16_16_8.jsonl \
    --similarity_top_k 16 \
    --enable_bm25_retriever True \
    --bm25_similarity_top_k 16 \
    --rerank_top_k 8 \
    --use_local_llm_for_query True &> ../test_logs/use_local_llm_hybrid_16_16_8.log

# test choose local llm and reranker
python3 run.py \
    --generation_file ../generations/test_choose_local_llm_reranker.jsonl \
    --similarity_top_k 16 \
    --enable_bm25_retriever True \
    --bm25_similarity_top_k 16 \
    --rerank_top_k 8 \
    --use_local_llm_for_query True \
    --num_questions 10 &> ../test_logs/test_choose_local_llm_reranker.log