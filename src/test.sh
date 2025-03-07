#!/bin/bash


# chunking
python3 chunking.py \
    --chunk_size 512 \
    --persist_dir ../docs_store

# run

# basic vec search
python3 run.py \
    --answer_file ../generations/basic.jsonl \
    --similarity_top_k 20 &> ../test_logs/basic.log

# hybrid search
python3 run.py \
    --answer_file ../generations/basic_hybrid.jsonl \
    --similarity_top_k 16 \
    --enable_bm25_retriever True \
    --bm25_similarity_top_k 4 &> ../test_logs/basic_hybrid.log

# pruning
python3 run.py \
    --answer_file ../generations/test_pruning.jsonl \
    --similarity_top_k 20 \
    --num_questions 2 \
    --rerank_top_k 20 \
    --pruning_strategy Naive \
    --no_generate True &> ../test_logs/test_prompt_pruning.log