#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com


dataset="hotpotqa"

generation_dir="../generations/${dataset}"
output_dir="../test_logs/${dataset}"
summary_file="${output_dir}/summary.log"

# check if dir exists, if not, create it
if [ ! -d "$generation_dir" ]; then
    mkdir -p $generation_dir
fi

if [ ! -d "$output_dir" ]; then
    mkdir -p $output_dir
fi

# run
python3 run_chunk_rag.py \
    --embedding_model BAAI/bge-small-en-v1.5 \
    --query_file ../data/${dataset}/questions/questions.jsonl \
    --num_questions 20 \
    --generation_file ../generations/${dataset}/chunk_rag_basic_chunk.jsonl \
    --answer_file ../data/${dataset}/answers/answers.jsonl \
    --docstore /pan/wk/chunkrag_docs_store/chunkrag_${dataset}_512 \
    --vec_topk 20 \
    --bm25_topk 20 \
    --rerank_top_k 8 \
    --detailed_logging &> "../test_logs/${dataset}/chunk_rag_basic_chunk.log"

echo "Completed test with chunk RAG"