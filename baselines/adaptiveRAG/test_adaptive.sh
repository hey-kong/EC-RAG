#!/bin/bash
export DEEPSEEK_API_KEY="your-key"

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
python3 run_adaptive_rag.py \
    --embedding_model BAAI/bge-small-en-v1.5 \
    --query_file ../data/${dataset}/questions/questions.jsonl \
    --num_questions 20 \
    --generation_file ../generations/${dataset}/adaptive_rag.jsonl \
    --answer_file ../data/${dataset}/answers/answers.jsonl \
    --docstore /pan/docs_store/${dataset}_512 \
    --similarity_top_k 20 \
    --rerank_top_k 8 \
    --detailed_logging &> "../test_logs/${dataset}/adaptive_rag.log"

echo "Completed test with adaptive RAG"