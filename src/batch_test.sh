#!/bin/bash
dataset="qasper"

# chunking
python3 chunking.py \
    --embedding_model "/data/wk/models/bge-small-en-v1.5" \
    --chunk_size 512 \
    --chunk_overlap 10 \
    --dataset_name $dataset \
    --docs_dir "../data/${dataset}/documents" \
    --persist_dir "../docs_store"

# run
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


for k in {1..10}
do
    echo "Testing with rerank_top_k=${k}"
    python3 run.py \
        --embedding_model /data/wk/models/bge-small-en-v1.5 \
        --query_file ../data/${dataset}/questions/questions.jsonl \
        --generation_file ../generations/${dataset}/use_local_llm_hybrid_16_16_${k}.jsonl \
        --answer_file ../data/${dataset}/answers/answers.jsonl \
        --local_llm_model_path /data/wk/models/Meta-Llama-3.1-8B-Instruct \
        --use_local_llm_for_query \
        --docstore ../docs_store/${dataset}_512 \
        --similarity_top_k 16 \
        --enable_bm25_retriever \
        --bm25_similarity_top_k 16 \
        --reranker_layerwise \
        --rerank_top_k $k \
        --pruning_strategy None &> "../test_logs/${dataset}/use_local_llm_hybrid_16_16_${k}.log"
    
    # fetch statistics
    python3 fetch_statistics.py "../test_logs/${dataset}/use_local_llm_hybrid_16_16_${k}.log" >> $summary_file

    echo "Completed test with rerank_top_k=${k}"
done
