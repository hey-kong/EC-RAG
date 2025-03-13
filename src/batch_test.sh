#!/bin/bash
dataset="qasper"
python3 chunking.py \
    --embedding_model "/data/wk/models/bge-small-en-v1.5" \
    --chunk_size 512 \
    --chunk_overlap 10 \
    --dataset_name $dataset \
    --docs_dir "../data/${dataset}/documents" \
    --persist_dir "../docs_store"


for k in {1..10}
do
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
