# EC-RAG

This repository contains the implementation code for the paper "EC-RAG: Towards Efficient Edge-Cloud Retrieval-Augmented Generation Systems" (accepted by ICDE 2026).

### Environment Setup

Create a virtual environment using `uv` and install the project in editable mode:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

### Running Steps

#### Step 1: Indexing

To preprocess and index the document collection, run:

```
python3 chunking.py \
    --dataset_name ${dataset} \
    --docs_dir "../data/${dataset}/documents" \
    --persist_dir "../docs_store" \
    --chunk_kvcache_dir "../chunk_kvcache" \
    --save_kvcache
```

If you do not wish to use prefix chunk caching, simply run:

```
python3 chunking.py \
    --dataset_name ${dataset} \
    --docs_dir "../data/${dataset}/documents" \
    --persist_dir "../docs_store"
```

#### Step 2: Testing

Default Models:

- Embedding Model: bge-small-en-v1.5
- Reranker: bge-reranker-v2-m3
- Edge SLM: Qwen3-4B
- Cloud LLM: DeepSeek-V3

Before running, make sure to set your cloud LLM API key:

```
export LLM_API_KEY="YOUR_API_KEY"
```

To run the main benchmarking script with prefix chunk caching enabled, execute:

```
python3 run.py \
    --query_file ../data/${dataset}/questions/questions.jsonl \
    --generation_file ${generation_file} \
    --answer_file ../data/${dataset}/answers/answers.jsonl \
    --docstore ../docs_store/${dataset} \
    --pruning_strategy dynamic \
    --min_k 2 \
    --max_k 10 \
    --routing_strategy adaptive \
    --strategy hybrid \
    --use_kvcache \
    --preload_kvcache
```

If you do not wish to use prefix chunk caching, run:

```
python3 run.py \
    --query_file ../data/${dataset}/questions/questions.jsonl \
    --generation_file ${generation_file} \
    --answer_file ../data/${dataset}/answers/answers.jsonl \
    --docstore ../docs_store/${dataset} \
    --pruning_strategy dynamic \
    --min_k 2 \
    --max_k 10 \
    --routing_strategy adaptive \
    --strategy hybrid
```
