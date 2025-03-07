import json
import time
import argparse
import os
from tqdm import tqdm

# Llama Index Related
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# custom modules
from generate import generate_answer
from retriever import CustomedRetriever
from customed_statistic import global_statistic

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
# logging.INFO("RAG Benchmarking Script")
def check_args(args) -> bool:
    """检查参数有效性"""
    if not os.path.exists(args.embedding_model):
        print(f"Embedding model {args.embedding_model} not found.")
        return False
    if not os.path.exists(args.query_file):
        print(f"Query file {args.query_file} not found.")
        return False
    if not os.path.exists(args.docstore+"_docstore.pkl"):
        print(f"Docstore file {args.docstore} not found.")
        return False
    if not os.path.exists(args.docstore+"_vec"):
        print(f"Vector store dir {args.docstore} not found.")
        return False
    # mkdir for answer_file if necessary
    answer_dir = os.path.dirname(args.answer_file)
    if not os.path.exists(answer_dir):
        os.makedirs(answer_dir)
    return True



def main():
    # Parse command-line arguments at global scope
    parser = argparse.ArgumentParser(description='RAG Benchmarking Script')
    parser.add_argument('--embedding_model', type=str, default='../models/bge-small-en-v1.5', help='Embedding model name or path')
    parser.add_argument('--query_file', type=str, default='../data/hotpotqa/questions/questions.jsonl',
                        help='Path to the file containing queries')
    parser.add_argument('--num_questions', type=int, default=0, help='Number of questions to process, 0 means all')
    parser.add_argument('--answer_file', type=str, default='../generations/generations.jsonl', help='Path to the output JSONL file to save answers.')
    parser.add_argument('--no_generate', type=bool, default=False, help='Close generate stage for test')
    # retriver related (Basic: vectorIndex)
    parser.add_argument('--docstore', type=str, default='../chunking_data/hotpotqa_512', help='Path of nodes')
    parser.add_argument('--similarity_top_k', type=int, default=20, help='Top N of vector retriver')
    parser.add_argument('--enable_bm25_retriever', type=bool, default=False, help='Whether to enable BM25 retriever')
    parser.add_argument('--bm25_similarity_top_k', type=int, default=4, help='Top N of BM25 retriever')
    # reranker related
    parser.add_argument('--rerank_top_k', type=int, default=8, help='Top k')
    # pruning related
    parser.add_argument('--pruning_strategy', type=str, default='None', help='Pruning strategy: None, Naive')
    args = parser.parse_args()
    if not check_args(args):     # 检查参数有效性
        return

    # prepare stage
    global_statistic.init(args)     # 初始化统计模块
    print("Loading index...")
    # Set up embedding model and load index
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.embedding_model)
    start = time.perf_counter()
    customed_retriever = CustomedRetriever(args)
    end = time.perf_counter()
    global_statistic.add("retriever_init_time", end - start)

    # running stage
    print("Running benchmark...")
    questions = []
    with open(args.query_file, 'r', encoding='utf-8') as file:
        for item in file:
            item = json.loads(item)
            questions.append(item)
    if args.num_questions > 0 and args.num_questions < len(questions):
        questions = questions[:args.num_questions]
    global_statistic.add("num_questions", len(questions))

    with open(args.answer_file, 'a', encoding='utf-8') as file:
        start = time.perf_counter()
        for item in tqdm(questions):
            query = item["query"]
            
            # retrieve(include rerank and pruning) and generate
            chunk_list = customed_retriever.retrieve(query)
            if not args.no_generate:
                answer, n = generate_answer(chunk_list, query)
                result = {"id": item["id"], "answer": answer, "num_chunks": n}
                file.write(json.dumps(result, ensure_ascii=False) + '\n')

        end = time.perf_counter()
        use_time = end - start
        avg_time = use_time / len(questions)
        global_statistic.add("rag_avg_time", avg_time)

    global_statistic.dump()


if __name__ == "__main__":
    main()
