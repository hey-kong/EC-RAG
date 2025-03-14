import json
import time
import argparse
import os
import sys
from tqdm import tqdm
import random


# Llama Index Related
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# custom modules
from generate import generate_answer
from retriever import CustomedRetriever
from customed_statistic import global_statistic
from cal_f1 import calc_f1_score
from local_llm_inference.core import local_llm
from reranker import local_reranker

def check_args(args) -> bool:
    """检查参数有效性"""
    if not os.path.exists(args.query_file):
        print(f"Query file {args.query_file} not found.")
        return False
    if not os.path.exists(args.answer_file):
        print(f"Answer file {args.answer_file} not found.")
        return False
    if not os.path.exists(args.docstore+"_docstore.pkl"):
        print(f"Docstore file {args.docstore} not found.")
        return False
    if not os.path.exists(args.docstore+"_vec"):
        print(f"Vector store dir {args.docstore} not found.")
        return False
    # mkdir for generation_file if necessary
    answer_dir = os.path.dirname(args.generation_file)
    if not os.path.exists(answer_dir):
        os.makedirs(answer_dir)
    return True

def print_cmd(parser, args):
    # 输出用户输入的命令（方便复制重新运行）
    # 生成可复用的完整命令
    command_lines = ["python3 run.py"]  # 假设脚本固定名称，可按需替换为 sys.argv[0]
    
    # 遍历所有参数定义
    for action in parser._actions:
        if not action.option_strings:  # 跳过位置参数
            continue
        # 跳过默认生成的help参数
        if action.dest == "help":
            continue

        # 获取参数名称和值
        option = max(action.option_strings, key=lambda x: len(x))  # 取最长参数名
        value = getattr(args, action.dest)
        
        # 特殊处理布尔值
        if isinstance(value, bool):
            value = str(value)
            # bool类型参数，不带值
            if value == "True":
                command_lines.append(f"    {option}")
            continue
            
        command_lines.append(f"    {option} {value}")
    # 格式化为带换行的命令
    formatted_command = " \\\n".join(command_lines)
    print(f"Command:\n{formatted_command}")

def main():
    # Parse command-line arguments at global scope
    parser = argparse.ArgumentParser(description='RAG Benchmarking Script')
    parser.add_argument('--embedding_model', type=str, default='BAAI/bge-small-en-v1.5', help='Embedding model name or path')
    parser.add_argument('--query_file', type=str, default='../data/hotpotqa/questions/questions.jsonl',
                        help='Path to the file containing queries')
    parser.add_argument('--num_questions', type=int, default=0, help='Number of questions to process, 0 means all')
    parser.add_argument('--generation_file', type=str, help='Path to the output JSONL file to save generations.')
    parser.add_argument('--no_generate', action='store_true', default=False, help='Close generate stage for test')
    parser.add_argument('--answer_file', type=str, default='../data/hotpotqa/answers/answers.jsonl', help='Path to the file containing answers')
    # use local llm
    parser.add_argument('--local_llm_model_path', type=str, default='LLM-Research/Meta-Llama-3.1-8B-Instruct', help='Path of local llm model')
    parser.add_argument('--use_local_llm_for_query', action='store_true', default=False, help='Whether to use local llm for query')
    # retriver related (Basic: vectorIndex)
    parser.add_argument('--docstore', type=str, default='../docs_store/hotpotqa_512', help='Path of nodes')
    parser.add_argument('--similarity_top_k', type=int, default=20, help='Top N of vector retriver')
    parser.add_argument('--enable_bm25_retriever', action='store_true', default=False, help='Whether to enable BM25 retriever')
    parser.add_argument('--bm25_similarity_top_k', type=int, default=4, help='Top N of BM25 retriever')
    # reranker related
    parser.add_argument('--reranker_layerwise', action='store_true', default=True, help='Whether to use layerwise reranker')
    parser.add_argument('--rerank_top_k', type=int, default=8, help='Top k')
    # pruning related
    parser.add_argument('--pruning_strategy', type=str, default='None', help='Pruning strategy: None, Naive, rrf_dynamic')
    # log related
    parser.add_argument('--detailed_logging', action='store_true', default=False, help='Whether to enable detailed logging')
    parser.add_argument('--estimate_cost', action='store_true', default=False, help='Whether to estimate cost of cloud llm api')
    args = parser.parse_args()
    if not check_args(args):     # 检查参数有效性
        return
    print_cmd(parser, args)

    # prepare stage
    global_statistic.init(args)     # 初始化统计模块
    # 本地模型非必须
    if args.use_local_llm_for_query or args.pruning_strategy == 'Naive':
        local_llm.init(args.local_llm_model_path)
    local_reranker.init(args.reranker_layerwise)     # 初始化reranker
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

    with open(args.generation_file, 'a', encoding='utf-8') as file:
        for item in tqdm(questions):
            query = item["query"]
            
            # retrieve(include rerank and pruning) and generate
            start = time.perf_counter()
            chunk_list = customed_retriever.retrieve(query)
            if not args.no_generate:
                if args.use_local_llm_for_query:
                    n = len(chunk_list)
                    answer = local_llm.generate_answer(chunk_list, query)
                    result = {"id": item["id"], "answer": answer, "num_chunks": n}
                    file.write(json.dumps(result, ensure_ascii=False) + '\n')
                    end = time.perf_counter()
                    global_statistic.add_to_list("rag_time", end - start)
                else:
                    answer, n = generate_answer(chunk_list, query, args.estimate_cost)
                    result = {"id": item["id"], "answer": answer, "num_chunks": n}
                    file.write(json.dumps(result, ensure_ascii=False) + '\n')
                    end = time.perf_counter()
                    global_statistic.add_to_list("rag_time", end - start)
                    # sleep for cloud llm api
                    # 请求过于频繁经常性地会出错, 暂停随机一段时间
                    # sleep_time = random.randint(2, 8)
                    # time.sleep(sleep_time)

        # end = time.perf_counter()
        # use_time = end - start
        # avg_time = use_time / len(questions)
        # global_statistic.add("rag_avg_time", avg_time)

    global_statistic.dump()
    calc_f1_score(args.answer_file, args.generation_file)

if __name__ == "__main__":
    main()
