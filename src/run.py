import json
import time
import argparse
import os
from tqdm import tqdm

# Llama Index Related
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# custom modules
from llm_inference import generate_answer
from retriever import Retriever
from customed_statistic import global_statistic
from cal_f1 import calc_f1_score
from slm_inference import slm
from reranker import local_reranker
from router import ROUTER_CLS


def check_args(args) -> bool:
    """检查参数有效性"""
    if not os.path.exists(args.query_file):
        print(f"Query file {args.query_file} not found.")
        return False
    if not os.path.exists(args.answer_file):
        print(f"Answer file {args.answer_file} not found.")
        return False
    if not os.path.exists(args.docstore + "_docstore.pkl"):
        print(f"Docstore file {args.docstore} not found.")
        return False
    if not os.path.exists(args.docstore + "_vec"):
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
    parser.add_argument('--embedding_model', type=str, default='BAAI/bge-small-en-v1.5',
                        help='Embedding model name or path')
    parser.add_argument('--query_file', type=str, default='../data/hotpotqa/questions/questions.jsonl',
                        help='Path to the file containing queries')
    parser.add_argument('--num_questions', type=int, default=0, help='Number of questions to process, 0 means all')
    parser.add_argument('--generation_file', type=str, help='Path to the output JSONL file to save generations')
    parser.add_argument('--no_generate', action='store_true', default=False, help='Close generate stage for test')
    parser.add_argument('--answer_file', type=str, default='../data/hotpotqa/answers/answers.jsonl',
                        help='Path to the file containing answers')
    parser.add_argument('--strategy', type=str, default='hybrid', choices=['edge_only', 'cloud_only', 'hybrid'],
                        help="Strategy for selecting inference location: edge, cloud, or hybrid")
    # use local slm
    parser.add_argument('--slm_model_path', type=str, default='LLM-Research/Llama-3.2-3B-Instruct',
                        help='Path of local slm model')
    # retriver related (Basic: vectorIndex)
    parser.add_argument('--docstore', type=str, default='../docs_store/hotpotqa_512', help='Path of nodes')
    parser.add_argument('--similarity_top_k', type=int, default=20, help='Top N of vector retriver')
    parser.add_argument('--enable_bm25_retriever', action='store_true', help='Whether to enable BM25 retriever')
    parser.add_argument('--bm25_similarity_top_k', type=int, default=20, help='Top N of BM25 retriever')
    # reranker related
    parser.add_argument('--reranker_layerwise', action='store_true', help='Whether to use layerwise reranker')
    parser.add_argument('--rerank_top_k', type=int, default=8, help='Top k')
    parser.add_argument('--min_k', type=int, default=2, help='Minimum Top-k used for dynamic pruning')
    parser.add_argument('--max_k', type=int, default=10, help='Maximum Top-k used for dynamic pruning')
    parser.add_argument('--rerank_batch_size', type=int, default=256, help='Rerank batch size')
    # pruning related
    parser.add_argument('--pruning_strategy', type=str, default='dynamic', choices=['topk', 'dynamic'],
                        help="Chunk pruning strategy")
    # routing related
    parser.add_argument('--routing_strategy', type=str, default='adaptive',
                        choices=['adaptive', 'slm_only', 'random', 'mf'],
                        help="Query routing strategy")
    # chunk kvcache
    parser.add_argument('--use_kvcache', action='store_true', help='Whether to use kv cache')
    parser.add_argument('--preload_kvcache', action='store_true', help='Whether to preload kv cache')
    # log related
    parser.add_argument('--detailed_logging', action='store_true', help='Whether to enable detailed logging')
    parser.add_argument('--estimate_cost', action='store_true', help='Whether to estimate cost of cloud llm api')
    args = parser.parse_args()
    if not check_args(args):  # 检查参数有效性
        return
    print_cmd(parser, args)

    # prepare stage
    global_statistic.init(args)  # 初始化统计模块
    slm.init(args.slm_model_path)
    local_reranker.init(args)  # 初始化reranker
    print("Loading index...")
    # Set up embedding model and load index
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.embedding_model)
    start = time.perf_counter()
    retriever = Retriever(args)
    end = time.perf_counter()
    global_statistic.add("retriever_init_time", end - start)

    router = None
    if args.strategy == "hybrid":
        router_cls = ROUTER_CLS[args.routing_strategy]
        router = router_cls()

    # running stage
    print("Running benchmark...")
    questions = []
    with open(args.query_file, 'r', encoding='utf-8') as file:
        for item in file:
            item = json.loads(item)
            questions.append(item)
    if 0 < args.num_questions < len(questions):
        questions = questions[:args.num_questions]
    global_statistic.add("num_questions", len(questions))

    # Clear the file before writing new results
    with open(args.generation_file, 'w', encoding='utf-8'):
        pass  # just open in write mode to truncate the file

    # Now open in append mode and start writing
    with open(args.generation_file, 'a', encoding='utf-8') as file:
        edge_count = 0
        cloud_count = 0
        for item in tqdm(questions):
            query = item["query"]

            # retrieve and generate
            start = time.perf_counter()
            bm25_nodes = retriever.bm25_retrieve(query)
            vec_nodes = retriever.vec_retrieve(query)
            nodes = retriever.rrf(bm25_nodes, vec_nodes)
            nodes = local_reranker.rerank_nodes(query, nodes)
            n = len(nodes)
            complexity_score = None
            if args.strategy == "hybrid" and args.routing_strategy in ("adaptive", "slm_only"):
                preload_node = None
                if args.preload_kvcache:
                    if args.pruning_strategy == "dynamic" and args.min_k < len(nodes):
                        preload_node = nodes[args.min_k][0].node
                    else:
                        preload_node = nodes[0][0].node
                complexity_score = slm.judge_complexity(query, preload_node)
            if args.pruning_strategy == "dynamic":
                nodes = retriever.dynamic_pruning(nodes, query, args.min_k)
            elif args.pruning_strategy == "topk":
                nodes = nodes[:min(len(nodes), args.rerank_top_k)]
            k = len(nodes)
            if not args.no_generate:
                to_edge = False
                if args.strategy == "edge_only":
                    to_edge = True
                elif args.strategy == "cloud_only":
                    to_edge = False
                elif args.strategy == "hybrid":
                    to_edge = router.route_to_edge(
                        query=query,
                        complexity_score=complexity_score,
                        k=k,
                        min_k=args.min_k,
                        max_k=n
                    )

                if to_edge:
                    answer = slm.generate_answer(query, nodes, args.use_kvcache)
                    edge_count += 1
                    generation_location = "edge"
                else:
                    chunk_list = [node.text for node in nodes]
                    answer = generate_answer(query, chunk_list, args.estimate_cost)
                    cloud_count += 1
                    generation_location = "cloud"

                result = {
                    "id": item["id"],
                    "answer": answer,
                    "num_chunks": k,
                    "generation_location": generation_location
                }
                file.write(json.dumps(result, ensure_ascii=False) + '\n')
                end = time.perf_counter()
                global_statistic.add_to_list("rag_time", end - start)
                global_statistic.add("edge_count", edge_count)
                global_statistic.add("cloud_count", cloud_count)

        # end = time.perf_counter()
        # use_time = end - start
        # avg_time = use_time / len(questions)
        # global_statistic.add("rag_avg_time", avg_time)

    global_statistic.dump()
    calc_f1_score(args.answer_file, args.generation_file)


if __name__ == "__main__":
    main()
