import json
import time
import argparse
import os
from tqdm import tqdm

# custom modules
from adaptive_rag import AdaptiveRAG
from customed_statistic import global_statistic
from cal_f1 import calc_f1_score
from reranker import local_reranker


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
    command_lines = ["python3 run_adaptive_rag.py"]

    for action in parser._actions:
        if not action.option_strings:  # 跳过位置参数
            continue
        if action.dest == "help":
            continue

        option = max(action.option_strings, key=lambda x: len(x))
        value = getattr(args, action.dest)

        if isinstance(value, bool):
            value = str(value)
            if value == "True":
                command_lines.append(f"    {option}")
            continue

        command_lines.append(f"    {option} {value}")
    formatted_command = " \\\n".join(command_lines)
    print(f"Command:\n{formatted_command}")


def main():
    # Parse command-line arguments at global scope
    parser = argparse.ArgumentParser(description='adaptive rag Benchmarking Script')
    parser.add_argument('--embedding_model', type=str, default='BAAI/bge-small-en-v1.5',
                        help='Embedding model name or path')
    parser.add_argument('--query_file', type=str, default='../data/hotpotqa/questions/questions.jsonl',
                        help='Path to the file containing queries')
    parser.add_argument('--num_questions', type=int, default=0, help='Number of questions to process, 0 means all')
    parser.add_argument('--generation_file', type=str, help='Path to the output JSONL file to save generations')
    parser.add_argument('--answer_file', type=str, default='../data/hotpotqa/answers/answers.jsonl',
                        help='Path to the file containing answers')

    # retriver related (Basic: vectorIndex)
    parser.add_argument('--docstore', type=str, default='../docs_store/hotpotqa_512', help='Path of nodes')
    parser.add_argument('--similarity_top_k', type=int, default=20, help='Top N of vector retriver')

    # reranker related
    parser.add_argument('--reranker_layerwise', action='store_true', help='Whether to use layerwise reranker')
    parser.add_argument('--rerank_top_k', type=int, default=8, help='Top k')

    # log related
    parser.add_argument('--detailed_logging', action='store_true', help='Whether to enable detailed logging')
    parser.add_argument('--estimate_cost', action='store_true', help='Whether to estimate cost of cloud llm api')
    args = parser.parse_args()
    if not check_args(args):  # 检查参数有效性
        return
    print_cmd(parser, args)


    local_reranker.init(args)
    agent = AdaptiveRAG(
        index_dir=args.docstore + "_vec",
        similarity_topk=args.similarity_top_k,
        embedding_model=args.embedding_model,
    )
    global_statistic.init(args)

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

    direct_qa_cnt = 0
    single_step_qa_cnt = 0
    multi_step_qa_cnt = 0

    with open(args.generation_file, 'a', encoding='utf-8') as file:
        for item in tqdm(questions):
            query = item["query"]

            # retrieve(include rerank and pruning) and generate
            start = time.perf_counter()
            answer, selected_tool_index = agent.query(query)
            end = time.perf_counter()

            global_statistic.add_to_list("rag_time", end - start)
            result = {
                "id": item["id"],
                "answer": answer,
                "selected_tool_index": selected_tool_index
            }
            file.write(json.dumps(result, ensure_ascii=False) + '\n')

            if selected_tool_index == 0:
                direct_qa_cnt += 1
            elif selected_tool_index == 1:
                single_step_qa_cnt += 1
            elif selected_tool_index == 2:
                multi_step_qa_cnt += 1

    global_statistic.add("direct_qa_cnt", direct_qa_cnt)
    global_statistic.add("single_step_qa_cnt", single_step_qa_cnt)
    global_statistic.add("multi_step_qa_cnt", multi_step_qa_cnt)
    global_statistic.dump()
    calc_f1_score(args.answer_file, args.generation_file)

if __name__ == "__main__":
    main()
