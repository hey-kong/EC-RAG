import json
import re
from typing import Dict, List, Set


def tokenize(text: str) -> Set[str]:
    return set(re.findall(r'\w+', text.lower()))


def compute_f1(pred_tokens: Set[str], ref_tokens: Set[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = pred_tokens & ref_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * (precision * recall) / (precision + recall + 1e-9)


def load_answers(file_path: str, is_reference: bool) -> Dict[int, List[Set[str]]]:
    answers = {}
    with open(file_path) as f:
        for line in f:
            data = json.loads(line)
            if is_reference:
                answers[data["id"]] = [tokenize(a) for a in data["answers"]]
            else:
                answers[data["id"]] = tokenize(data["answer"])
    return answers

def calc_f1_score(ref_file: str, pred_file: str):
    ref_answers = load_answers(ref_file, is_reference=True)
    pred_answers = load_answers(pred_file, is_reference=False)
    total_f1 = 0.0
    count = 0

    for id in ref_answers:
        if id not in pred_answers:
            continue

        max_f1 = 0.0
        for ref_tokens in ref_answers[id]:
            current_f1 = compute_f1(pred_answers[id], ref_tokens)
            max_f1 = max(max_f1, current_f1)

        print(f"ID {id}: {max_f1:.4f}")
        total_f1 += max_f1
        count += 1

    if count > 0:
        print(f"\nAverage F1: {total_f1 / count:.4f}")
    else:
        print("No matching answers")


# def main():
#     ref_answers = load_answers('../data/hotpotqa/answers/answers.jsonl', is_reference=True)
#     pred_answers = load_answers('../generations/basic.jsonl', is_reference=False)

#     total_f1 = 0.0
#     count = 0

#     for id in ref_answers:
#         if id not in pred_answers:
#             continue

#         max_f1 = 0.0
#         for ref_tokens in ref_answers[id]:
#             current_f1 = compute_f1(pred_answers[id], ref_tokens)
#             max_f1 = max(max_f1, current_f1)

#         print(f"ID {id}: {max_f1:.4f}")
#         total_f1 += max_f1
#         count += 1

#     if count > 0:
#         print(f"\nAverage F1: {total_f1 / count:.4f}")
#     else:
#         print("No matching answers")

# if __name__ == "__main__":
#     main()
