import json
import re
import string
from collections import Counter

"""
This code references: https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py
"""


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def load_answers(file_path: str, is_reference: bool):
    answers = {}
    with open(file_path) as f:
        for line in f:
            data = json.loads(line)
            if is_reference:
                answers[data["id"]] = [normalize_answer(a) for a in data["answers"]]
            else:
                processed_answer = data["answer"].replace('**', '').replace('\n', ' ')
                answers[data["id"]] = normalize_answer(processed_answer)
    return answers


def calc_f1_score(ref_file: str, pred_file: str):
    ref_answers = load_answers(ref_file, is_reference=True)

    total_f1, total_count = 0.0, 0
    edge_f1, cloud_f1 = 0.0, 0.0
    edge_count, cloud_count = 0, 0

    with open(pred_file) as f:
        for line in f:
            data = json.loads(line)
            id = data["id"]
            if id not in ref_answers:
                continue

            pred_answer = normalize_answer(data["answer"].replace('**', '').replace('\n', ' '))
            location = data.get("generation_location", "unknown")

            max_f1 = 0.0
            for ref_tokens in ref_answers[id]:
                current_f1 = f1_score(pred_answer, ref_tokens)
                max_f1 = max(max_f1, current_f1)

            print(f"ID {id} [{location}]: {max_f1:.4f}")

            # 总体统计
            total_f1 += max_f1
            total_count += 1

            # 分 location 统计
            if location == "edge":
                edge_f1 += max_f1
                edge_count += 1
            elif location == "cloud":
                cloud_f1 += max_f1
                cloud_count += 1

    # 避免除以 0 的保护
    avg_total = total_f1 / total_count if total_count else 0.0
    avg_edge = edge_f1 / edge_count if edge_count else 0.0
    avg_cloud = cloud_f1 / cloud_count if cloud_count else 0.0

    print(f"Overall F1: {avg_total:.4f} ({total_count}) | "
          f"Edge F1: {avg_edge:.4f} ({edge_count}) | "
          f"Cloud F1: {avg_cloud:.4f} ({cloud_count})")
