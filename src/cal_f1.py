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
    pred_answers = load_answers(pred_file, is_reference=False)
    total_f1 = 0.0
    count = 0

    for id in ref_answers:
        if id not in pred_answers:
            continue

        max_f1 = 0.0
        for ref_tokens in ref_answers[id]:
            current_f1 = f1_score(pred_answers[id], ref_tokens)
            max_f1 = max(max_f1, current_f1)

        print(f"ID {id}: {max_f1:.4f}")
        total_f1 += max_f1
        count += 1

    if count > 0:
        print(f"\nAverage F1: {total_f1 / count:.4f}")
    else:
        print("No matching answers")
