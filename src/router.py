import random
from slm_inference import slm


def route_to_edge(query, n, min_val, max_val):
    complexity_score = slm.judge_complexity(query)
    return complexity_score < complexity_threshold(n, min_val, max_val)


def complexity_threshold(x, min_val, max_val):
    if max_val <= min_val:
        raise ValueError("max_val must be greater than min_val")
    return -0.5 * (x - min_val) / (max_val - min_val) + 0.5


def random_route(prob=0.5):
    return random.random() < prob
