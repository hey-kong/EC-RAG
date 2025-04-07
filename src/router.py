import random
import time

from slm_inference import slm
from customed_statistic import global_statistic


def route_to_edge(query, n, min_val, max_val):
    start = time.perf_counter()
    complexity_score = slm.judge_complexity(query)
    global_statistic.add_to_list("judge_complexity_time", time.perf_counter() - start)
    return complexity_score < complexity_threshold(n, min_val, max_val)


def complexity_threshold(x, min_val, max_val):
    if max_val <= min_val:
        raise ValueError("max_val must be greater than min_val")
    return -0.5 * (x - min_val) / (max_val - min_val) + 0.5


def random_route(prob=0.5):
    return random.random() < prob
