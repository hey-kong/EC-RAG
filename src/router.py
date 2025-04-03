import time

from slm_inference import slm
from customed_statistic import global_statistic


def is_complex(query, chunk_list):
    start = time.perf_counter()
    complexity_score = slm.judge_complexity(query)
    global_statistic.add_to_list("judge_complexity_time", time.perf_counter() - start)
    return complexity_score > 1.0 / len(chunk_list)
