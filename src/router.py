import time
import random
from abc import ABC, abstractmethod

from slm_inference import slm
from customed_statistic import global_statistic
from matrix_factorization.model import MFModel


def complexity_threshold(x, min_val, max_val):
    if max_val <= min_val:
        raise ValueError("max_val must be greater than min_val")
    return -0.5 * (x - min_val) / (max_val - min_val) + 0.5


class BaseRouter(ABC):
    @abstractmethod
    def route_to_edge(self, **kwargs):
        pass


class AdaptiveRouter(BaseRouter):
    def route_to_edge(self, query, n, min_val, max_val, **kwargs):
        start = time.perf_counter()
        complexity_score = slm.judge_complexity(query)
        thre = complexity_threshold(n, min_val, max_val)
        global_statistic.add_to_list("routing_time", time.perf_counter() - start)
        return complexity_score < thre


class SLMRouter(BaseRouter):
    def route_to_edge(self, query, **kwargs):
        complexity_score = slm.judge_complexity(query)
        return complexity_score < 0.5


class RandomRouter(BaseRouter):
    def route_to_edge(self, **kwargs):
        prob = kwargs.get("prob", 0.5)
        return random.uniform(0, 1) < prob


class MFRouter(BaseRouter):
    def __init__(self):
        self.router = MFModel()
        self.router.load("/data/models/mf_model.pth")

    def route_to_edge(self, query, **kwargs):
        return self.router.pred_win_rate(query) < 0.5


ROUTER_CLS = {
    "adaptive": AdaptiveRouter,
    "slm_only": SLMRouter,
    "random": RandomRouter,
    "mf": MFRouter,
}
