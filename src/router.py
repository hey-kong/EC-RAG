import random
from abc import ABC, abstractmethod

from slm_inference import slm
from matrix_factorization.model import MFModel


def complexity_threshold(x, min_val, max_val, lam=0.8):
    if max_val <= min_val:
        raise ValueError("max_val must be greater than min_val")
    if lam < 0.5:
        raise ValueError("lambda must be greater than or equal to 0.5")
    return lam - (2 * lam - 1) * ((x - min_val) / (max_val - min_val))


class BaseRouter(ABC):
    @abstractmethod
    def route_to_edge(self, **kwargs):
        pass


class AdaptiveRouter(BaseRouter):
    def route_to_edge(self, query, complexity_score, k, min_k, max_k, **kwargs):
        return complexity_score < complexity_threshold(k, min_k, max_k)


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
        return self.router.pred_win_rate(query) < 0.11593


ROUTER_CLS = {
    "adaptive": AdaptiveRouter,
    "slm_only": SLMRouter,
    "random": RandomRouter,
    "mf": MFRouter,
}
