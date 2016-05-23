# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com
import numpy as np
from base_predicter import Predicter
from predicter_mixin import PredicterMixin


class ArPredicter(Predicter, PredicterMixin, ):

    def __init__(self, p, missing_ability=3, max_x=1.0, learning_rate=0.003):
        self.p = p
        self.d = int(p * missing_ability)
        self.w_len = (1 + self.d) * self.d / 2
        self.D = max_x * (self.w_len ** 0.5)
        self.G = self.D
        self.ws = np.array([0 for i in range(self.w_len)])
        self.xs = []
        self.errors = []
        self.learning_rate = learning_rate
        self.last = np.array([0 for i in range(self.w_len)])
        self.min_ob = self.d

    def fit(self, predict_x, ob_x):
        if ob_x == '*':
            self.xs.append(ob_x)
            return
        past_d_xs = self.xs[-self.d:]
        expand_xs = self.expand_xs(past_d_xs)
        self.last = self.last + (predict_x - ob_x) * expand_xs
        self.ws = -self.learning_rate * self.last
        norm = np.linalg.norm(self.last)
        self.ws = self.ws / max(1.0, self.learning_rate / self.D * norm)
        self.xs.append(ob_x)
        
    def predict(self):
        past_d_xs = self.xs[-self.d:]
        expand_xs = self.expand_xs(past_d_xs)
        return self.ws.dot(expand_xs)
        

    def expand_xs(self, past_d_xs):
        expand_xs = []
        missing_count = 0
        for i in range(len(past_d_xs)):
            this_x = past_d_xs[i]
            if this_x == '*':
                xs_for_expand = [0 for j in past_d_xs[i:]]
                expand_xs += xs_for_expand
                continue
            sub_sequece = past_d_xs[i:]
            xs_for_expand = [0 if x != '*' else this_x for x in sub_sequece]
            xs_for_expand[0] = this_x 
            expand_xs += xs_for_expand
        return np.array(expand_xs)
