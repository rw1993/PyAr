# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com
import numpy as np
import copy
from base_predicter import Predicter
from predicter_mixin import PredicterMixin

class ArPredicter(Predicter, PredicterMixin, ):

    def __init__(self, p, max_x, learning_rate=0.003, w_range=1):
        super(ArPredicter, self).__init__()
        self.p = p
        self.ws = [0 for i in range(self.p)]
        self.w_range = w_range
        self.learning_rate = learning_rate
        self.impute = True
        self.min_ob = self.p
        self.no_missing_in_min_ob = True

    def predict(self):
        past_p_xs = np.array(self.xs[-self.p:])
        ws = np.array(self.ws)
        return ws.dot(past_p_xs)

    def fit(self, predict_x, ob_x):
        past_p_xs = self.xs[-self.p:]
        deltas = map(lambda x:x*(predict_x-ob_x), past_p_xs)
        ws_deltas = zip(self.ws, deltas)
        def update(w_d):
            result = w_d[0] - self.learning_rate * w_d[1]
            if abs(result) > self.w_range:
                return self.w_range if result > 0 else - self.w_range
            return result
        self.ws = map(update, ws_deltas)
