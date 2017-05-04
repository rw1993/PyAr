# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com
import numpy as np
import copy
from base_predicter import Predicter
from predicter_mixin import PredicterMixin


class ArPredicter(Predicter, PredicterMixin):
    name = "YuleWalker"

    def __init__(self, p, max_x=1.0, learning_rate=0.003):
        super(ArPredicter, self).__init__()
        self.p = p
        self.min_ob = self.p + 1
        self.no_missing_in_min_ob = True
        self._ws = np.array([0.0 for i in range(self.p)])
        self.N = {}
        self.max_x = max_x
        self.u = None
        self.var = None
       
    def update_u(self):
        self.u = sum(self.xs) / len(self.xs)

    def update_var(self):
        self.var = sum(map(lambda x: (x-self.u)**2,
                       self.xs)) / (len(self.xs)-1)

    def r(self, i):
        xs = self.xs
        couples = []
        for index, x in enumerate(xs):
            if index + i < len(xs):
                couples.append((x, xs[index+i]))

        return sum(map(lambda x: (x[0]-self.u)*(x[1]-self.u),
                       couples)) / len(couples) / self.var

                

    def predict(self):
        past_p_xs = self.xs[-self.p:][::-1]
        r = self.ws.dot(past_p_xs)

        return r

    def fit(self, pre_x, ob_x):
        if ob_x == '*':
            self.xs.append(pre_x)
            self.update_u()
            self.update_var()
            return
        self.xs.append(ob_x)
        self.update_u()
        self.update_var()

    @property
    def ws(self):
        if len(self.xs) < self.p + 1:
            return self._ws
        if self.u is None or self.var is None:
            return self._ws
        rs = [self.r(i) for i in range(self.p+1)]
        vector = np.array([rs[i] for i in range(1, self.p+1)])
        matrix = np.eye(self.p)
        for i in range(self.p):
            for j in range(i, self.p):
                matrix[i][j] = rs[j-i]
                matrix[j][i] = rs[j-i]
        matrix = np.array(matrix)
        return np.linalg.inv(matrix).dot(vector)
