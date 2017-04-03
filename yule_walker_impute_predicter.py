# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com
import numpy as np
import copy
from base_predicter import Predicter
from predicter_mixin import PredicterMixin


class ArPredicter(Predicter, PredicterMixin):

    def __init__(self, p, max_x=1.0, learning_rate=0.003):
        super(ArPredicter, self).__init__()
        self.p = p
        self.min_ob = self.p + 1
        self.no_missing_in_min_ob = True
        self._ws = [0 for i in range(self.p)]
        self._r = [0 for i in range(self.p+1)]
        self.N = {}
        self.max_x = max_x
        

    def predict(self):
        past_p_xs = self.xs[-self.p:][::-1]
        try:
            r = self.ws.dot(past_p_xs)
            if r > self.max_x:
                return self.max_x
            if r < -self.max_x:
                return -self.max_x
            return r
        except:
            print "too early"
            return 0

    def fit(self, pre_x, ob_x):
        if ob_x == '*':
            self.xs.append(pre_x)
            return
        self.xs.append(ob_x)
        for i in range(self.p+1):
            self.update_r(i)

    def update_r(self, i):
        if i >= len(self.xs):
            return 
        N = self.N.get(i, 0)
        r_i = N * self._r[i]
        r_i += self.xs[-1] * self.xs[-1-i]
        r_i = r_i / float(N + 1)
        self._r[i] = r_i
        self.N[i] = N + 1

    @property
    def ws(self):
        rs_left = [self._r[i+1] for i in range(self.p)]
        rs_right = [self._r[i] for i in range(self.p)]
        r_matrix = [rs_right,]
        for i in range(self.p - 1):
            rs_right = [rs_right[-1]] + rs_right[:-1]
            r_matrix.append(rs_right)
        r_matrix =np.array(r_matrix)
        rs_left = np.array(rs_left)
        return np.linalg.inv(r_matrix).dot(rs_left)
