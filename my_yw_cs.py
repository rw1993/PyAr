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
        self.min_ob = 0
        self.no_missing_in_min_ob = True
        self._ws = np.array([0.0 for i in range(self.p)])
        self.N = {}
        self.max_x = max_x
        self.cs = [[] for i in range(self.p+1)]
       
    def r(self, i):
        ci = sum(self.cs[i])
        c0 = sum(self.cs[0][:len(self.cs[i])])
        return ci / c0
        
    def predict(self):
        past_p_xs = self.xs[-self.p:][::-1]
        if len(past_p_xs) != self.p:
            return 0.0
        r = self.ws.dot(past_p_xs)
        return r
   
    def update_c(self, x):
        self.xs.append(x)
        c_index = 0
        final_index = len(self.xs) - 1
        while final_index - c_index > 0 and c_index < len(self.cs):
            new_add = x * self.xs[final_index-c_index]
            self.cs[c_index].append(new_add)
            c_index += 1


    def fit(self, pre_x, ob_x):
        if ob_x == '*':
            self.update_c(pre_x)
            return
        self.update_c(ob_x)
   
    def check_c(self):
        for c in self.cs:
            if not c:
                return False
        return True

    @property
    def ws(self):
        if not self.check_c():
            return self._ws
        rs = [self.r(i) for i in range(self.p+1)]
        vector = np.array([rs[i] for i in range(1, self.p+1)])
        matrix = []
        for i in range(self.p):
            row = []
            for j in range(self.p):
                row.append(rs[(j+i)%self.p])
            matrix.append(row)
        matrix = np.array(matrix)
        return np.linalg.inv(matrix).dot(vector)
