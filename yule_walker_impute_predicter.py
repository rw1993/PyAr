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
        self.impute = True
        self.no_missing_in_min_ob = True
        

    def predict(self):
        past_p_xs = self.xs[-self.p:][::-1]
        return self.ws.dot(past_p_xs)

    def fit(self, pre_x, ob_x):
        pass

    @property
    def ws(self):
        c0 = sum([x*x for x in self.xs]) / len(self.xs)
        c0 = float(c0)
        #print 'c0',c0
        xs = self.xs[::-1]

        def r(i):
            x_xis = [x * xs[index+i] for index, x in enumerate(xs) if index+i<len(xs)]
            #if len(x_xis) == 0:
                #print i,xs
            ci = sum(x_xis) / len(x_xis)

            return ci/c0

        rs_left = [r(i+1) for i in range(self.p)]
        rs_right = [r(i) for i in range(self.p)]
        r_matrix = [rs_right,]
        for i in range(self.p - 1):
            rs_right = [rs_right[-1]] + rs_right[:-1]
            r_matrix.append(rs_right)
        r_matrix =np.array(r_matrix)
        rs_left = np.array(rs_left)
        return np.linalg.inv(r_matrix).dot(rs_left)
