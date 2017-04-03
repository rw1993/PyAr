# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com
import numpy as np
from numpy.linalg import inv
import copy
import pykalman
from base_predicter import Predicter
from predicter_mixin import PredicterMixin
import pdb


class ArPredicter(Predicter, PredicterMixin, ):

    @property
    def ws(self):
        return self.X

    def __init__(self, p, w_range=1.0, max_x=1.0):
        super(ArPredicter, self).__init__()
        self.p = p
        self.min_ob = self.p + 1
        self.max_x = max_x
        self.x = np.matrix(0.0001 * np.random.random(p)).T
        self.phi = 1
        self.P = np.matrix(999999*np.eye(p))
    
    @property
    def M(self):
        return np.matrix(self.xs[-self.p:])
        
    def predict(self):
        pre_x = self.x
        #print self.x
        pre_y = self.M.dot(pre_x)
        print pre_y
        return pre_y[0, 0]


    def fit(self, pre_y, ob_y):
        K = self.P.dot(self.M.T) / (self.M.dot(self.P).dot(self.M.T))
        if ob_y != '*':
            self.x = self.x + K * (ob_y - pre_y)
        self.P = self.P - K.dot(self.M).dot(self.P)
        if ob_y == '*':
            self.xs.append(pre_y)
        else:
            self.xs.append(ob_y)


if __name__ == "__main__":
    p = ArPredicter(1)
