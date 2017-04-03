# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com
import numpy as np
import copy
from base_predicter import Predicter
from predicter_mixin import PredicterMixin
import random

class ArPredicter(Predicter, PredicterMixin, ):

    def __init__(self, p, max_x, learning_rate=0.3, w_range=1,
                 B = 20.0, mr=0.0):
        super(ArPredicter, self).__init__()
        self.p = p
        self.d = self.p
        self.k = self.p
        self.w = [random.random()*0.1 for i in range(self.p)]
        self.Ws = [self.w]
        self.w_range = w_range
        self.learning_rate = learning_rate
        self.min_ob = self.p
        self.no_missing_in_min_ob = True
        self.max_x = max_x
        self.B = B
        self.mr = mr
        self.train_xs = None

    def w_square(self, w):
        return sum(map(lambda x: x*x, w))

    def predict(self):
        past_p_xs = np.array(self.xs[-self.p:])
        ws = np.array(self.w)
        r = ws.dot(past_p_xs)
        if r > self.max_x:
            return self.max_x
        if r < - self.max_x:
            return -self.max_x
        return r

    def fit(self, predict_x, ob_x):
        if self.train_xs is None:
            self.train_xs = [x for x in self.xs]
        if ob_x == '*':
            self.xs.append(predict_x)
            self.train_xs.append(ob_x)
            return
        past_p_xs = self.train_xs[-self.p:]
        self.train(past_p_xs, ob_x)
        self.xs.append(ob_x)
        self.train_xs.append(ob_x)

    def get_current_X(self, example):
        current_X = [0.0 for f in example]
        indexs = [index for index in range(self.d)]
        indexs = filter(lambda index: example[index]!='*',
                        indexs)
        if len(indexs) == 0:
            return [0.0 for i in range(self.d)]
        for i in range(self.k):
            index = random.choice(indexs)
            current_X[index] += example[index] * self.d
        current_X = map(lambda x: x/self.k, current_X)
        return current_X

    def get_index_w(self, current_w):
        w_square = sum(map(lambda x: x*x, current_w))
        w_percents = map(lambda x: x*x/w_square, current_w)
        index = 0
        r_number = random.random()
        s = 0
        while True:
            s = s + w_percents[index]
            if s > r_number:
                return index
            index += 1
    
    def get_new_w(self, v):
        B = self.B
        v_square = sum(map(lambda x: x**2, v)) ** 0.5
        if v_square > self.B:
            self.Ws.append(map(lambda x: x*B/v_square, v))
        else:
            self.Ws.append(v)

        num_ws = len(self.Ws) - 1
        sum_w = map(lambda x: x*num_ws, self.w)
        self.w = map(lambda x, y: (x+y)/(num_ws + 1),
                     self.w, sum_w)
        
    def train(self, e, l):
        current_W = self.Ws[-1]
        current_X = self.get_current_X(e)
        w_index = self.get_index_w(current_W)
        w_square = self.w_square(current_W)
        if e[w_index] == '*':
            e[w_index] = 0.0
        o = w_square * e[w_index] / current_W[w_index] / (1-self.mr) - l
        # o = w_square * e[w_index] / current_W[w_index] - l
        gs = map(lambda x: x*o, current_X)
        v = map(lambda w, g: w-self.learning_rate*g, current_W, gs)
        self.get_new_w(v)
