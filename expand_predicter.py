# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com
import numpy as np
from base_predicter import Predicter
from predicter_mixin import PredicterMixin


class ArPredicter(Predicter, PredicterMixin ):

    def __init__(self, p, missing_ability=3, max_x=1.0,
                 learning_rate=0.03):
        super(ArPredicter, self).__init__()
        self.p = p
        self.d = int(p * missing_ability)
        self.D = max_x * (2 ** (self.d / 2))
        self.G = self.D
        self.ws = np.array([0 for i in range(2 ** self.d -1)])
        self.learning_rate = learning_rate
        self.last = np.array([0 for i in range(2 ** self.d -1)])
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

        def bin_represent(past_d_xs):
            return [1 if x != '*' else 0 for x in past_d_xs]

        b_xs = bin_represent(past_d_xs)

        for i in range(1, 2 ** self.d):
            bi = [int(num) for num in str(bin(i)[2:])]
            locate = len(bi)
            compare = b_xs[-len(bi):]
            if compare[0] == 0:
                expand_xs.append(0)
                continue
            if_add = True
            for index, (num1, num2) in enumerate(zip(compare, bi)):
                if index == 0:
                    pass
                else:
                    if num1 > num2:
                        if_add = False
                        continue
            if if_add:
                expand_xs.append(past_d_xs[-locate])
            else:
                expand_xs.append(0)
        return np.array(expand_xs)


if __name__ == "__main__":
    p = ArmaPredicter(1, 4, 1)
    expand_xs = p.expand_xs(['5','*',3,2,1])
