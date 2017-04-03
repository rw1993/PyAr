# -*- coding: utf-8 -*-
# author: rw
import random
import mnist_data
import k_fold


class ArPredicter(object):
    

    def __init__(self, mr, B=56, learning_rate=0.03, k=56):
        self.B = B
        self.learning_rate = learning_rate
        self.k = k
        self._w = None
        self.mr = mr
    
    def w_square(self, w):
        return sum(map(lambda x: x*x, w))

    def init(self):
        assert len(self.examples) == len(self.labels)
        self.m = len(self.examples)
        self.d = len(self.examples[0])
        self.Ws = []
        self.init_w()

    def init_w(self):
        w = [random.random() for i in range(self.d)]
        while self.w_square(w) ** 0.5 > self.B:
            print "init w again"
            w = [random.random() for i in range(self.d)]
        self.Ws.append(w)

    def fit(self, examples, labels):
        self.examples = examples
        self.labels = labels
        self.init()
        self.train()
    
    def get_current_X(self, example):
        current_X = [0.0 for f in example]
        indexs = [index for index in range(self.d)]
        indexs = filter(lambda index: example[index]!='*',
                        indexs)
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
        
    def fit(self, e, l):
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
    
    @property
    def w(self):
        if self._w is None:
            self.update_w()
        return self._w

    def update_w(self):
        _w = [0.0 for i in range(self.d)]
        for w in self.Ws:
            _w = map(lambda x,y: x+y, w, _w)
        self._w = map(lambda x: x/len(self.examples), _w)


    def predict(self, X, max_value=1.0):
        def tmp_f(x, y):
            if x == '*':
                return 0
            return x * y
        result = sum(map(tmp_f, X, self.w))
        if result > max_value:
            return max_value
        if result < -max_value:
            return -max_value
        return result

    def predict_label(self, X):
        result = self.predict(X)
        if result > 0.0:
            return 1.0
        else:
            return -1.0
        


if __name__ == "__main__":
    train_set = mnist_data.get_mask_train_set(0.1)
    examples = [x[0] for x in train_set]
    labels = [x[1] for x in train_set]
    print k_fold.k_fold(examples, labels, Regressor)
