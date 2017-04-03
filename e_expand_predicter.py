# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com
from base_predicter import Predicter
from predicter_mixin import PredicterMixin


class ArPredicter(Predicter, PredicterMixin, ):

    def __init__(self, p, missing_ability=3.0, max_x=1.0,
                 learning_rate=0.03):
        super(ArPredicter, self).__init__()
        self.p = p
        self.d = int(missing_ability*p)
        self.min_ob = self.d
        self.predict_xs = [0,]
        self.Errs = []
        self._learning_rate = learning_rate
        self.max_x = max_x
        self.norm = 0
        self.f = self.d

    @property
    def learning_rate(self):
        return self._learning_rate
        return 1.0 / (float(self.f) ** 0.5) * self._learning_rate

    def predict(self):
        return self.predict_xs[-1]

    def fit(self, pre_x, ob_x):
        if ob_x == '*':
            Err = 0
        else:
            Err = pre_x - ob_x
            self.f += 1
        self.xs.append(ob_x)
        self.Errs.append(Err)
        up = 0.0
        t = len(self.Errs)
        for index, err in enumerate(self.Errs):
            up = up + err * self.K(index, t)
        up = - up * self.learning_rate
        if up > self.max_x:
            self.predict_xs.append(1)
        elif up < -self.max_x:
            self.predict_xs.append(-1)
        else:
            self.predict_xs.append(up)

    def K(self, s, t):
        s_xs = self.xs[s:s+self.d][::-1]
        t_xs = self.xs[t:t+self.d][::-1]
        assert len(s_xs) == len(t_xs)
        k = 0
        for index, s_t in enumerate(zip(s_xs, t_xs)):
            if s_t[0] == '*':
                continue
            if s_t[1] == '*':
                continue
            c = 0
            for s_t1 in zip(s_xs, t_xs)[:index]:
                if s_t1[0] == s_t1[1] == '*':
                    c += 1
            k += 2 ** c * s_t[0] * s_t[1]
        return k
