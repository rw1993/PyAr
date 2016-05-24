# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com
import e_expand_predicter 


class ArPredicter(e_expand_predicter.ArPredicter, ):

    def __init__(self, p, missing_ability=3.0, max_x=1.0,
                 learning_rate=0.003):
        super(ArPredicter, self).__init__(p=p,
                                          missing_ability=missing_ability,
                                          max_x=max_x,
                                          learning_rate=learning_rate)
        self.hit = 0
        self.total = 0
        self.cache = {}

    def K(self, s, t):
        self.total += 1
        s_xs = self.xs[s:s+self.d][::-1]
        t_xs = self.xs[t:t+self.d][::-1]
        assert len(s_xs) == len(t_xs)
        k = 0
        def to_bin(xs):
            return [0 if x =='*' else 1 for x in xs]
        s_bin_xs = to_bin(s_xs)
        t_bin_xs = to_bin(t_xs)
        def to_key(xs):
            key = "".join(str(i) for i in xs)
            return key
        key1 = to_key(s_bin_xs)
        key2 = to_key(t_bin_xs)
        f_key = (key1, key2)
        s_key = (key2, key1)
        if self.cache.has_key(f_key):
            cs = self.cache[f_key]
            self.hit += 1
        elif self.cache.has_key(s_key):
            cs = self.cache[s_key]
            self.hit += 1
        else:
            cs = []
            for index, s_t in enumerate(zip(s_bin_xs, t_bin_xs)):
                if s_t[0] == 0:
                    c = -1
                    cs.append(c)
                    continue
                if s_t[1] == 0:
                    c = -1
                    cs.append(c)
                    continue
                c = 0
                for s_t1 in zip(s_xs, t_xs)[:index]:
                    if s_t1[0] == s_t1[1] == '*':
                        c += 1
                cs.append(c)
            self.cache[s_key] = cs
        for c, x, x1 in zip(cs, s_xs, t_xs):
            if c == -1:
                continue
            k += 2 ** c * x * x1
        print float(self.hit) / float(self.total)
        return k
