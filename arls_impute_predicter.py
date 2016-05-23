# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com
import numpy as np
import copy
from sklearn import linear_model
from arma import ARMA
import random


class ArPredicter(object, ):

    def __init__(self, p, time_series, missing_indexs,
                 stop_error=0.0005, max_iter_time=6):
        self.p = p
        self.time_series = time_series
        self.missing_indexs = missing_indexs
        self.max_iter_time = max_iter_time
        self.stop_error = stop_error
        self.fit()

    def fit(self):
        iter_time = 0
        while iter_time < self.max_iter_time:
            clf, error = self.get_clf()
            if error < self.stop_error:
                self.clf = clf
                return
            iter_time += 1
            print iter_time,"th complete"
        self.clf = clf
        return

    def get_clf(self):
        def get_x_y(time_series):
            index = 0
            while index + self.p < len(time_series):
                x = time_series[index:index+self.p]
                y = time_series[index+self.p]
                index += 1
                yield x,y

        xs_ys = [x_y for x_y in get_x_y(self.time_series)]
        xs = [x for x,y in xs_ys]
        ys = [y for x,y in xs_ys]
        clf = linear_model.LinearRegression()
        print "training"
        clf.fit(xs, ys)
        def impute(index):
            xs = self.time_series[index-self.p:index]
            predict_y = clf.predict(xs)
            error = abs(self.time_series[index] - predict_y)
            self.time_series[index] = predict_y[0]
            return error
        print "imputing"
        errors = map(impute, self.missing_indexs)
        return clf, max(errors)
    
    @property
    def ses(self):
        ses = []
        for index, x in enumerate(self.time_series):
            if index < self.p:
                continue
            if index in self.missing_indexs:
                continue
            xs = self.time_series[index-self.p:index]
            y = self.clf.predict(xs)
            ses.append((y-self.time_series[index])**2)
        return ses


if __name__ == "__main__":
    a = ARMA([0.3, -0.4, 0.4, -0.5, 0.6], [], 0.1)
    time_series = [a.generater.next() for i in range(4000)]
    missing_rate = 0.1
    if_missing = [1 if random.random() > missing_rate or time_series.index(t) < 6 else 0 for t in time_series]
    time_series = [t if flag==1 else 0 for t,flag in zip(time_series,if_missing)]
    missing_indexs = [index for index, flag in enumerate(if_missing) if flag == 0]
    p = ArPredicter(5, time_series, missing_indexs)
    print sum(p.ses)/4000
