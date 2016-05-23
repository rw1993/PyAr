# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com
import numpy as np
import copy
import pykalman
from base_predicter import Predicter
from predicter_mixin import PredicterMixin

class ArPredicter(Predicter, PredicterMixin, ):

    @property
    def ws(self):
        return self.filter_state_mean

    def __init__(self, p, w_range=1.0, max_x=1.0):
        super(ArPredicter, self).__init__()
        self.p = p
        self.filter_state_mean = [0 for i in range(self.p)]
        self.row = [0 for i in range(self.p)]
        self.filter_state_covariance = filtered_state_covariance = np.array([copy.deepcopy(self.row) for i in range(self.p)])
        for index, row in enumerate(self.filter_state_covariance):
            row[index] = 99999999.0
        self.w_range = w_range
        self.kalman = pykalman.KalmanFilter()
        self.min_ob = self.p

    def predict(self):
        past_p_xs = np.array(self.xs[-self.p:])
        ws = np.array(self.ws)
        return ws.dot(past_p_xs)

    def fit(self, pre_x, ob_x):
        past_p_xs = self.xs[-self.p:]
        filtered_state_mean = np.array(self.ws)
        filtered_state_covariance = self.filter_state_covariance
        transition_matrix = np.array([copy.deepcopy(self.row) for i in range(self.p)])
        for i in range(self.p):
            transition_matrix[i][i] = 1
        observation_matrix = np.array([copy.deepcopy(self.row) for i in range(self.p)])
        for index, x in enumerate(past_p_xs):
            observation_matrix[index][index] = x
        observation = ob_x if ob_x!= '*' else None
        f_s_t, f_s_c = self.kalman.filter_update(filtered_state_mean=filtered_state_mean,
                                                 filtered_state_covariance=filtered_state_covariance,
                                                 observation=observation,
                                                 transition_matrix=transition_matrix,
                                                 observation_matrix=observation_matrix)
        self.filter_state_mean = f_s_t
        self.filter_state_covariance = f_s_c

        if ob_x == '*':
            self.xs.append(pre_x)
        else:
            self.xs.append(ob_x)



if __name__ == "__main__":
    p = ArPredicter(1)
    print p.ws
