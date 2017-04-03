# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com
import numpy as np
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
        self.state_cor = np.matrix(np.zeros((self.p, self.p)))
        self.state = np.matrix(np.zeros(self.p)).T
        '''
        self.train_matrix = [[0.0 for i in range(p)]]
        for i in range(p-1):
            new_p = [0.0 for i in range(p)]
            new_p[i] = 1
            self.train_matrix.append(new_p)
        '''
        self.train_matrix = np.matrix(np.zeros((p, p)))
        self.ob_matrix = np.matrix(np.zeros(p))
        self.ob_matrix[0, 0] = 1.0
        self.em_vars=["transition_matrices",
                      "transition_covariance",]

        self.kalman = pykalman.KalmanFilter(em_vars=["transition_matrices",
                                                     "transition_covariance",
                                                     ],
                                            transition_matrices=self.train_matrix)


    def predict(self):
        return self.state[0]

    def fit(self, pre_x, ob_x):
        #print self.kalman.transition_matrices
        if ob_x == '*':
            ob_x = None
        self.kalman.em([self.xs[-1]], em_vars=self.em_vars, n_iter=1)
        self.state, self.state_cor = self.kalman.filter_update(self.state,
                self.state_cor, ob_x, observation_matrix=self.ob_matrix)
        self.xs.append(ob_x)


if __name__ == "__main__":
    p = ArPredicter(1)
