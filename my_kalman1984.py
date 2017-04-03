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
        '''
        self.M = np.matrix(np.zeros((self.p, self.p)))
        self.M[0, 0] = 1
        '''
        #self.M = np.matrix(np.zeros(p))
        #self.M[0,0] = 1
        #print self.M
        self.xtts = []
        self.x0 = np.matrix(np.array([0.01*np.random.random() for i in range(p)])).T
        self.xtts.append(self.x0)
        print self.xtts[-1]
        self.ps = []
        self.ptts = []
        self.ptt0 = np.matrix(999999 * np.eye(self.p))
        self.ptts.append(self.ptt0)
        print self.ptts[-1]
        self.pt_ts = []
        self.phis = [np.matrix(np.eye(p))]
        #self.phis = [0.01*np.matrix(np.random.random((p, p)))]
        print self.phis[-1]
        
        self.Qs = [np.matrix(np.zeros((p, p)))]
        self.Js = []

    @property
    def M(self):
        return np.matrix(self.xs[-self.p:])
        
    def predict(self, return_num=False):
        phi = self.phis[-1]
        old_x = self.xtts[-1]
        predict_x = phi.dot(old_x)
        old_p = self.ptts[-1]
        Q = self.Qs[-1]
        predict_y = self.M.dot(predict_x)
        if not return_num:
            return predict_y[0,0]
        predict_p = phi.dot(old_p).dot(phi.T) + Q
        return predict_x, predict_p, predict_y

    def update(self, new_x, new_p):
        old_p = self.ptts[-1]
        phi = self.phis[-1]
        new_J = old_p.dot(phi.T).dot(inv(self.pt_ts[-1]))
        self.Js.append(new_J)
        xns = [new_x]
        for x, J in zip(self.xtts, self.Js)[::-1]:
            xn = xns[-1]
            new_xn = x + J.dot(xn-phi.dot(x))
            xns.append(new_xn)
        pns = [new_p]
        for ptt, pt_t, J in zip(self.ptts, self.pt_ts, self.Js)[::-1]:
            pn = pns[-1]
            new_pn = ptt + J.dot(pn-pt_t).dot(J.T)
            pns.append(new_pn)
        pnn_ = np.matrix(np.eye(self.p))
        pnn_s = [pnn_]
        for index, (ptt, pt_t, J) in enumerate(zip(self.ptts, self.pt_ts,
                                                   self.Js)[::-1]):
            if index + 1 >= len(self.Js):
                break
            o_pnn_ = pnn_s[-1]
            J_ = self.Js[::-1][index+1]
            new_pnn_ = ptt.dot(J_.T) + J.dot(o_pnn_-phi.dot(ptt)).dot(J_.T)
            pnn_s.append(new_pnn_)
        A = sum(p+x.dot(x.T) for p, x in zip(pns, xns)[1:])
        C = sum(p+x.dot(x.T) for p, x in zip(pns, xns)[:-1])
        B = None
        for index, pnn_ in enumerate(pnn_s):
            x1 = xns[index]
            x2 = xns[index+1]
            add = pnn_ + x1.dot(x2.T)
            if B is None:
                B = add
            else:
                B += add
        #new_phi = B.dot(inv(A))
        new_Q = 1.0 / len(self.xtts) * (C - B.dot(inv(A).dot(B.T)))
        self.Qs.append(new_Q)
        #self.phis.append(new_phi)
        self.phis.append(self.phis[-1])
        self.xtts.append(new_x)
        self.ptts.append(new_p)
            

    def fit(self, pre_y, ob_y):
        p_x, p_p, p_y = self.predict(True)
        self.pt_ts.append(p_p)
        M = self.M
        K = p_p.dot(M.T) / (M.dot(p_p)).dot(M.T)
        new_p = p_p - K.dot(M).dot(p_p)
        if ob_y != '*':
            new_x = p_x - K * (ob_y - ob_y)
        else:
            new_x = p_x
        self.update(new_x, new_p)
        if ob_y == '*':
            self.xs.append(pre_y)
        else:
            self.xs.append(ob_y)


if __name__ == "__main__":
    p = ArPredicter(5)
