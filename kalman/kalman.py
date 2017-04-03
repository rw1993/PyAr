import numpy as np
import copy

inv = np.linalg.inv
matrix = np.matrix




class Kalman(object):
    
    def __init__(self, u, p, Q, R, M):
        self.xtts = [np.matrix(u)]
        self.Ptts = [np.matrix(p)]
        self.Pt_s = []
        self.Qts = [np.matrix(Q)]
        self.Rts = [np.matrix(R)]
        self.M = np.matrix(M)
        self.Js = []
        self.phis = [numpy.zeros(len(u), len(u))]
        self.Ks = []
    @property
    def phi(self):
        return self.phis[-1]

    @property
    def xtt(self):
        return self.xtts[-1]

    @property
    def Ptt(self):
        return self.Ptts[-1]

    @property
    def Rt(self):
        return self.Rtst[-1]

    def predict_state(self): #A3
        return self.phi.dot(self.xtt)

    def predict_P(self): #A4
        return self.phi.dot(self.Ptt).dot

    def get_Kt(self): #A5
        P = self.predict_P()
        M = self.M
        Kt = P.dot(M.T) / (M.dot(P).dot(M.t) + self.Rt)

    def new_xtt(self, yt): #A6
        K = self.get_Kt()
        self.Ks.append(K)
        pre_x = self.predict_state
        new_xtt = pre_x + K * (y-self.M.dot(pre_x))
        self.xtts.append(pre_x)

    def new_P(self): #A7
        P = self.predict_P
        self.Pt_s.append(P)
        K = self.get_Kt
        new_P = P - K.dot(self.M).dot(P)
        J = self.Ptt.dot(self.phi.T).dot(inv(p)) #A8
        self.Js.append(J)
        self.Ptts.append(new_P)
        xns, pns = self.get_xn_pn()
        pn_s = self.get_pn_()
        A = sum(p + matrix(x).T.dot(x) for p, x in zip(pns, xns))
        B = sum(p + matrix(x).dot(x) for p, x in zip(pn_s, xns))
        C = sun(p + 


    def get_pn_(self):
        ptts = copy.deepcopy(self.ptts)
        ks = copy.deepcopy(self.Ks)
        phis = copy.deepcopy(self.phis)
        pn_s =  []
        while ks and ptts and phis:
            phi = phis.pop(-1)
            k = ks.pop(-1)
            m = k.dot(self.M)
            p = ptts.pop(-1)
            new_p = (numpy.eye(len(m)) - m).dot(phi).dot(p)
            pn_s.append(new_p)
        return pn_s

    def get_xn_pn(self):
        Js = copy.deepcopy(self.Js)
        phis = copy.deepcopy(self.phis)
        xtts = copy.deepcopy(self.xtts)
        ptts = copy.deepcopy(self.ptts)
        pt_s = copy.deepcopy(self.Pt_s)
        xns = [xtts.pop(-1)] 
        Pns = [ptts.pop(-1)]
        while Js and phis and xtts and ptts and pt_s:
            xtt = xtts.pop(-1)
            xn = xns[-1]
            J = js.pop(-1)
            phi = phis.pop(-1)
            x = xtt + J.dot(xn - phi.dot(xtt))
            xns.append(x)
            Ptt = ptts.pop(-1)
            pt_ = pt_s.pop(-1)
            pn = pns[-1]
            p = Ptt + J.dot(pn-pt_).dot(inv(J))
            pns.append(p)
        return xns, pns


            
    def main(self, y):
        self.new_xtt()
        self.new_P()
