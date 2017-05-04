import numpy
import projection
import math

class ArPredicter:
    name = "AERR"

    def __init__(self, p, max_x):
        self.p = p
        self.xs = []
        self.fs = []
        self.ls = []
        self.former_ws = []
        self.xs_for_predict = []

    @property
    def w(self):
        p = self.p
        if len(self.fs) == 0:
            return numpy.zeros(self.p)
        mr = float(len(filter(lambda x: x=='*', self.xs))) / len(self.xs)
        r = AERR(p, mr=mr)
        return r.train(self.fs[-1], self.ls[-1], self.former_ws)

    def predict_and_fit(self, x):
        p = self.p
        if len(self.xs) < self.p:
            self.xs.append(x)
            self.xs_for_predict.append(x)
        else:
            rec_x = self.w.dot(self.xs_for_predict[-p:])
            if x == '*':
                self.xs.append(x)
                self.xs_for_predict.append(rec_x)
                return 0
            self.ls.append(x)
            self.fs.append(self.xs[-p:])
            self.xs.append(x)
            self.xs_for_predict.append(x)
            return rec_x


class AERR(object):

    def __init__(self, k, B=1, lr=0.0625, mr=0.0):
        self.k = k
        self.B = B
        self.lr = lr
        self.mr = mr
  
    #@one_or_one
    def predict(self, X):
        return self.avg_w.dot(X)

    def predict_label(self, X):
        if self.predict(X) > 0:
            return 1
        else:
            return -1

            
    def train(self, f, l, former_ws):
        B = self.B
        d = len(f)
        w = 0.000001 * numpy.random.random(d) if not former_ws else former_ws[-1]
        # assert numpy.linalg.norm(w, 2) < B
        k = self.k
        self.lr = float(k)/(2.0*d*len(former_ws)+1) ** 0.5
        self.indexs = [i for i in range(d)]
        y = l
        x = f
        if y == "*":
           return sum(former_ws) / len(former_ws) 
        x_t = numpy.zeros(d)
        for i in range(k):
            x_index = numpy.random.choice(self.indexs)
            if x[x_index] != '*':
                x_t[x_index] += d * x[x_index] / (1-self.mr)
        x_t = x_t / k
        w_norm = numpy.linalg.norm(w, 2) ** 2
        percents = w * w / w_norm
        w_index = numpy.random.choice(self.indexs, p=percents)
        if x[w_index] != '*':
            phi = w_norm * x[w_index] / w[w_index] / (1-self.mr) - y
        else:
            phi = -y
        g = phi * x_t
        v = w - self.lr * g
        new_w = v * B / max(B, numpy.linalg.norm(v, 2))
        former_ws.append(new_w)
        return sum(former_ws) / len(former_ws)


def prepare_ar(lenth, missing_rate):
    #a = arma.ARMA([0.3, -0.4, 0.4, -0.5, 0.6], [], 0.3)
    a = arma.ARMA([0.11, -0.5], [], 0.5, noise_type="uni")
    p = a.p
    time_series = [a.generater.next() for i in range(lenth)]
    train_lenth = int(len(time_series)*0.7)
    train_time_series = time_series[:train_lenth]
    test_time_series = time_series[train_lenth:]
    for index, t in enumerate(train_time_series):
        if numpy.random.random() < missing_rate:
            train_time_series[index] = '*'
    train_fs = []
    train_ls = []
    test_fs = []
    test_ls = []
    for index, t in enumerate(train_time_series):
        try:
            l = time_series[index+p]
            f = time_series[index: index+p]
            train_fs.append(f)
            train_ls.append(l)
        except:
            break
    for index, t in enumerate(test_time_series):
        try:
            l = time_series[index+p]
            f = time_series[index: index+p]
            test_fs.append(f)
            test_ls.append(l)
        except:
            break
    return train_fs, train_ls, test_fs, test_ls, train_time_series
