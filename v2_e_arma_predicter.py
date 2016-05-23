# -*- coding:utf8 -*-


class ArmaPredicter(object,):

    def __init__(self, p, max_noise, missing_ability=3,
                 learning_rate=1.0/30):
        #p = m + k p = 10
        self.p = p
        self.d = int(missing_ability * p) # more we choose, righter we get but slower
        self.xs = []
        self.errors = []
        self.ERRs = []
        self.max_noise = max_noise
        self.predict_xs =[0.0]
        self._learning_rate = learning_rate
        self.d_kernel = 0
        self.tmp = 0

    def predict_and_fit(self, x):
        if len(self.xs) < self.d:
            self.xs.append(x)
            #self.predict_xs.append(0.0)
            self.errors.append(self.max_noise)
            self.errors.append(0)
            return 0
        else:
            self.xs.append(x)
            if x == '*':
                self.tmp = 0
                self.ERRs.append(0)
                self.errors.append(0)
                return 0
            p_x = self.predict_xs[-1]
            error = self.predict_xs[-1] - x
            self.errors.append(abs(error))
            self.ERRs.append(error)
            self.d_kernel += error * self.tmp
            self.predict()
            return p_x

    def predict(self):
        predict_x = 0
        for index, error in enumerate(self.ERRs):
            predict_x += error*self.K(index, len(self.ERRs))
        self.tmp = predict_x
        predict_x = -self._learning_rate * predict_x
        predict_x = predict_x / self.current_divider()
        self.predict_xs.append(predict_x)
        

    def current_divider(self):
        divider = self.d_kernel * (2 ** -self.d)
        divider = divider ** 0.5
        divider = divider * self._learning_rate
        return max(1.0, divider)

    def K(self, small, big):
        small_xs = self.xs[small:small+self.d][::-1]
        big_xs = self.xs[big:big+self.d][::-1]
        def C(k):
            count = 0
            for i in range(k):
                if small_xs[i] != '*' or big_xs[i] != '*':
                    continue
                else:
                    count +=1
            return count
        k = 0
        for i in range(self.d):
            if small_xs[i] == '*' or big_xs[i] == '*':
                continue
            k += (2**C(i))*small_xs[i]*big_xs[i]
        return k
