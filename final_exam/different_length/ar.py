# -*- coding:utf8 -*-
import numpy

class AR(object, ):
    
    
    @property
    def max_noise(self):
        return max(map(abs, self.noises))

    def __init__(self, alphas, sigma, noise_type="normal"):
        self.noise_map = {"normal": self.normal_noise,
                          "uni": self.uni_noise}
        self.noise =  self.noise_map[noise_type]
        self.alphas =numpy.array(alphas)
        self.sigma = sigma
        self.p = len(self.alphas)

    def get_time_series(self, length, mr):
        self.init_first_few_xs()
        timeseries = [t for t in self.xs]
        while len(timeseries) != length:
            current_x = numpy.array(timeseries[-self.p:])
            new = self.alphas.dot(current_x)+self.noise()
            if new > 1:
                new = 1
            elif new < -1:
                new = -1
            timeseries.append(new)
        for index, t in enumerate(timeseries):
            if index > self.p * 2:
                if numpy.random.random() < mr:
                    timeseries[index] = '*'
        return timeseries
        

    def init_first_few_xs(self):
        self.xs = [self.noise() for alpha in self.alphas]
        # self.xs = [1 for alpha in self.alphas]

       

    def normal_noise(self):
        if self.sigma == 0:
            return 0
        return numpy.random.normal(0, self.sigma, 1)[0]
    
    def uni_noise(self):
        if self.sigma == 0:
            return 0
        return numpy.random.uniform(-self.sigma, self.sigma)

if __name__ == '__main__':
    classical_ar = [0.3, -0.4, 0.4, -0.5, 0.6]
    a = AR([2, 1], 0.0)
