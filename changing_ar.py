# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com
import random
import numpy


class AR(object, ):

    def __init__(self, parameters, frequency, sd=0.3):
        self.parameters = parameters
        self.p = len(self.parameters[0])
        self.frequency = frequency
        self.group = len(self.parameters)
        self._xs = [random.random() for i in range(self.p)]
        self.sd = 0.3

    @property
    def noise(self):
        return numpy.random.normal(scale=self.sd)

    @property
    def timeseries(self):
        group_count = 0
        while group_count < self.group:
            group_count += 1
            frequency_count = 0
            parameter = numpy.array(self.parameters.pop())
            while frequency_count < self.frequency:
                past_p_xs = numpy.array(self._xs[-self.p:])
                new_x = parameter.dot(past_p_xs)
                new_x += self.noise
                if new_x > 1.0:
                    new_x = 1.0
                elif new_x < -1.0:
                    new_x = -1.0
                self._xs.append(new_x)
                frequency_count += 1
        return self._xs
