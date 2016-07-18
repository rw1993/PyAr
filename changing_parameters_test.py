# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com
import weighted_e_expand_predicter as wep
import kalman_impute_predicter as kalman
import random
from changing_ar import AR


def random_parameters(parameter_lenth, group):
    rp = lambda :[random.random() for i in range(parameter_lenth)]
    rps = [rp() for i in range(group)]
    return rps


def changing_parameter_test(parameter_lenth, group, frequency=100000,
                            sd=0.3, missing=0.1):
    rps = random_parameters(parameter_lenth, group)
    ar = AR(rps, frequency, sd)
    #p = wep.ArPredicter(parameter_lenth)
    p = kalman.ArPredicter(parameter_lenth)
    for index, x in enumerate(ar.timeseries):
        print index
        if random.random < missing:
            p.predict_and_fit('*')
        else:
            p.predict_and_fit(x)
    ses = map(lambda x:x*x, p.errors)
    print float(sum(ses))/float(len(ses))
    print p.ws
    print p.filter_state_covariance




        


if __name__ == "__main__":
    changing_parameter_test(2, 1)
