from arma import ARMA
#from expand_predicter import *
#from e_expand_predicter import *
#from hash_e_expand_predicter import *
#from similar_expand_predicter import *
#from yule_walker_impute_predicter import *
#from kalman_impute_predicter import *
#from ogd_impute_predicter import *
#from weighted_e_expand_predicter import *
import random
import pickle
import sys
from matplotlib import pyplot



missing_percent = 0.2
a = ARMA([0.3, -0.4, 0.4, -0.5, 0.6], [-0.2, 0.3], 0.1)
#a = ARMA([0.3, -0.4, 0.4, -0.5, 0.6], [], 0.3)
#a2 = ARMA([0.3, -0.4, 0.4, -0.5, 0.6], [], 0.1)
#a = ARMA([0.3, -0.4, 0.4, -0.5, 0.6, 0.3, -0.4, 0.4, -0.5, 0.6], [], 0.3)
#a = ARMA([0.4, 0.5], [], 0.02 ** 0.5)
#a2 = ARMA([0.4, 0.1], [], 0.5 ** 0.5)
#a2 = ARMA([0.4, 0.1], [], 0.5 ** 0.5)
#a1 = ARMA([0.4, 0.6], [], 0.14 ** 0.5)
#a = ARMA([0.25, 0.23, 0.19], [], 0.07 ** 0.5)
#a1 = ARMA([0.02, 0.50, 0.036], [], 0.16 ** 0.5)
time_series = [a.generater.next() for i in range(2000)]
#time_series += [a2.generater.next() for i in range(1000)]
#time_series += [a2.generater.next() for i in range(500)]
p = ArPredicter(len(a.alphas), max_x = max(time_series))

def run_test():
    for index, x in enumerate(time_series):
        print index
        if index < p.min_ob:
            p.predict_and_fit(x)
        elif random.random() > missing_percent:
            rec_x = p.predict_and_fit(x)
        else:
            p.predict_and_fit('*')
    serrors = [error * error for error in p.errors]
    sserror = sum(serrors)
    lenth = len(filter(lambda x:x!='*', p.xs))
    return sserror / float(lenth)


def plot(p):
    serrors = [error*error for error in p.errors]
    xs = [index for index, e in enumerate(serrors)]
    ys = [sum(serrors[:index+1])/(index+1) for index, e in enumerate(serrors)]
    print ys[-1]
    pyplot.plot(xs, ys)
    pyplot.show()


if __name__ == '__main__':
    mse = sum(run_test() for i in range(20)) / float(20)
    print mse
