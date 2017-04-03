from arma import ARMA
import e_expand_predicter
import expand_predicter
import yule_walker_impute_predicter
import kalman_impute_predicter
import ogd_impute_predicter
import sample_not_imput_predicter
from matplotlib import pyplot
import random
import pickle
import sys
import tushare
import math

print "get stock codes"
codes = tushare.get_stock_basics()
codes = [code for code in codes.index]
print "codes geted"
missing_percents = [0.0, 0.1, 0.2]
order = 5

def get_stock_data():
    while True:
        code = random.choice(codes)
        prices = []
        days = ["2015-12-05", "2015-12-06", "2015-12-07", "2015-12-08", "2015-12-09"]
        prices = []
        for day in days:
            df = tushare.get_tick_data(code, day)
            for p in df['price']:
                if not math.isnan(p):
                      prices.append(p)
                if len(prices) >= 2000:
                    max_p = max(prices)
                    min_p = min(prices)
                    return map(lambda x: -1.0+(x-min_p)/(max_p-min_p),
                               prices)
        


def ran_test(order, missing_percent, ArPredicter, name, time=20):

    result = {}
    result["name"] = name
    result["errors"] = []
    result["mses"] = []

    def ran_a_test():
        time_series = get_stock_data()
        p = ArPredicter(order, max_x = max(time_series),)
        errors = []
        for index, x in enumerate(time_series):
            if index < p.min_ob:
                p.predict_and_fit(x)
            elif random.random() > missing_percent:
                rec_x = p.predict_and_fit(x)
                errors.append(rec_x - x)
            else:
                p.predict_and_fit('*')
        mse = sum(map(lambda x: x**2, errors)) / len(errors)
        result['errors'].append(errors)
        result['mses'].append(mse)
    for i in range(time):
        print i
        ran_a_test()

    pickle.dump(result, open(str(order)+"result_"+name+str(missing_percent), "wb"))
    return sum(result['mses']) / time


if __name__ == '__main__':
    for missing_percent in missing_percents:
        print ran_test(order, missing_percent, ogd_impute_predicter.ArPredicter, "ogd")
        print ran_test(order, missing_percent, kalman_impute_predicter.ArPredicter, "kalman")
        print ran_test(order, missing_percent,  e_expand_predicter.ArPredicter,
                       "expand_predicter", time=5)
        print ran_test(order, missing_percent, yule_walker_impute_predicter.ArPredicter, "yw")
