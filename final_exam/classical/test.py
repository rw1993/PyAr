import ogd_impute_predicter
import my_yw
import aerr_with_missing
import pickle
import utils


def ran_test(missing_percen, time_serieses, ArPredicter):
    name = ArPredicter.name
    result = {}
    result["name"] = name
    result["errors"] = []
    result["mses"] = []

    def ran_a_test(time_series):
        p = ArPredicter(5, max_x=max(time_series))
        errors = []
        for index, x in enumerate(time_series):
            if x != '*':
                rec_x = p.predict_and_fit(x)
                if rec_x is not None:
                    errors.append(rec_x - x)
            else:
                p.predict_and_fit('*')
        mse = sum(map(lambda x: x**2, errors)) / len(errors)
        result['errors'].append(errors)
        result['mses'].append(mse)
        #print mse
    for t in time_serieses:
        ran_a_test(t)
    pickle.dump(result, open("./result/result_"+name+str(missing_percent), "wb"))
    return sum(result['mses']) / len(time_serieses)


if __name__ == '__main__':
    missing_percents = [0.0, 0.1, 0.2, 0.3]
    for missing_percent in missing_percents:
        time_serieses = utils.load_timeserieses(missing_percent)
        print ran_test(missing_percent,
                       time_serieses,
                       my_yw.ArPredicter)
