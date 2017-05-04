import ogd_impute_predicter
import my_yw
import aerr_with_missing
import pickle
import utils
import generate_data


def ran_test(missing_percen, time_serieses, ArPredicter, length):
    name = ArPredicter.name
    result = {}
    result["name"] = name
    result["errors"] = []
    result["mses"] = []
    result["length"] = length

    def ran_a_test(time_series):
        p = ArPredicter(length, max_x=max(time_series))
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
    result_name = "./result/length{0}result_{1}{2}"
    result_name = result_name.format(length, name, missing_percent)
    pickle.dump(result, open(result_name, "wb"))
    return sum(result['mses']) / len(time_serieses)


if __name__ == '__main__':
    missing_percents = [0.0, 0.1, 0.2, 0.3]
    for length in generate_data.lengths:
        for missing_percent in missing_percents:
            time_serieses = utils.load_timeserieses(missing_percent,
                                                    length)
            print (ran_test(missing_percent,
                           time_serieses,
                           aerr_with_missing.ArPredicter, length),
                   length,
                   missing_percent)
            print (ran_test(missing_percent,
                           time_serieses,
                           ogd_impute_predicter.ArPredicter, length),
                   length,
                   missing_percent)
            print (ran_test(missing_percent,
                           time_serieses,
                           my_yw.ArPredicter, length),
                   length,
                   missing_percent)
