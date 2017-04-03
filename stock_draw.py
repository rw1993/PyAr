import pickle
from matplotlib import pyplot
import numpy


#names = ["ogd", "kalman", "expand_predicter", "sample"]
names = ["ogd", "yw", "kalman", "expand_predicter", "arls",]# 'sample']
colors = ["r", "g", "b", "y", "black",]# "brown"]
#colors = ["r", "g", "b", "y", "black"]
#missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4]
missing_rates = [0.0, 0.1, 0.2]
missing_rates = map(str, missing_rates)
#missing_rates = ['0.0']
order = 5

name_color = {}
for name, color in zip(names, colors):
    name_color[name] = color


def get_from_pickle(name, missing_rate):
    name1 = str(order) + "_result_" + name + str(missing_rate)
    name2 = str(order) + "result_" + name + str(missing_rate)
    try:
        return pickle.load(open(name1, "rb"))
    except:
        return pickle.load(open(name2, "rb"))


def mse_plots():
    for missing_rate in missing_rates:
        mse_plot(missing_rate)
    pyplot.show()
def mse_plot(missing_rate):
    fig, ax = pyplot.subplots()
    ax.set_ylabel("mse")
    ax.set_xlabel("observation index")
    ax.set_title("mse when missing rate is "+missing_rate)
    results = {}
    for name in names:
        results[name] = get_from_pickle(name, missing_rate)
    
    def get_mse(former_mse, index, e):
        mse = former_mse * index + e
        return mse / (index + 1)

    def draw_a_name(name):
        errors = results[name]['errors']
        min_lenth = len(errors[0])
        for es in errors:
            min_lenth = min(min_lenth, len(es))
        errors = map(lambda x: [x[i]*x[i] for i in range(min_lenth)],
                     errors)
        mses = []
        for es in errors:
            mse = []
            former_mse = 0.0
            for index, e in enumerate(es):
                former_mse = get_mse(former_mse, index, e)
                mse.append(former_mse)
            mses.append(mse)
        def get_sum(*args):
            return sum(args)

        sum_mse = map(get_sum, *mses)
        mse = map(lambda x: x/len(errors), sum_mse)
        xs = [i for i in range(min_lenth)]
        ax.plot(xs, mse, color=name_color[name], label=name)
    for name in names:
        draw_a_name(name)
    ax.legend(loc='upper right')



            
def mse_compare_hist():
    fig, ax = pyplot.subplots()
    results = {}
    ind = numpy.arange(len(missing_rates))
    for name in names:
        for rate in missing_rates:
            results[name+str(rate)] = get_from_pickle(name,
                                                      rate)
    ax.set_xticks(ind + .15)
    ax.set_xticklabels(tuple(missing_rates))
    ax.set_ylabel("mse")
    ax.set_xlabel("missing rate")
    ax.set_ylim(ymax=0.2)
    def draw_a_name(name, index):
        mses = []
        color = name_color[name]
        for mr in missing_rates:
            key = name + str(mr)
            result = results[key]
            tmses = result['mses']
            mse = sum(tmses) / len(tmses)
            mses.append(mse)
        print mses
        mses = map(lambda x: x if x < 1.0 else 1.0,
                   mses)

        rt = ax.bar(ind+index*.15, mses, .15, color=color)
        return rt
    rs = [draw_a_name(name, index) for index, name in enumerate(names)]
    ax.legend((r[0] for r in rs), tuple(names))
    pyplot.show()


if __name__ == "__main__":
    #mse_compare_hist()
    #errors_plot(missing_rates[0])
    mse_plots()
