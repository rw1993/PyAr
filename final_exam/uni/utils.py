import pickle


def load_timeserieses(mr):
    name = "./data/series"
    name = name + str(mr)
    return pickle.load(open(name, "rb"))

