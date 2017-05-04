import pickle
import generate_data

def load_timeserieses(mr, length):
    name = generate_data.file_name.format(length, mr)
    return pickle.load(open(name, "rb"))

