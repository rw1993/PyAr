import ar
import pickle
import numpy


file_name = "data/length{0}series{1}"
lengths = [2, 5, 10, 50, 100]
missing_rates = [0.0, 0.1, 0.2, 0.3]
if __name__ == "__main__":
    for length in lengths:    
        for missing_rate in missing_rates:
            time_serieses = []
            while len(time_serieses) < 20:
                print len(time_serieses)
                ars = [numpy.random.normal() for i in range(length)]
                model = ar.AR(ars, 0.3)
                time_series = model.get_time_series(2000, missing_rate)
                if numpy.inf in time_series:
                    continue
                if numpy.NaN in time_series:
                    continue
                time_serieses.append(time_series)
            with open(file_name.format(length, missing_rate), "wb") as f:
                pickle.dump(time_serieses, f)
