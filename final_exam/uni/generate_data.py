import ar
import pickle


missing_rates = [0.0, 0.1, 0.2, 0.3]
#classical_ar = [0.3, -0.4, 0.4, -0.5, 0.6]
classical_ar = [-0.5, 0.11]
model = ar.AR(classical_ar, 0.5, noise_type="uni")

for missing_rate in missing_rates:
    time_serieses = []
    for i in range(20):
        time_serieses.append(model.get_time_series(2000, missing_rate))
    with open("data/series"+str(missing_rate), "wb") as f:
        pickle.dump(time_serieses, f)
