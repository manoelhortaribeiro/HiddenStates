
import datetime
import pickle


__author__ = 'Manoel Ribeiro'


def save_model(model, stats, results, description, latent_svm, report, cm, y_p, y_t):

    date = datetime.datetime.utcnow().strftime("%d_%m_%y-%H:%M")
    suffix = description + "_" + date

    f = open(model + suffix, "a")
    pickle.dump([latent_svm], f)
    f.close()

    f = open(stats + suffix, "a")
    f.write(report)
    f.close()

    with open(results + suffix, 'a') as f:
        pickle.dump([cm, y_p, y_t], f)
    f.close()
