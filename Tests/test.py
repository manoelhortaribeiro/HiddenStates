# Deap Imports
from pystruct.learners import NSlackSSVM, LatentSSVM
# Internal Imports
from Util.data_parser import load_data
from Models.GraphLDCRF import GraphLDCRF
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

path = "../Dataset/Data/fold"

x, y, x_t, y_t = load_data(path + "0data.csv", path + "0label.csv",
                           path + "1data.csv", path + "1label.csv")

print(x[0])
print(y)

latent_pbl = GraphLDCRF(n_states_per_label=3,inference_method='max-product')
base_ssvm = NSlackSSVM(latent_pbl, C=1, tol=.01, inactive_threshold=1e-3, batch_size=10, verbose=1, n_jobs=8)
latent_svm = LatentSSVM(base_ssvm=base_ssvm, latent_iter=10)
latent_svm.fit(x, y)

test = latent_svm.score(x_t, y_t)
train = latent_svm.score(x, y)

print("Test", test)
print("Train", train)

Y_ts = []
for Y_ti in y_t:
    Y_ts.append(Y_ti.tolist())

Y_p = latent_svm.predict(x_t)
Y_ps = []
for Y_pi in Y_p:
    Y_ps.append(Y_pi.tolist())

print Y_ts, y_t

Y_ts2 = MultiLabelBinarizer().fit_transform(Y_ts)
Y_ps2 = MultiLabelBinarizer().fit_transform(Y_ps)

print metrics.classification_report(Y_ts2, Y_ps2)

cm = confusion_matrix(sum(Y_ts, []), sum(Y_ps, []))
# Show confusion matrix in a separate window
plt.matshow(cm, cmap=plt.get_cmap('Greys'))

plt.title('Confusion matrix')
for i, cas in enumerate(cm):
    for j, c in enumerate(cas):
        if c > 0:
            plt.text(j - .2, i + .2, c, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

