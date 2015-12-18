import numpy as np

from pystruct.learners import NSlackSSVM, LatentSSVM
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt

# Internal Imports
from Util.data_parser import load_data
from Models.GraphLDCRF import GraphLDCRF

datatrain = "/home/manoel/Projects/hidden_states_entropy/Dataset/Data/ArmGesture/Discrete1/dataTrainArmGestureDiscreteFold11.csv"
datatest = "/home/manoel/Projects/hidden_states_entropy/Dataset/Data/ArmGesture/Discrete1/dataTestArmGestureDiscreteFold11.csv"
seqtrain = "/home/manoel/Projects/hidden_states_entropy/Dataset/Data/ArmGesture/Discrete1/seqLabelsTrainArmGestureDiscreteFold11.csv"
seqtest = "/home/manoel/Projects/hidden_states_entropy/Dataset/Data/ArmGesture/Discrete1/seqLabelsTestArmGestureDiscreteFold11.csv"

x, y, x_t, y_t = load_data(datatrain, seqtrain, datatest, seqtest)

np.random.seed(1)

states = [1, 1, 1, 1, 1, 1]

latent_pbl = GraphLDCRF(n_states_per_label=states, inference_method='dai')

base_ssvm = NSlackSSVM(latent_pbl, C=1, tol=.01, inactive_threshold=1e-3,
                       verbose=0, n_jobs=6, batch_size=10)

latent_svm = LatentSSVM(base_ssvm=base_ssvm, latent_iter=5)
latent_svm.fit(x, y)

print "------- Results -------"
print("Train: {:2.6f}".format(latent_svm.score(x, y)))
print("Test: {:2.6f}".format(latent_svm.score(x_t, y_t)))

Y_ts = []
for Y_ti in y_t:
    Y_ts.append(Y_ti.tolist())

Y_p = latent_svm.predict(x_t)
Y_ps = []
for Y_pi in Y_p:
    Y_ps.append(Y_pi.tolist())

print metrics.classification_report(Y_ts, Y_ps)

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
plt.savefig("/home/manoel/Projects/hidden_states_entropy/Dataset/Output/Results/imgs/ConfusionMatrix_6x1.pdf")

np.random.seed(1)

states = [2, 2, 2, 1, 1, 1]

latent_pbl = GraphLDCRF(n_states_per_label=states, inference_method='dai')

base_ssvm = NSlackSSVM(latent_pbl, C=1, tol=.01, inactive_threshold=1e-3,
                       verbose=0, n_jobs=6, batch_size=10)

latent_svm = LatentSSVM(base_ssvm=base_ssvm, latent_iter=5)
latent_svm.fit(x, y)

print "------- Results -------"
print("Train: {:2.6f}".format(latent_svm.score(x, y)))
print("Test: {:2.6f}".format(latent_svm.score(x_t, y_t)))

Y_ts = []
for Y_ti in y_t:
    Y_ts.append(Y_ti.tolist())

Y_p = latent_svm.predict(x_t)
Y_ps = []
for Y_pi in Y_p:
    Y_ps.append(Y_pi.tolist())

print metrics.classification_report(Y_ts, Y_ps)

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


plt.savefig("/home/manoel/Projects/hidden_states_entropy/Dataset/Output/Results/imgs/ConfusionMatrix_3x2_3x1.pdf")
