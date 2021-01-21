import utils
import glob
import preprocess as pp
import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, r2_score, mean_squared_error


def calcAUC(Y_pred, Y_tst):
    n_classes = Y_pred.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = np.zeros((n_classes, 1))
    for i in np.arange(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_tst[:, i], Y_pred[:, i] / n_classes)
        roc_auc[i] = auc(fpr[i], tpr[i])
    p_class = np.sum(Y_tst, axis=0) / np.sum(Y_tst)
    return np.sum(roc_auc.flatten() * p_class)


gm = True
ryc = False

# Load data
if ryc:
    with open("./data/ryc_data.pkl", 'rb') as pkl:
        ryc_data = pickle.load(pkl)
    ryc_x = ryc_data["maldi"].copy(deep=True)
    ryc_y = ryc_data[ryc_data.columns[-12:]].copy(deep=True)
    nan_index = ryc_data.isnull().any(axis=1)
    del ryc_data
    y_seen = ryc_y[~nan_index]
    y_ss = ryc_y[nan_index]
    x_seen = ryc_x[~nan_index]
    x_ss = ryc_x[nan_index]
    with open("./data/ryc_folds.pkl", 'rb') as pkl:
        folds = pickle.load(pkl)

if gm:
    with open("./data/gm_data.pkl", 'rb') as pkl:
        gm_x, gm_y = pickle.load(pkl)
    y_seen = gm_y[~np.isnan(gm_y).any(axis=1)]
    y_ss = gm_y[np.isnan(gm_y).any(axis=1)]
    ss_index = np.argwhere(np.isnan(gm_y).any(axis=1))[:, 0]
    x_seen = np.delete(gm_x, ss_index, axis=0)
    x_ss = gm_x[ss_index]
    with open("./data/gm_folds.pkl", 'rb') as pkl:
        folds = pickle.load(pkl)

with open("./data/common_labels.pkl", 'rb') as pkl:
    common_labels = pickle.load(pkl)

with open("./data/gm_labels.pkl", 'rb') as pkl:
    gm_labels = pickle.load(pkl)

# gm_x, gm_y = pp.process_hospital_data(gm_path=gm_path, gm_n_labels=18, ryc_path='data/Klebsiellas_RyC/',
#                                       keep_common=True, return_GM=True, return_RYC=False, norm_both=False)


# ======== Parameters to plot and store ========
# If you want the results to be plot
plot = True
# If you want to store the plots
store = True
# If you want to print the results
print_r = True

# for file in glob.glob("./Results/SSHIBAmult.pkl"):
#     results = utils.load_results(main_folder=file, labels=labels_name.tolist(), file=file, plot=plot, print_r=print_r, store=store)

print("Estamos cargando el modelo de 10GB")
with open("Results/GM_commonaxis_commonlabels_reg_model.pkl", 'rb') as pkl:
    results = pickle.load(pkl)
    reg = True
print("Por fin lo hemos cargado")

c = 0
lf_imp = []
auc_list = []
r2_list = []
mse_list = []
for f in range(len(folds["train"])):
    print("Testing fold: ", c)
    model = "model_fold" + str(c)
    if ryc:
        x_tr, y_tr, x_tst, y_tst = x_seen.loc[folds["train"][f]], y_seen.loc[folds["train"][f]], \
                                   x_seen.loc[folds["test"][f]], y_seen.loc[folds["test"][f]]
        y_tst = y_tst.to_numpy()
        y_tst[y_tst > 1] = 0
    if gm:
        x_tr, y_tr, x_tst, y_tst = x_seen[folds["train"][f]], y_seen[folds["train"][f]], x_seen[folds["test"][f]], \
                                   y_seen[folds["test"][f]]

    if reg:
        y_pred = results[model].X[1]["mean"][-y_tst.shape[0]:, :]
        r2_list.append(r2_score(y_tst, y_pred))
        mse_list.append(mean_squared_error(y_tst, y_pred))
        auc_list.append(roc_auc_score(y_tst, y_pred, average="micro"))
    else:
        y_pred = results[model].t[1]["mean"][-y_tst.shape[0]:, :]
        auc_list.append(roc_auc_score(y_tst, y_pred, average="micro"))

    lf_imp.append(np.argwhere(np.mean(np.abs(results[model].q_dist.W[1]["mean"]), axis=0) > 0.2))

    # auc_list.append(roc_auc_score(y_true=y_tst, y_score=y_tst))

    print(auc_list)
    c += 1


# plt.figure()
# for key in results:
#    label_name = str(key)
#    plt.plot(results[key].AUC, label=label_name)
# plt.xlabel("Iterations")
# plt.ylabel("AUC")
# plt.legend()
# plt.title("AUC with multilabel approach")
# plt.show()


def plot_W(W):
    plt.figure()
    plt.imshow((np.abs(W)), aspect=W.shape[1] / W.shape[0])
    plt.colorbar()
    plt.title('W')
    plt.ylabel('features')
    plt.xlabel('K')
    plt.show()


for f in range(len(folds["train"])):
    model = "model_fold" + str(f)
    plot_W(results[model].q_dist.W[1]["mean"])

# plt.figure()
# plt.plot(results_onelinear["model"].L, label="One linear kernel view")
# plt.xlabel("Iterations")
# plt.ylabel("ELBO")
# plt.legend()
# plt.title("ELBO with multilabel approach")
# plt.show()
#
# plt.figure()
# plt.plot(results_onenormal["model"].L, label="One no kernelized view")
# plt.xlabel("Iterations")
# plt.ylabel("ELBO")
# plt.legend()
# plt.title("ELBO with multilabel approach")
# plt.show()
#
# #plt.stem(range(results_onenormal["model"].X[0]["mean"].shape[1]), results_onelinear["model"].sparse_K[0].kernel.variance_unconstrained.data.numpy())
#
# plt.figure()
# plt.plot(results_onelinear["model"].L, label="One linear kernel view")
# plt.plot(results_onenormal["model"].L, label="One no kernelized view")
# plt.xlabel("Iterations")
# plt.ylabel("ELBO")
# plt.legend()
# plt.title("ELBO with multilabel approach")
# plt.show()

fold_show = "model_fold8"
# samples to plot: all except the semisupervised ones
f = 8
x_tr, y_tr, x_tst, y_tst = x_seen[folds["train"][f]], y_seen[folds["train"][f]], x_seen[folds["test"][f]], \
                                   y_seen[folds["test"][f]]
s_plot = x_tr.shape[0] + x_tst.shape[0]
y_total = np.concatenate((y_tr, y_tst))
for i in range(12):
    plt.figure()
    plt.title(common_labels[i])
    scatter = plt.scatter(results[fold_show].q_dist.Z["mean"][-s_plot:, int(lf_imp[f][0])],
                results[fold_show].q_dist.Z["mean"][-s_plot:, int(lf_imp[f][1])],
                c=y_total[:, i])
    plt.legend(handles=scatter.legend_elements()[0], labels=[0, 1])
    plt.show()
