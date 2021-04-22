import utils
import glob
import preprocess as pp
import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


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


gm = False
ryc = True
both = False
# Load data
if ryc:
    with open("./data/ryc_data_TIC.pkl", 'rb') as pkl:
        ryc_data = pickle.load(pkl)
    ryc_x = ryc_data["maldi"].copy(deep=True)
    ryc_y = ryc_data['binary_ab'].copy(deep=True)
    nan_index = ryc_data['binary_ab'].isnull().any(axis=1)
    y_seen = ryc_y[~nan_index]
    y_ss = ryc_y[nan_index]
    x_seen = ryc_x[~nan_index]
    x_ss = ryc_x[nan_index]
    with open("./data/ryc_5folds_NOERTAPENEM.pkl", 'rb') as pkl:
        folds = pickle.load(pkl)
if gm:
    with open("./data/old_data/gm_data.pkl", 'rb') as pkl:
        gm_x, gm_y = pickle.load(pkl)
    y_seen = gm_y[~np.isnan(gm_y).any(axis=1)]
    y_ss = gm_y[np.isnan(gm_y).any(axis=1)]
    ss_index = np.argwhere(np.isnan(gm_y).any(axis=1))[:, 0]
    x_seen = np.delete(gm_x, ss_index, axis=0)
    x_ss = gm_x[ss_index]
    with open("./data/old_data/gm_folds.pkl", 'rb') as pkl:
        folds = pickle.load(pkl)

if both:
    with open("./data/old_data/gm_data_both.pkl", 'rb') as pkl:
        gm_x, gm_y = pickle.load(pkl)
    with open("./data/old_data/ryc_data_both.pkl", 'rb') as pkl:
        ryc_full_data = pickle.load(pkl)
    ryc_x = ryc_full_data["maldi"].copy(deep=True)
    ryc_y = ryc_full_data[ryc_full_data.columns[-12:]].copy(deep=True)
    nan_index = ryc_full_data.isnull().any(axis=1)

    gm_y_seen = gm_y[~np.isnan(gm_y).any(axis=1)]
    gm_y_ss = gm_y[np.isnan(gm_y).any(axis=1)]
    ss_index = np.argwhere(np.isnan(gm_y).any(axis=1))[:, 0]
    gm_x_seen = np.delete(gm_x, ss_index, axis=0)
    gm_x_ss = gm_x[ss_index]

    ryc_y_seen = ryc_y[~nan_index]
    ryc_y_ss = ryc_y[nan_index]
    ryc_x_seen = ryc_x[~nan_index]
    ryc_x_ss = ryc_x[nan_index]

    with open("./data/old_data/ryc_folds.pkl", 'rb') as pkl:
        ryc_folds = pickle.load(pkl)

    with open("./data/old_data/gm_folds.pkl", 'rb') as pkl:
        gm_folds = pickle.load(pkl)

    folds = gm_folds


with open("./data/old_data/common_labels.pkl", 'rb') as pkl:
    common_labels = pickle.load(pkl)
#
# with open("./data/old_data/gm_labels.pkl", 'rb') as pkl:
#     gm_labels = pickle.load(pkl)

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
#"Results/RyC_5fold_cat_fen_gen_NOERTAPENEM_nogpu.pkl

with open("./Results/RyC_5fold_MULT_varpriori_NOERTAPENEM_NOSTD_nogpu.pkl", 'rb') as pkl:
    results = pickle.load(pkl)
    reg = False
print("Por fin lo hemos cargado")

c = 0
lf_imp = []
auc_list = []
r2_list = []
mse_list = []
for f in range(len(folds["train"])):
    print("Validating fold: ", c)
    model = "model_fold" + str(c)
    if ryc:
        x_tr, y_tr, x_tst, y_tst = x_seen.loc[folds["train"][f]], y_seen.loc[folds["train"][f]], \
                                   x_seen.loc[folds["val"][f]], y_seen.loc[folds["val"][f]]
        y_tst = y_tst.to_numpy()

        x_tr = np.vstack(x_tr.values).astype(float)
        scaler = StandardScaler()
        x0_tr_norm = scaler.fit_transform(x_tr)

    if gm:
        x_tr, y_tr, x_tst, y_tst = x_seen[folds["train"][f]], y_seen[folds["train"][f]], x_seen[folds["test"][f]], \
                                   y_seen[folds["test"][f]]
    if both:
        ryc_x_tr, ryc_y_tr, ryc_x_tst, ryc_y_tst = ryc_x_seen.loc[ryc_folds["train"][f]], \
                                                   ryc_y_seen.loc[ryc_folds["train"][f]], \
                                                   ryc_x_seen.loc[ryc_folds["test"][f]], \
                                                   ryc_y_seen.loc[ryc_folds["test"][f]]
        gm_x_tr, gm_y_tr, gm_x_tst, gm_y_tst = gm_x_seen[gm_folds["train"][f]], gm_y_seen[gm_folds["train"][f]], \
                                               gm_x_seen[gm_folds["test"][f]], gm_y_seen[gm_folds["test"][f]]

        y_tst = np.vstack((np.vstack(ryc_y_tst.values), gm_y_tst))
    if reg:
        y_pred = results[model].X[3]["mean"][-y_tst.shape[0]:, :]
        r2_list.append(r2_score(y_tst, y_pred))
        mse_list.append(mean_squared_error(y_tst, y_pred))
        auc_list.append(roc_auc_score(y_tst, y_pred, average="micro"))
    else:
        y_pred = results[model].t[3]["mean"][-y_tst.shape[0]:, :]
        auc_list.append(roc_auc_score(y_tst, y_pred, average="micro"))
        print(auc_list[-1])

    lf_imp.append(np.argwhere(np.mean(np.abs(results[model].q_dist.W[3]["mean"]), axis=0) > 0.5))

    # auc_list.append(roc_auc_score(y_true=y_tst, y_score=y_tst))

    print(auc_list)
    c += 1




#predict

if gm:
    with open("./data/old_data/ryc_data.pkl", 'rb') as pkl:
        ryc_data = pickle.load(pkl)
    ryc_x = ryc_data["maldi"].copy(deep=True)
    ryc_y = ryc_data[ryc_data.columns[-12:]].copy(deep=True)
    nan_index = ryc_data.isnull().any(axis=1)
    del ryc_data
    true_y = ryc_y[~nan_index]
    true_y[true_y > 1] = 0
    true_x = np.vstack(ryc_x[~nan_index].values)
    with open("./data/old_data/ryc_folds.pkl", 'rb') as pkl:
        pred_folds = pickle.load(pkl)

# pred_auc = []
# for f in range(len(pred_folds["train"])):
#     print("Testing the other hospital fold: ", c)
#     model = "model_fold" + str(c)
#     X_true = results[model].struct_data(true_x, method='reg', V=true_x, kernel="linear", sparse_fs=1)
#     y_pred = results[model].predict([0], 1, X_true)
#     if reg:
#         y_pred = results[model].X[1]["mean"][-y_tst.shape[0]:, :]
#         r2_list.append(r2_score(y_tst, y_pred))
#         mse_list.append(mean_squared_error(y_tst, y_pred))
#         auc_list.append(roc_auc_score(y_tst, y_pred, average="micro"))
#     else:
#         y_pred = results[model].t[1]["mean"][-y_tst.shape[0]:, :]
#         auc_list.append(roc_auc_score(y_tst, y_pred, average="micro"))
#         print(roc_auc_score(y_tst, y_pred, average="micro"))
#
#     lf_imp.append(np.argwhere(np.mean(np.abs(results[model].q_dist.W[1]["mean"]), axis=0) > 0.5))
#
#     # auc_list.append(roc_auc_score(y_true=y_tst, y_score=y_tst))
#
#     print(auc_list)
#     c += 1

# plt.figure()
# for key in results:
#    label_name = str(key)
#    plt.plot(results[key].AUC, label=label_name)
# plt.xlabel("Iterations")
# plt.ylabel("AUC")
# plt.legend()
# plt.title("AUC with multilabel approach")
# plt.show()


def plot_W(W, title):
    plt.figure()
    plt.imshow((np.abs(W)), aspect=W.shape[1] / W.shape[0])
    plt.colorbar()
    plt.title(title)
    plt.ylabel('features')
    plt.xlabel('K')
    plt.show()


for f in range(len(folds["train"])):
    print(f)
    model = "model_fold" + str(f)
    plot_W(results[model].q_dist.W[2]["mean"], title='W Genotipo')

for f in range(len(folds["train"])):
    model = "model_fold" + str(f)
    plt.figure()
    plt.title("Linear ARD by feature")
    plt.stem(results[model].sparse_K[0].get_params()[1])
    plt.xlabel("Features")
    plt.ylabel("ARD value NO PRIOR")
    plt.show()




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

fold_show = "model_fold3"
# samples to plot: all except the semisupervised ones
f = 3
x_tr, y_tr, x_tst, y_tst = x_seen.loc[folds["train"][f]], y_seen.loc[folds["train"][f]], x_seen.loc[folds["val"][f]], \
                                   y_seen.loc[folds["val"][f]]
s_plot = x_tr.shape[0] + x_tst.shape[0]
y_total = np.concatenate((y_tr, y_tst))
i=0
for key in ryc_y.keys():
    plt.figure()
    plt.title(key)
    scatter = plt.scatter(results[fold_show].q_dist.Z["mean"][:, 5],
                results[fold_show].q_dist.Z["mean"][:, 8],
                c=y_total[:, i])

    plt.legend(handles=scatter.legend_elements()[0], labels=[0, 1])

    # fig = plt.figure()
    # fig.title(key)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(results[fold_show].q_dist.Z["mean"][:, 4],
    #            results[fold_show].q_dist.Z["mean"][:, 5],
    #            results[fold_show].q_dist.Z["mean"][:, 6],
    #            c=y_total[:, i], marker='o')
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_zlabel('Z-axis')
    plt.show()
    # plt.show()
    i+=1

