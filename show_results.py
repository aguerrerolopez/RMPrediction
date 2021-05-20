import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, r2_score, mean_squared_error


def sigmoid(x):
  return np.exp(-np.log(1 + np.exp(-x)))

def plot_W(W, title):
    plt.figure()
    plt.imshow((np.abs(W)), aspect=W.shape[1] / W.shape[0])
    plt.colorbar()
    plt.title(title)
    plt.ylabel('features')
    plt.xlabel('K')
    plt.show()

ryc = True
hgm = False
both = False
no_ertapenem = False

#
modelo_a_cargar = "/Users/alexjorguer/Downloads/RyC_fold0_exponencial_noprior_cephalos_prun0.1.pkl"
# HGM_5fold_fulldata_noprior_prun0.01.pkl
# Load data

if no_ertapenem:
    with open("./data/ryc_data_NOERTAPENEM_TIC.pkl", 'rb') as pkl:
        ryc_data = pickle.load(pkl)
    with open("./data/ryc_5folds_NOERTAPENEM.pkl", 'rb') as pkl:
        ryc_folds = pickle.load(pkl)
else:
    with open("./data/ryc_data_full_TIC.pkl", 'rb') as pkl:
        ryc_data = pickle.load(pkl)
    with open("./data/ryc_5folds_full.pkl", 'rb') as pkl:
        ryc_folds = pickle.load(pkl)

ryc_x = ryc_data["maldi"].copy(deep=True)
ryc_y = ryc_data['binary_ab'].copy(deep=True)
penicilinas = ['AMOXI/CLAV .1', 'PIP/TAZO.1']
cephalos = ['CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1']
monobactams = ['AZTREONAM.1']
carbapenems = ['IMIPENEM.1', 'MEROPENEM.1', 'ERTAPENEM.1']
aminos = ['GENTAMICINA.1', 'TOBRAMICINA.1', 'AMIKACINA.1', 'FOSFOMICINA.1']
fluoro = ['CIPROFLOXACINO.1']
otro = ['COLISTINA.1']

with open("./data/gm_data_sinreplicados_TIC.pkl", 'rb') as pkl:
    hgm_data = pickle.load(pkl)
with open("./data/gm_5folds_sinreplicados.pkl", 'rb') as pkl:
    hgm_folds = pickle.load(pkl)

hgm_x = hgm_data["maldi"].copy(deep=True)
hgm_y = hgm_data['binary_ab'].copy(deep=True)


print("Estamos cargando el modelo de 10GB")
with open(modelo_a_cargar, 'rb') as pkl:
    results = pickle.load(pkl)
print("Por fin lo hemos cargado")

c = 0
lf_imp = []
auc_self_list = []
auc_other_list = []
auc_by_ab = np.zeros((5,15))
r2_list = []
mse_list = []
for f in range(len(ryc_folds["train"])):
    f=0
    print("Validating fold: ", c)
    model = "model_fold" + str(c)

    # HGM data
    hgm_x_tr, hgm_y_tr, hgm_x_tst, hgm_y_tst = hgm_x.loc[hgm_folds["train"][f]], hgm_y.loc[hgm_folds["train"][f]], \
                               hgm_x.loc[hgm_folds["val"][f]], hgm_y.loc[hgm_folds["val"][f]]

    # RYC data
    ryc_x_tr, ryc_y_tr, ryc_x_tst, ryc_y_tst = ryc_x.loc[ryc_folds["train"][f]], ryc_y.loc[ryc_folds["train"][f]], \
                                               ryc_x.loc[ryc_folds["val"][f]], ryc_y.loc[ryc_folds["val"][f]]
    if ryc:
        ryc_y_tst = ryc_y_tst.to_numpy().astype(float)

        penicilinas = ['AMOXI/CLAV .1', 'PIP/TAZO.1']
        cephalos = ['CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1']
        monobactams = ['AZTREONAM.1']
        carbapenems = ['IMIPENEM.1', 'MEROPENEM.1', 'ERTAPENEM.1']
        aminos = ['GENTAMICINA.1', 'TOBRAMICINA.1', 'AMIKACINA.1', 'FOSFOMICINA.1']
        fluoro = ['CIPROFLOXACINO.1']
        otro = ['COLISTINA.1']

        y_pred = results[model].t[3]["mean"][-ryc_y_tst.shape[0]:, :]
        for i_pred, ab in enumerate(cephalos):
            auc_by_ab[c, i_pred] = roc_auc_score(ryc_y_tst[:, i_pred+2], y_pred[:, i_pred], average="weighted")
        #####################################################################
        # y_pred = results[model].t[3]["mean"][-ryc_y_tst.shape[0]:, :]
        # for ab in range(ryc_y_tst.shape[1]):
        #     if len(np.unique(ryc_y_tst[:, ab+2])) == 1:
        #         continue
        #     auc_by_ab[c, ab] = roc_auc_score(ryc_y_tst[:, ab], y_pred[:, ab], average="weighted")
        # ######################################################################
        # aux, aux2, aux3, aux4 = 0,0,0,0
        # for ab in range(ryc_y_tst.shape[1]):
        #     if len(np.unique(ryc_y_tst[:, ab])) == 1:
        #         continue
        #     if ab<2:
        #         y_pred = results[model].t[3]["mean"][-ryc_y_tst.shape[0]:, :]
        #         auc_by_ab[c, ab] = roc_auc_score(ryc_y_tst[:, ab], y_pred[:, aux])
        #         aux+=1
        #     elif ab>=2 and ab<5:
        #         y_pred = results[model].t[4]["mean"][-ryc_y_tst.shape[0]:, :]
        #         auc_by_ab[c, ab] = roc_auc_score(ryc_y_tst[:, ab], y_pred[:, aux2])
        #         aux2 += 1
        #     elif ab==5:
        #         y_pred = results[model].t[5]["mean"][-ryc_y_tst.shape[0]:, :]
        #         auc_by_ab[c, ab] = roc_auc_score(ryc_y_tst[:, ab], y_pred[:, 0])
        #     elif ab>5 and ab<9:
        #         y_pred = results[model].t[6]["mean"][-ryc_y_tst.shape[0]:, :]
        #         auc_by_ab[c, ab] = roc_auc_score(ryc_y_tst[:, ab], y_pred[:, aux3])
        #         aux3 += 1
        #     elif ab>=9 and ab<13:
        #         y_pred = results[model].t[7]["mean"][-ryc_y_tst.shape[0]:, :]
        #         auc_by_ab[c, ab] = roc_auc_score(ryc_y_tst[:, ab], y_pred[:, aux4])
        #         aux4 += 1
        #     elif ab==13:
        #         y_pred = results[model].t[8]["mean"][-ryc_y_tst.shape[0]:, :]
        #         auc_by_ab[c, ab] = roc_auc_score(ryc_y_tst[:, ab], y_pred[:, 0])
        #     elif ab==14:
        #         y_pred = results[model].t[9]["mean"][-ryc_y_tst.shape[0]:, :]
        #         auc_by_ab[c, ab] = roc_auc_score(ryc_y_tst[:, ab], y_pred[:, 0])

        # Predecir:
        # vista0 = results[model].struct_data(np.vstack(ryc_x_tst.values), method="reg", V=np.vstack(ryc_x_tr.values),
        #                                     kernel="linear", sparse_fs=1)
        # vista1 = results[model].struct_data(np.hstack((ryc_data['fen'].loc[ryc_x_tst.index], ryc_data['gen'].loc[ryc_x_tst.index])), method="mult")
        # y_pred = results[model].predict([0, 1], 2, vista0, vista1)
        # print(roc_auc_score(ryc_y_tst, y_pred, average="micro"))

        # # TODO: PREDECIR HOSPITAL B CON MODELO A
        # hgm_x_tst = hgm_data['full'][~hgm_data['full'][hgm_y.keys()].isna().any(axis=1)]['maldi']
        # hgm_x1_tst = hgm_data['full'][~hgm_data['full'][hgm_y.keys()].isna().any(axis=1)].iloc[:, 5:9]
        # hgm_y_tst = hgm_data['full'][~hgm_data['full'][hgm_y.keys()].isna().any(axis=1)][hgm_y.keys()]
        #
        # hgm_x0 = results[model].struct_data(np.vstack(hgm_x_tst.values), method="reg", V=np.vstack(hgm_x_tr.values),
        #                                     kernel="linear", sparse_fs=1)
        # hgm_x1 = results[model].struct_data(np.vstack(hgm_x1_tst.values), method="mult", sparse=0)
        #
        #
        # y_pred = results[model].predict([0, 1, 2], 3, hgm_x0, hgm_x1)
        # auc_other_list.append(roc_auc_score(hgm_y_tst.values, y_pred))

    if hgm:
        hgm_y_tst = hgm_y_tst.to_numpy().astype(float)
        y_pred = results[model].t[2]["mean"][-hgm_y_tst.shape[0]:, :]
        auc_self_list.append(roc_auc_score(hgm_y_tst, y_pred, average="micro"))

        ryc_x0_tst = ryc_data['full'][~ryc_data['full'][ryc_y.keys()].isna().any(axis=1)]['maldi']
        ryc_x1_tst = ryc_data['full'][~ryc_data['full'][ryc_y.keys()].isna().any(axis=1)].iloc[:, 5:9]
        ryc_y_tst = ryc_data['full'][~ryc_data['full'][ryc_y.keys()].isna().any(axis=1)][ryc_y.keys()]

        ryc_x0 = results[model].struct_data(np.vstack(ryc_x0_tst.values), method="reg", V=np.vstack(hgm_x_tr.values), kernel="linear", sparse_fs=1)
        ryc_x1 = results[model].struct_data(np.vstack(ryc_x1_tst.values), method="mult", sparse=0)
        ryc_y_tst = np.vstack(ryc_y_tst.values)

        # y_pred = results[model].predict([0, 1], 2, ryc_x0, ryc_x1)
        # auc_other_list.append(roc_auc_score(ryc_y_tst, y_pred))

        # TODO: PREDECIR HOSPITAL B CON MODELO A
        # results[model].predict([0,1],[2], hgm_x0, hgm_x1)

    if both:
        y_tst = np.vstack((hgm_y_tst.to_numpy().astype(float), ryc_y_tst.to_numpy().astype(float)))
        y_pred = results[model].t[2]["mean"][-(hgm_y_tst.shape[0]+ryc_y_tst.shape[0]):, :]
        auc_self_list.append(roc_auc_score(y_tst, y_pred, average="micro"))

        k_importance = np.argsort(np.abs(np.mean(results[model].q_dist.W[2]['mean'], axis=0)))
        x0 = np.vstack(np.vstack(hgm_x_tst.values), np.vstack(ryc_x_tst.values))
        X0 = results[model].struct_data(x0, method="reg",
                                     V=np.vstack((np.vstack(x0_val_gm.values), np.vstack(x0_val_ryc.values))),
                                     kernel="linear", sparse_fs=1)
        X1 = results[model].struct_data(np.hstack((x1, x2)), method="mult", sparse=0)

    lf_imp.append(np.argwhere(np.mean(np.abs(results[model].q_dist.W[2]["mean"]), axis=0) > 0.5))

    print(auc_self_list)
    c += 1

# TODO: Dibujar pesos ARD para cad fold
for f in range(len(ryc_folds["train"])):
    model = "model_fold" + str(f)
    plt.figure()
    plt.title("Init "+str(f))
    plt.stem(results[model].sparse_K[0].get_params()[1])
    plt.xlabel("Features")
    plt.ylabel("ARD value NO PRIOR")
    plt.show()

# TODO: ANALIZAR ESPACIO LATENTE
for f in range(len(ryc_folds["train"])):
    model = "model_fold" + str(f)
    plot_W(results[model].q_dist.W[3]['mean'], title=model)

# TODO: predict with 1 k less.




# fold_show = "model_fold3"
# # samples to plot: all except the semisupervised ones
# f = 3
# x_tr, y_tr, x_tst, y_tst = x_seen.loc[folds["train"][f]], y_seen.loc[folds["train"][f]], x_seen.loc[folds["val"][f]], \
#                                    y_seen.loc[folds["val"][f]]
# s_plot = x_tr.shape[0] + x_tst.shape[0]
# y_total = np.concatenate((y_tr, y_tst))
# i=0
# for key in ryc_y.keys():
#     plt.figure()
#     plt.title(key)
#     scatter = plt.scatter(results[fold_show].q_dist.Z["mean"][:, 5],
#                 results[fold_show].q_dist.Z["mean"][:, 8],
#                 c=y_total[:, i])
#
#     plt.legend(handles=scatter.legend_elements()[0], labels=[0, 1])
#
#     # fig = plt.figure()
#     # fig.title(key)
#     # ax = fig.add_subplot(111, projection='3d')
#     # ax.scatter(results[fold_show].q_dist.Z["mean"][:, 4],
#     #            results[fold_show].q_dist.Z["mean"][:, 5],
#     #            results[fold_show].q_dist.Z["mean"][:, 6],
#     #            c=y_total[:, i], marker='o')
#     # ax.set_xlabel('X-axis')
#     # ax.set_ylabel('Y-axis')
#     # ax.set_zlabel('Z-axis')
#     plt.show()
#     # plt.show()
#     i+=1
#



fig, ax = plt.subplots()
width = 0.15
x = np.arange(0,14)
rects1 = ax.bar(x - 4*width/2, auc_by_ab[1, :-1], width, label='Init1')
rects0 = ax.bar(x - 2*width/2, auc_by_ab[0, :-1], width, label='Init0')
rects2 = ax.bar(x, auc_by_ab[2, :-1], width, label='Init2')
rects4 = ax.bar(x + 2*width/2, auc_by_ab[4, :-1], width, label='Init4')
rects3 = ax.bar(x + 4*width/2, auc_by_ab[3, :-1], width, label='Init3')
ax.axhline(y=0.5, color='r')
ax.set_ylabel('AUC')
ax.set_ylim(bottom=0.2, top=1)
ax.set_title('HGM NO prior: AUC by AB in fold 0')
ax.set_xticks(x)
ax.set_xticklabels(ryc_y.keys().to_list()[:-1], rotation=30, fontsize='xx-small')
ax.legend()
fig.tight_layout()

plt.show()