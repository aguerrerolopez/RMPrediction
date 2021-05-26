import pickle
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, r2_score, mean_squared_error


def sigmoid(x):
  return np.exp(-np.log(1 + np.exp(-x)))

def plot_W(W, title):
    plt.figure()
    plt.imshow((np.abs(W)), aspect=W.shape[1] / W.shape[0])
    plt.colorbar()
    plt.title(title)
    plt.yticks(np.arange(0, W.shape[0]))
    plt.ylabel('features')
    plt.xlabel('K')
    plt.show()


ryc = False
hgm = True
both = False
no_ertapenem = False
resultados = []

familia="penicilinas"
W_byfold = np.zeros((10, 10000))
models_sel = [None]*10
print("############################## "+familia+" ############################## ")
for fold in range(10):

    if ryc: hospital="RyC"
    elif hgm: hospital="HGM"
    elif both: hospital="Both"
    modelo_a_cargar = "./Results/mediana_10fold_rbf/"+hospital+"_10fold"+str(fold)+"_muestrascompensadas_2-12maldi_"+familia+"_prun0.1.pkl"

    if ryc:
        familias = {
                    "penicilinas": ['AMOXI/CLAV .1', 'PIP/TAZO.1'],
                    "cephalos": ['CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1'],
                    "monobactams": ['AZTREONAM.1'],
                    "carbapenems": ['IMIPENEM.1', 'MEROPENEM.1', 'ERTAPENEM.1'],
                    "fluoro":['CIPROFLOXACINO.1'],
                    "aminos": ['GENTAMICINA.1', 'TOBRAMICINA.1'],
                    "otros":['COLISTINA.1']
                    }
        with open("./data/ryc_data_mediansample_only2-12_noamika_nofosfo_TIC.pkl", 'rb') as pkl:
            data = pickle.load(pkl)
        with open("./data/RyC_5STRATIFIEDfolds_noamika_nofosfo_"+familia+".pkl", 'rb') as pkl:
            folds = pickle.load(pkl)

    if hgm:
        familias = {
                    "penicilinas": ['AMOXI/CLAV ', 'PIP/TAZO'],
                    "cephalos": ['CEFTAZIDIMA', 'CEFOTAXIMA', 'CEFEPIME'],
                    "monobactams": ['AZTREONAM'],
                    "carbapenems": ['IMIPENEM', 'MEROPENEM', 'ERTAPENEM'],
                    "aminos": ['GENTAMICINA', 'TOBRAMICINA', 'AMIKACINA', 'FOSFOMICINA'],
                    "fluoro":['CIPROFLOXACINO'],
                    "otros":['COLISTINA']
                    }
        with open("./data/hgm_data_mediansample_only2-12_TIC.pkl", 'rb') as pkl:
            data = pickle.load(pkl)
        with open("./data/HGM_10STRATIFIEDfolds_muestrascompensadas_"+familia+".pkl", 'rb') as pkl:
            folds = pickle.load(pkl)

    if both:
        familias_ryc = {
                    "penicilinas": ['AMOXI/CLAV .1', 'PIP/TAZO.1'],
                    "cephalos": ['CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1'],
                    "monobactams": ['AZTREONAM.1'],
                    "carbapenems": ['IMIPENEM.1', 'MEROPENEM.1', 'ERTAPENEM.1'],
                    "aminos": ['GENTAMICINA.1', 'TOBRAMICINA.1', 'AMIKACINA.1', 'FOSFOMICINA.1'],
                    "fluoro":['CIPROFLOXACINO.1'],
                    "otros":['COLISTINA.1']
                    }
        familias_hgm = {
            "penicilinas": ['AMOXI/CLAV ', 'PIP/TAZO'],
            "cephalos": ['CEFTAZIDIMA', 'CEFOTAXIMA', 'CEFEPIME'],
            "monobactams": ['AZTREONAM'],
            "carbapenems": ['IMIPENEM', 'MEROPENEM', 'ERTAPENEM'],
            "aminos": ['GENTAMICINA', 'TOBRAMICINA', 'AMIKACINA', 'FOSFOMICINA'],
            "fluoro":['CIPROFLOXACINO'],
            "otros":['COLISTINA']
            }
        familias = familias_hgm
        with open("./data/ryc_data_mediansample_only2-12_TIC.pkl", 'rb') as pkl:
            ryc_data = pickle.load(pkl)
        with open("./data/RyC_5STRATIFIEDfolds_both_"+familia+".pkl", 'rb') as pkl:
            ryc_folds = pickle.load(pkl)
        with open("./data/hgm_data_mediansample_only2-12_TIC.pkl", 'rb') as pkl:
            hgm_data = pickle.load(pkl)
        with open("./data/HGM_10STRATIFIEDfolds_muestrascompensadas_"+familia+".pkl", 'rb') as pkl:
            hgm_folds = pickle.load(pkl)
        hgm_y = hgm_data["binary_ab"]
        ryc_y = ryc_data["binary_ab"]

    else:
        data_x = data["maldi"]
        data_y = data['binary_ab']

    with open(modelo_a_cargar, 'rb') as pkl:
        results = pickle.load(pkl)

    if both: 
        ryc_y_tst = ryc_y[familias_ryc[familia]].loc[ryc_folds["val"][fold]].to_numpy()
        hgm_y_tst = hgm_y[familias_hgm[familia]].loc[hgm_folds["val"][fold]].to_numpy()
        y_tst = np.vstack((hgm_y_tst, ryc_y_tst))

    else:
        # import missingno as msno
        # print(msno.matrix(data_y[familias[familia]].loc[folds["train"][fold]]))
        # for ab in familias[familia]:
        #     print(data_y[ab].loc[folds["train"][fold]].value_counts())
        y_tst = data_y[familias[familia]].loc[folds["val"][fold]]
        y_tst = y_tst.to_numpy().astype(float)

        
    c = 0
    lf_imp = []
    auc_by_ab = np.zeros((5,len(familias[familia])))
    W_2norm=0

    elbo=[]
    Ks = [1e5]
    for i in range(5):
        model = "model_fold" + str(c)

        elbo.append(results[model].L[-1])

        if results[model].q_dist.W[2]["mean"].shape[1]<Ks[-1]:
            models_sel[fold] = results[model]
        Ks.append(results[model].q_dist.W[0]["mean"].shape[1])

        if hgm: y_pred = results[model].t[2]["mean"][-y_tst.shape[0]:, :]
        elif ryc: y_pred = results[model].t[3]["mean"][-y_tst.shape[0]:, :]
        else: y_pred = results[model].t[4]["mean"][-y_tst.shape[0]:, :]
        for i_pred in range(auc_by_ab.shape[1]):
            auc_by_ab[c, i_pred] = roc_auc_score(y_tst[:, i_pred], y_pred[:, i_pred])
            
        # lf_imp.append(np.argwhere(np.mean(np.abs(results[model].q_dist.W[2]["mean"]), axis=0) > 0.5))
        X=np.vstack((np.vstack(data_x.loc[folds["train"][fold]].values), np.vstack(data_x.loc[folds["val"][fold]].values)))
        W_2norm += np.linalg.norm(X.T@results[model].q_dist.W[0]['mean'], axis=1)

        c += 1
    elbo_max = np.argmax(elbo)
    k_min = np.argmin(Ks)


    W_byfold[fold, :] = W_2norm/5

    # # TODO: Dibujar pesos ARD para cada fold
    # f, axs = plt.subplots(1,5, figsize=(16,9))
    # f.suptitle('ARD WEIGHTS '+familia)
    # axs[0].set_title("Init 0")
    # axs[0].stem(results["model_fold0"].sparse_K[0].get_params()[1])
    # axs[1].set_title("Init 1")
    # axs[1].stem(results["model_fold1"].sparse_K[0].get_params()[1])
    # axs[2].set_title("Init 2")
    # axs[2].stem(results["model_fold2"].sparse_K[0].get_params()[1])
    # axs[3].set_title("Init 3")
    # axs[3].stem(results["model_fold3"].sparse_K[0].get_params()[1])
    # axs[4].set_title("Init 4")
    # axs[4].stem(results["model_fold4"].sparse_K[0].get_params()[1])
    # for ax in axs.flat:
    #     ax.label_outer()

    # # TODO: Dibujar W primal space
    

    # f, axs = plt.subplots(1,5, figsize=(16,9))
    # f.suptitle('2norm of W_d primal '+familia)

    # X=np.vstack((np.vstack(ryc_x.loc[ryc_folds["train"][0]].values), np.vstack(ryc_x.loc[ryc_folds["val"][0]].values)))
    # Wprimal = X.T@results["model_fold0"].q_dist.W[0]['mean']
    # W2norm_d = np.linalg.norm(Wprimal, axis=1)
    # ax = axs[0]
    # ax.set_title("Init 0")
    # ax.stem(W2norm_d)

    # X=np.vstack((np.vstack(ryc_x.loc[ryc_folds["train"][1]].values), np.vstack(ryc_x.loc[ryc_folds["val"][1]].values)))
    # Wprimal = X.T@results["model_fold1"].q_dist.W[0]['mean']
    # W2norm_d = np.linalg.norm(Wprimal, axis=1)
    # ax = axs[1]
    # ax.set_title("Init 1")
    # ax.stem(W2norm_d)

    # X=np.vstack((np.vstack(ryc_x.loc[ryc_folds["train"][2]].values), np.vstack(ryc_x.loc[ryc_folds["val"][2]].values)))
    # Wprimal = X.T@results["model_fold2"].q_dist.W[0]['mean']
    # W2norm_d = np.linalg.norm(Wprimal, axis=1)
    # ax = axs[2]
    # ax.set_title("Init 2")
    # ax.stem(W2norm_d)

    # X=np.vstack((np.vstack(ryc_x.loc[ryc_folds["train"][3]].values), np.vstack(ryc_x.loc[ryc_folds["val"][3]].values)))
    # Wprimal = X.T@results["model_fold3"].q_dist.W[0]['mean']
    # W2norm_d = np.linalg.norm(Wprimal, axis=1)
    # ax = axs[3]
    # ax.set_title("Init 3")
    # ax.stem(W2norm_d)

    # X=np.vstack((np.vstack(ryc_x.loc[ryc_folds["train"][4]].values), np.vstack(ryc_x.loc[ryc_folds["val"][4]].values)))
    # Wprimal = X.T@results["model_fold4"].q_dist.W[0]['mean']
    # W2norm_d = np.linalg.norm(Wprimal, axis=1)
    # ax = axs[4]
    # ax.set_title("Init 4")
    # ax.stem(W2norm_d)


    # # TODO: Dibujar W*ARD
    # f, axs = plt.subplots(1,5, figsize=(16,9))
    # f.suptitle('ARD by W2norm of '+familia)

    # X=np.vstack((np.vstack(ryc_x.loc[ryc_folds["train"][0]].values), np.vstack(ryc_x.loc[ryc_folds["val"][0]].values)))
    # Wprimal = X.T@results["model_fold0"].q_dist.W[0]['mean']
    # W2norm_d = np.linalg.norm(Wprimal, axis=1)
    # ard_w = results["model_fold0"].sparse_K[0].get_params()[1]*W2norm_d
    # ax = axs[0]
    # ax.set_title("Init 0")
    # ax.stem(ard_w)

    # X=np.vstack((np.vstack(ryc_x.loc[ryc_folds["train"][1]].values), np.vstack(ryc_x.loc[ryc_folds["val"][1]].values)))
    # Wprimal = X.T@results["model_fold1"].q_dist.W[0]['mean']
    # W2norm_d = np.linalg.norm(Wprimal, axis=1)
    # ard_w = results["model_fold1"].sparse_K[0].get_params()[1]*W2norm_d
    # ax = axs[1]
    # ax.set_title("Init 1")
    # ax.stem(ard_w)

    # X=np.vstack((np.vstack(ryc_x.loc[ryc_folds["train"][2]].values), np.vstack(ryc_x.loc[ryc_folds["val"][2]].values)))
    # Wprimal = X.T@results["model_fold2"].q_dist.W[0]['mean']
    # W2norm_d = np.linalg.norm(Wprimal, axis=1)
    # ard_w = results["model_fold2"].sparse_K[0].get_params()[1]*W2norm_d
    # ax = axs[2]
    # ax.set_title("Init 2")
    # ax.stem(ard_w)

    # X=np.vstack((np.vstack(ryc_x.loc[ryc_folds["train"][3]].values), np.vstack(ryc_x.loc[ryc_folds["val"][3]].values)))
    # Wprimal = X.T@results["model_fold3"].q_dist.W[0]['mean']
    # W2norm_d = np.linalg.norm(Wprimal, axis=1)
    # ard_w = results["model_fold3"].sparse_K[0].get_params()[1]*W2norm_d
    # ax = axs[3]
    # ax.set_title("Init 3")
    # ax.stem(ard_w)

    # X=np.vstack((np.vstack(ryc_x.loc[ryc_folds["train"][4]].values), np.vstack(ryc_x.loc[ryc_folds["val"][4]].values)))
    # Wprimal = X.T@results["model_fold4"].q_dist.W[0]['mean']
    # W2norm_d = np.linalg.norm(Wprimal, axis=1)
    # ard_w = results["model_fold4"].sparse_K[0].get_params()[1]*W2norm_d
    # ax = axs[4]
    # ax.set_title("Init 4")
    # ax.stem(ard_w)



    # # TODO: Dibujar auc por familia
    fig, ax = plt.subplots()
    width = 0.15
    x = np.arange(len(familias[familia]))
    rects1 = ax.bar(x - 4*width/2, auc_by_ab[0, :], width, label='Init0')
    rects0 = ax.bar(x - 2*width/2, auc_by_ab[1, :], width, label='Init1')
    rects2 = ax.bar(x, auc_by_ab[2, :], width, label='Init2')
    rects4 = ax.bar(x + 2*width/2, auc_by_ab[3, :], width, label='Init3')
    rects3 = ax.bar(x + 4*width/2, auc_by_ab[4, :], width, label='Init4')
    ax.axhline(y=0.5, color='r')
    ax.set_ylabel('AUC')
    ax.set_ylim(bottom=0.2, top=1)
    ax.set_title('RyC '+familia+': AUC by AB in fold '+str(fold))
    ax.set_xticks(x)
    ax.set_xticklabels(familias[familia], rotation=30, fontsize='xx-small')
    ax.legend()
    fig.tight_layout()

    # plt.show()
    resultados.append(auc_by_ab)

    print("Results"+str(np.mean(auc_by_ab, axis=0))+"+/-"+str(np.std(auc_by_ab, axis=0)))
    # print(np.mean(np.hstack(resultados), axis=0))
    # print(np.std(np.hstack(resultados), axis=0))


print("Resultados finales:")
print("Mean")
print(np.mean(np.vstack(resultados), axis=0))
print("STD")
print(np.std(np.vstack(resultados), axis=0))


# TODO: Sacar vistas comunes
from mpl_toolkits.axes_grid1 import make_axes_locatable
for f in range(10):
    plt.figure(figsize=[20,10])
    ax = plt.gca()
    plt.title("Latent space distribution by views  for fold "+str(f))
    matrix_views = np.zeros((3,models_sel[f].q_dist.W[0]['mean'].shape[1]))
    matrix_views[0, :]=np.mean(np.abs(models_sel[f].q_dist.W[0]['mean']), axis=0)
    matrix_views[1, :]=np.mean(np.abs(models_sel[f].q_dist.W[1]['mean']), axis=0)  
    matrix_views[2, :]=np.mean(np.abs(models_sel[f].q_dist.W[2]['mean']), axis=0)    
    plt.xlabel("K features latent space")
    plt.yticks(np.arange(3), ["Maldi", "Fenotype", "Antibiotic"])
    plt.xticks(range(models_sel[f].q_dist.W[0]['mean'].shape[1]))
    im = ax.imshow(matrix_views, cmap="binary")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()


# # TODO: Sacar la W por familia SOLO SI EL KERNEL ES LINEAL
# plt.figure()
# plt.title(familia+": W primal space")
# for i in range(10):
#     label = "Fold "+str(i)
#     plt.plot(W_byfold[i, :], label=label)
# plt.legend()
# plt.show()
# plt.figure()
# plt.title(familia+": Random sample with W primal space over it")
# plt.plot(np.vstack(data_x.loc[folds["train"][fold]].sample(1).values).ravel(), label="A random sample")
# plt.plot(np.mean(W_byfold, axis=0), label="W primal space mean", alpha=0.2)
# plt.legend()
# plt.show()
# W_byfam = np.zeros((7, 10000))
# for j, familia in enumerate(familias):
#     with open("./data/hgm_data_mediansample_only2-12_TIC.pkl", 'rb') as pkl:
#         data = pickle.load(pkl)
#     with open("./data/HGM_10STRATIFIEDfolds_muestrascompensadas_"+familia+".pkl", 'rb') as pkl:
#         folds = pickle.load(pkl)
#     data_x = data['maldi']
#     W_byfold = np.zeros((10, 10000))

#     for fold in range(10):
#         modelo_a_cargar = "./Results/mediana_10fold_rbf/"+hospital+"_10fold"+str(fold)+"_muestrascompensadas_2-12maldi_"+familia+"_prun0.1.pkl"
#         with open(modelo_a_cargar, 'rb') as pkl:
#             results = pickle.load(pkl)
#         W_2norm=0
#         for i in range(5):
#             model="model_fold"+str(i)
#             X=np.vstack((np.vstack(data_x.loc[folds["train"][fold]].values), np.vstack(data_x.loc[folds["val"][fold]].values)))
#             W_2norm += np.linalg.norm(X.T@results[model].q_dist.W[0]['mean'], axis=1)

#         W_byfold[fold, :] = W_2norm/5

#     W_byfam[j, :] = np.mean(W_byfold, axis=0)

# plt.figure(figsize=[20, 10])
# plt.title("W primal space by family")
# for j, familia in enumerate(familias):
#     plt.plot(W_byfam[j, :], label=familia)
# plt.plot(np.vstack(data_x.loc[folds["train"][fold]].sample(1).values).ravel(), label="A random sample")
# plt.legend()
# plt.show()