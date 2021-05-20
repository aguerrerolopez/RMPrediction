import pickle
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "./lib")
from lib import fast_fs_ksshiba_b_ord as ksshiba
import json
import telegram

def notify_ending(message):
    with open('./keys_file.json', 'r') as keys_file:
        k = json.load(keys_file)
        token = k['telegram_token']
        chat_id = k['telegram_chat_id']
    bot = telegram.Bot(token=token)
    bot.sendMessage(chat_id=chat_id, text=message)

# FAMILIAS
# penicilinas = ['AMOXI/CLAV .1', 'PIP/TAZO.1']
penicilinas = ['PIP/TAZO.1']
cephalos = ['CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1']
monobactams = ['AZTREONAM.1']
carbapenems = ['IMIPENEM.1', 'MEROPENEM.1', 'ERTAPENEM.1']
# aminos = ['GENTAMICINA.1', 'TOBRAMICINA.1', 'AMIKACINA.1', 'FOSFOMICINA.1']
aminos = ['GENTAMICINA.1', 'TOBRAMICINA.1']
fluoro = ['CIPROFLOXACINO.1']
otro = ['COLISTINA.1']

full = True

hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 1e-2, "max_it": int(1500)}}

if full:
    data_path = "./data/ryc_data_mediansample_only2-12_noamoxi_noamika_nofosfo_TIC.pkl"
    folds_path = "./data/ryc_5STRATIFIEDfolds_noamoxi_noamika_nofosfo_fullAB.pkl"
    store_path = "Results/mediana_5fold_noard_/RyC_5fold4_noamoxi_amika_fosfo_2-12maldi_fullab_prun"+str(hyper_parameters['sshiba']["pruning_crit"])+".pkl"
    message = "CODIGO TERMINADO EN SERVIDOR: " +"\n Data used: " + data_path + "\n Folds used: " + folds_path +\
              "\n Storage name: "+store_path
else:
    folds_path = "./data/ryc_5folds_NOERTAPENEM.pkl"
    data_path = "./data/ryc_data_NOERTAPENEM_TIC.pkl"
    store_path = "Results/RyC_5fold_FENGEMULT_noERTAPENEM_noprior_prun" + str(
        hyper_parameters['sshiba']["pruning_crit"]) + ".pkl"


with open(data_path, 'rb') as pkl:
    ryc_data = pickle.load(pkl)

maldi = ryc_data['maldi']
fen = ryc_data['fen']
gen = ryc_data['gen']
cmi = ryc_data['cmi']
ab = ryc_data['binary_ab']

results = {}

with open(folds_path, 'rb') as pkl:
    folds = pickle.load(pkl)

c = 0
for f in range(len(folds["train"])):
    f=4
    print("Training fold: ", c)
    x0_tr, x0_val = maldi.loc[folds["train"][f]], maldi.loc[folds["val"][f]]
    x1_tr, x1_val = fen.loc[folds["train"][f]], fen.loc[folds["val"][f]]
    x2_tr, x2_val = gen.loc[folds["train"][f]], gen.loc[folds["val"][f]]
    y_tr, y_val = ab.loc[folds["train"][f]], ab.loc[folds["val"][f]]

    x0_tr = np.vstack(x0_tr.values).astype(float)
    x0_val = np.vstack(x0_val.values).astype(float)

    myModel_mul = ksshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], fs=1)

    # Concatenate the X fold of seen points and the unseen points
    x0 = np.vstack((x0_tr, x0_val)).astype(float)
    print(x0.shape)
    # Fenotype
    x1 = np.vstack((np.vstack(x1_tr.values), np.vstack(x1_val.values))).astype(float)
    # Genotype
    x2 = np.vstack((np.vstack(x2_tr.values), np.vstack(x2_val.values))).astype(float)
    # Both together in one multilabel
    x_fengen = np.hstack((x1, x2))

    # Familias
    y0 = np.vstack((np.vstack(y_tr[penicilinas].values)))
    y1 = np.vstack((np.vstack(y_tr[cephalos].values)))
    y2 = np.vstack((np.vstack(y_tr[monobactams].values)))
    y3 = np.vstack((np.vstack(y_tr[carbapenems].values)))
    y4 = np.vstack((np.vstack(y_tr[aminos].values)))
    y5 = np.vstack((np.vstack(y_tr[fluoro].values)))
    y6 = np.vstack((np.vstack(y_tr[otro].values)))

    X0 = myModel_mul.struct_data(x0, method="reg", V=x0, kernel="linear", sparse_fs=0)
    # X1 = myModel_mul.struct_data(x_fengen, method="mult", sparse=0)
    # X2 = myModel_mul.struct_data(x2, method="mult", sparse=0)
    X1 = myModel_mul.struct_data(x1, method="mult", sparse=0)
    X2 = myModel_mul.struct_data(x2, method="mult", sparse=0)
    Y0 = myModel_mul.struct_data(y0.astype(float), method="mult", sparse=0)
    Y1 = myModel_mul.struct_data(y1.astype(float), method="mult", sparse=0)
    Y2 = myModel_mul.struct_data(y2.astype(float), method="mult", sparse=0)
    Y3 = myModel_mul.struct_data(y3.astype(float), method="mult", sparse=0)
    Y4 = myModel_mul.struct_data(y4.astype(float), method="mult", sparse=0)
    Y5 = myModel_mul.struct_data(y5.astype(float), method="mult", sparse=0)
    Y6 = myModel_mul.struct_data(y6.astype(float), method="mult", sparse=0)

    myModel_mul.fit(X0,
                    X1,
                    X2,
                    Y0, Y1, Y2, Y3, Y4, Y5, Y6,
                    max_iter=hyper_parameters['sshiba']['max_it'],
                    pruning_crit=hyper_parameters['sshiba']['pruning_crit'],
                    verbose=1,
                    feat_crit=1e-2)

    model_name = "model_fold" + str(c)
    results[model_name] = myModel_mul
    c += 1

with open(store_path, 'wb') as f:
    pickle.dump(results, f)

notify_ending(message)
