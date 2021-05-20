import pickle
import sys
import numpy as np
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


hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 1e-1, "max_it": int(1500)}}

folds_path = "./data/gm_5folds_sinreplicados.pkl"
data_path = "./data/gm_data_sinreplicados_TIC.pkl"
store_path = "Results/HGM_fold0_lowprior_proyectXtr_prun"+str(hyper_parameters['sshiba']["pruning_crit"])+".pkl"
message = "CODIGO TERMINADO EN SERVIDOR: " +"\n Data used: " + data_path + "\n Folds used: " + folds_path +\
              "\n Storage name: "+store_path

with open(data_path, 'rb') as pkl:
   gm_data = pickle.load(pkl)

maldi = gm_data['maldi']
fen = gm_data['fen']
gen = gm_data['gen']
cmi = gm_data['cmi']
ab = gm_data['binary_ab']

results = {}

with open(folds_path, 'rb') as pkl:
    folds = pickle.load(pkl)

c = 0
for f in range(len(folds["train"])):
    f=0
    print("Training fold: ", c)
    x0_tr, x0_val = maldi.loc[folds["train"][f]], maldi.loc[folds["val"][f]]
    x1_tr, x1_val = fen.loc[folds["train"][f]], fen.loc[folds["val"][f]]
    # x2_tr, x2_tst = gen.loc[folds["train"][f]], gen.loc[folds["val"][f]]
    y1_tr, y1_val = ab.loc[folds["train"][f]], ab.loc[folds["val"][f]]

    x0_tr= np.vstack(x0_tr.values).astype(float)
    x0_val = np.vstack(x0_val.values).astype(float)

    myModel_mul = ksshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], fs=1)

    # Concatenate the X fold of seen points and the unseen points
    x0 = np.vstack((x0_tr, x0_val)).astype(float)
    x1 = np.vstack((np.vstack(x1_tr.values), np.vstack(x1_val.values))).astype(float)
    y1 = np.vstack((np.vstack(y1_tr.values)))

    X0 = myModel_mul.struct_data(x0, method="reg", V=x0_tr, kernel="linear", sparse_fs=1)
    X1 = myModel_mul.struct_data(x1, method="mult", sparse=0)
    Y1 = myModel_mul.struct_data(y1.astype(float), method="mult", sparse=0)

    myModel_mul.fit(X0,
                    X1,
                    Y1,
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