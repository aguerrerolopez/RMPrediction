import pickle
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "./lib")
#from lib import fast_fs_ksshiba_b_ord as ksshiba
from lib import ksshiba_new as ksshiba
import json
import telegram
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, roc_auc_score, r2_score, mean_squared_error

def notify_ending(message):
    with open('./keys_file.json', 'r') as keys_file:
        k = json.load(keys_file)
        token = k['telegram_token']
        chat_id = k['telegram_chat_id']
    bot = telegram.Bot(token=token)
    bot.sendMessage(chat_id=chat_id, text=message)


# FAMILIAS
penicilinas = ['AMOXI/CLAV .1', 'PIP/TAZO.1']
cephalos = ['CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1']
monobactams = ['AZTREONAM.1']
carbapenems = ['IMIPENEM.1', 'MEROPENEM.1', 'ERTAPENEM.1']
aminos = ['GENTAMICINA.1', 'TOBRAMICINA.1', 'AMIKACINA.1', 'FOSFOMICINA.1']
fluoro = ['CIPROFLOXACINO.1']
otro = ['COLISTINA.1']

full = True

hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 1e-1, "max_it": int(1500)}}

if full:
    familia = "monobactams"
    data_path = "./data/ryc_data_full_TIC.pkl"
    folds_path = "./data/ryc_5folds_"+familia+".pkl"
    store_path = "Results/RyC_fold0_pruebanewksshiba_noprior_"+familia+"_prun"+str(hyper_parameters['sshiba']["pruning_crit"])+".pkl"
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
    if f>0:
        break
    print("Training fold: ", c)
    x0_tr, x0_val = maldi.loc[folds["train"][f]], maldi.loc[folds["val"][f]]
    x1_tr, x1_val = fen.loc[folds["train"][f]], fen.loc[folds["val"][f]]
    x2_tr, x2_val = gen.loc[folds["train"][f]], gen.loc[folds["val"][f]]
    y_tr, y_val = ab.loc[folds["train"][f]], ab.loc[folds["val"][f]]

    x0_tr = np.vstack(x0_tr.values).astype(float)
    x0_val = np.vstack(x0_val.values).astype(float)


    # Familias
    if familia=="penicilinas":
        y0 = np.vstack((np.vstack(y_tr[penicilinas].values)))
    elif familia == "cephalos":
        y0 = np.vstack((np.vstack(y_tr[cephalos].values)))
    elif familia == "monobactams":
        y0 = np.vstack((np.vstack(y_tr[monobactams].values)))
        y0_tst = np.vstack((np.vstack(y_val[monobactams].values)))
    elif familia == "carbapenems":
        y0 = np.vstack((np.vstack(y_tr[carbapenems].values)))
    elif familia == "aminos":
        y0 = np.vstack((np.vstack(y_tr[aminos].values)))
    elif familia == "fluoro":
        y0 = np.vstack((np.vstack(y_tr[fluoro].values)))
    elif familia == "otro":
        y0 = np.vstack((np.vstack(y_tr[otro].values)))

    
    clf = SVC(C=1, kernel="linear", probability=1)
    clf.fit(x0_tr, y0.ravel())
    y_pred = clf.predict_proba(x0_val)
    print(roc_auc_score(y0_tst, y_pred[:, -1]))
    # model_name = "model_fold" + str(c)
    # results[model_name] = myModel_mul
    c += 1

with open(store_path, 'wb') as f:
    pickle.dump(results, f)

notify_ending(message)
