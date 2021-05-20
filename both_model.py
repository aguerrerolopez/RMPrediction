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

full = False

hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 1e-1, "max_it": int(1500)}}

ryc_data_path = "./data/ryc_data_full_TIC.pkl"
ryc_folds_path = "./data/ryc_5folds_full.pkl"
gm_folds_path = "./data/gm_5folds_sinreplicados.pkl"
gm_data_path = "./data/gm_data_sinreplicados_TIC.pkl"
store_path = "Results/Both_5fold_hospitalview_fengen_yesprior_newkernelproyectadotodoX_prun"+str(hyper_parameters['sshiba']["pruning_crit"])+".pkl"
message = "CODIGO TERMINADO EN SERVIDOR: " +"\n AMBOS HOSPITALES" "\n Storage name: "+store_path

with open(gm_data_path, 'rb') as pkl:
    gm_data = pickle.load(pkl)
with open(ryc_data_path, 'rb') as pkl:
    ryc_data = pickle.load(pkl)

maldi_gm = gm_data['maldi']
fen_gm = gm_data['fen']
cmi_gm = gm_data['cmi']
ab_gm = gm_data['binary_ab']

maldi_ryc = ryc_data['maldi']
fen_ryc = ryc_data['fen']
gen_ryc = ryc_data['gen']
cmi_ryc = ryc_data['cmi']
ab_ryc = ryc_data['binary_ab']

results = {}

with open(gm_folds_path, 'rb') as pkl:
    gm_folds = pickle.load(pkl)
with open(ryc_folds_path, 'rb') as pkl:
    ryc_folds = pickle.load(pkl)


c = 0
for f in range(len(gm_folds["train"])):
    print("Training fold: ", c)
    x0_tr_gm, x0_val_gm = maldi_gm.loc[gm_folds["train"][f]], maldi_gm.loc[gm_folds["val"][f]]
    x1_tr_gm, x1_val_gm = fen_gm.loc[gm_folds["train"][f]], fen_gm.loc[gm_folds["val"][f]]
    y1_tr_gm, y1_val_gm = ab_gm.loc[gm_folds["train"][f]], ab_gm.loc[gm_folds["val"][f]]

    x0_tr_ryc, x0_val_ryc = maldi_ryc.loc[ryc_folds["train"][f]], maldi_ryc.loc[ryc_folds["val"][f]]
    x1_tr_ryc, x1_val_ryc = fen_ryc.loc[ryc_folds["train"][f]], fen_ryc.loc[ryc_folds["val"][f]]
    x2_tr_ryc, x2_val_ryc = gen_ryc.loc[ryc_folds["train"][f]], gen_ryc.loc[ryc_folds["val"][f]]
    y1_tr_ryc, y1_val_ryc = ab_ryc.loc[ryc_folds["train"][f]], ab_ryc.loc[ryc_folds["val"][f]]

    # MALDI VIEW
    x0 = np.vstack((np.vstack(x0_tr_gm.values), np.vstack(x0_tr_ryc.values),
                    np.vstack(x0_val_gm.values),  np.vstack(x0_val_ryc.values))).astype(float)
    # FEN VIEW
    x1 = np.vstack((np.vstack(x1_tr_gm.values), np.vstack(x1_tr_ryc.values),
                    np.vstack(x1_val_gm.values), np.vstack(x1_val_ryc.values))).astype(float)
    # GEN VIEW
    x2 = np.vstack((np.nan*np.ones((np.vstack(x0_tr_gm.values).shape[0], 4)), np.vstack(x2_tr_ryc.values),
                    np.nan*np.ones((np.vstack(x0_val_gm.values).shape[0], 4)), np.vstack(x2_val_ryc.values))).astype(float)
    # HOSPITAL VIEW
    x3 = np.vstack((np.ones((np.vstack(x0_tr_gm.values).shape[0], 1)), np.zeros((np.vstack(x0_tr_ryc.values).shape[0],1)),
                    np.ones((np.vstack(x0_val_gm.values).shape[0], 1)), np.zeros((np.vstack(x0_val_ryc.values).shape[0], 1))))
    # ANTIBIOTIC VIEW
    y1 = np.vstack((np.vstack(y1_tr_gm.values), np.vstack(y1_tr_ryc.values))).astype(float)

    myModel_mul = ksshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], fs=1)

    X0 = myModel_mul.struct_data(x0, method="reg", V=x0,
                                 kernel="linear", sparse_fs=1)
    X1 = myModel_mul.struct_data(np.hstack((x1, x2)), method="mult", sparse=0)
    X2 = myModel_mul.struct_data(x3, method="mult", sparse=0)
    Y1 = myModel_mul.struct_data(y1, method="mult", sparse=0)

    myModel_mul.fit(X0,
                    X1,
                    X2,
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