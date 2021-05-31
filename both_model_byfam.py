import pickle
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "./lib")
from lib import fast_fs_ksshiba_b_ord as ksshiba
# from lib import ksshiba_new as ksshiba
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
familias_hgm = { "penicilinas": ['AMOXI/CLAV ', 'PIP/TAZO'],
                    "cephalos": ['CEFTAZIDIMA', 'CEFOTAXIMA', 'CEFEPIME'],
                    "monobactams": ['AZTREONAM'],
                    "carbapenems": ['IMIPENEM', 'MEROPENEM', 'ERTAPENEM'],
                    "aminos": ['GENTAMICINA', 'TOBRAMICINA', 'AMIKACINA', 'FOSFOMICINA'],
                    "fluoro":['CIPROFLOXACINO'],
                    "otros":['COLISTINA']
                    }
familias_ryc = { "penicilinas": ['AMOXI/CLAV .1', 'PIP/TAZO.1'],
                    "cephalos": ['CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1'],
                    "monobactams": ['AZTREONAM.1'],
                    "carbapenems": ['IMIPENEM.1', 'MEROPENEM.1', 'ERTAPENEM.1'],
                    "aminos": ['GENTAMICINA.1', 'TOBRAMICINA.1', 'AMIKACINA.1', 'FOSFOMICINA.1'],
                    "fluoro":['CIPROFLOXACINO.1'],
                    "otros":['COLISTINA.1']
                    }

full = True

hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 1e-1, "max_it": int(1500)}}

############# CARGAR DATOS ######################
data_path = "./data/hgm_data_mediansample_only2-12_TIC.pkl"
with open(data_path, 'rb') as pkl:
    hgm_data = pickle.load(pkl)

data_path = "./data/ryc_data_mediansample_only2-12_TIC.pkl"
with open(data_path, 'rb') as pkl:
    ryc_data = pickle.load(pkl)

for familia in familias_ryc:

    folds_path = "./data/HGM_10STRATIFIEDfolds_muestrascompensadas_"+familia+".pkl"
    with open(folds_path, 'rb') as pkl:
       folds_hgm = pickle.load(pkl)

    folds_path = "./data/RYC_10STRATIFIEDfolds_muestrascompensadas_"+familia+".pkl"
    with open(folds_path, 'rb') as pkl:
       folds_ryc = pickle.load(pkl)

    for f in range(10):
        print(familia)

        x0_tr = np.vstack((np.vstack(hgm_data['maldi'].loc[folds_hgm["train"][f]].values), np.vstack(ryc_data['maldi'].loc[folds_ryc["train"][f]].values)))
        x0_tst = np.vstack((np.vstack(hgm_data['maldi'].loc[folds_hgm["val"][f]].values), np.vstack(ryc_data['maldi'].loc[folds_ryc["val"][f]].values)))

        x1_tr = np.vstack((np.vstack(hgm_data['fen'].loc[folds_hgm["train"][f]].values), np.vstack(ryc_data['fen'].loc[folds_ryc["train"][f]].values)))
        x1_tst =  np.vstack((np.vstack(hgm_data['fen'].loc[folds_hgm["val"][f]].values), np.vstack(ryc_data['fen'].loc[folds_ryc["val"][f]].values)))

        x2_tr = np.vstack((np.nan*np.zeros(hgm_data['fen'].loc[folds_hgm["train"][f]].shape), np.vstack(ryc_data['gen'].loc[folds_ryc["train"][f]].values)))
        x2_tst = np.vstack((np.nan*np.zeros(hgm_data['fen'].loc[folds_hgm["val"][f]].shape), np.vstack(ryc_data['gen'].loc[folds_ryc["val"][f]].values)))

        x3_tr = np.vstack((np.ones(hgm_data['fen'].loc[folds_hgm["train"][f]].shape), np.zeros(ryc_data['fen'].loc[folds_ryc["train"][f]].shape)))
        x3_tst = np.vstack((np.ones(hgm_data['fen'].loc[folds_hgm["val"][f]].shape), np.zeros(ryc_data['fen'].loc[folds_ryc["val"][f]].shape)))

        y0_tr = np.vstack((np.vstack(hgm_data['binary_ab'][familias_hgm[familia]].loc[folds_hgm["train"][f]].values), 
                           np.vstack(ryc_data['binary_ab'][familias_ryc[familia]].loc[folds_ryc["train"][f]].values)))


        x0 = np.vstack((x0_tr, x0_tst))
        x1 = np.vstack((x1_tr, x1_tst))
        x2 = np.vstack((x2_tr, x2_tst))
        x3 = np.vstack((x3_tr, x3_tst))
        
        store_path = "Results/mediana_10fold_rbf/Both_5fold"+str(f)+"_2-12maldi_"+familia+"_prun"+str(hyper_parameters['sshiba']["pruning_crit"])+".pkl"
        message = "CODIGO TERMINADO EN SERVIDOR: " +"\n Data used: " + data_path +"\n Storage name: "+store_path

        results = {}
        
        c=0
        for i in range(5):

            myModel_mul = ksshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], fs=1)
            print(x0.shape)
            # maldi
            X0 = myModel_mul.struct_data(x0, method="reg", V=x0, kernel="rbf", sparse_fs=0)
            # fenot
            X1 = myModel_mul.struct_data(x1, method="mult")
            # genot
            X2 = myModel_mul.struct_data(x2, method="mult")
            # Hospital label
            X3 = myModel_mul.struct_data(x3, method="mult")
            # ab target
            Y0 = myModel_mul.struct_data(y0_tr.astype(float), method="mult")

            myModel_mul.fit(X0,
                            X1,
                            X2,
                            X3,
                            Y0,
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
