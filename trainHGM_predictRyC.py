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
familias = { "penicilinas": ['AMOXI/CLAV ', 'PIP/TAZO'],
                    "cephalos": ['CEFTAZIDIMA', 'CEFOTAXIMA', 'CEFEPIME'],
                    "monobactams": ['AZTREONAM'],
                    "carbapenems": ['IMIPENEM', 'MEROPENEM', 'ERTAPENEM'],
                    "aminos": ['GENTAMICINA', 'TOBRAMICINA', 'AMIKACINA', 'FOSFOMICINA'],
                    "fluoro":['CIPROFLOXACINO'],
                    "otros":['COLISTINA']
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

x0_tr, x0_tst  = np.vstack(hgm_data['maldi'].values), np.vstack(ryc_data['maldi'].values)
x1_tr, x1_tst  = np.vstack(hgm_data['fen'].values), np.vstack(ryc_data['fen'].values)
x2_tr, x2_tst  = np.nan*np.zeros(hgm_data['fen'].shape), np.vstack(ryc_data['gen'].values)

x0 = np.vstack((x0_tr, x0_tst))
x1 = np.vstack((x1_tr, x1_tst))
x2 = np.vstack((x2_tr, x2_tst))

for familia in familias:
    print(familia)
    
    store_path = "Results/mediana_noard_trainHGM_predictRyC/TrainHGM_PredictRyC_2-12maldi_"+familia+"_prun"+str(hyper_parameters['sshiba']["pruning_crit"])+".pkl"
    message = "CODIGO TERMINADO EN SERVIDOR: " +"\n Data used: " + data_path +"\n Storage name: "+store_path

    results = {}
    # Y0 de HGM
    y0_tr = np.vstack(hgm_data['binary_ab'][familias[familia]].values)
    c=0
    for i in range(5):

        myModel_mul = ksshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], fs=1)
        print(x0.shape)
        #maldi
        X0 = myModel_mul.struct_data(x0, method="reg", V=x0, kernel="linear", sparse_fs=0)
        #fenot
        X1 = myModel_mul.struct_data(x1, method="mult")
        #genot
        X2 = myModel_mul.struct_data(x2, method="mult")
        #ab
        Y0 = myModel_mul.struct_data(y0_tr.astype(float), method="mult")

        myModel_mul.fit(X0,
                        X1,
                        X2,
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
