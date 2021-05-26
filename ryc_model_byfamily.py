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
familias = { "penicilinas": ['AMOXI/CLAV .1', 'PIP/TAZO.1'],
                    "cephalos": ['CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1'],
                    "monobactams": ['AZTREONAM.1'],
                    "carbapenems": ['IMIPENEM.1', 'MEROPENEM.1', 'ERTAPENEM.1'],
                    "aminos": ['GENTAMICINA.1', 'TOBRAMICINA.1', 'AMIKACINA.1', 'FOSFOMICINA.1'],
                    "fluoro":['CIPROFLOXACINO.1'],
                    "otros":['COLISTINA.1']
                    }

full = True

hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 1e-1, "max_it": int(1500)}}

# for familia in familias:
familia = "penicilinas"

for fold in range(5):
    for familia in familias:
        print(familia)
        data_path = "./data/ryc_data_mediansample_only2-12_TIC.pkl"
        folds_path = "./data/RYC_10STRATIFIEDfolds_muestrascompensadas_"+familia+".pkl"
        store_path = "Results/mediana_10fold_rbf/RyC_10fold"+str(fold)+"_muestrascompensadas_ARDgamma_2-12maldi_"+familia+"_prun"+str(hyper_parameters['sshiba']["pruning_crit"])+".pkl"
        message = "CODIGO TERMINADO EN SERVIDOR: " +"\n Data used: " + data_path + "\n Folds used: " + folds_path +\
                "\n Storage name: "+store_path

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
            f=fold
            print("Training fold: ", c)
            x0_tr, x0_val = maldi.loc[folds["train"][f]], maldi.loc[folds["val"][f]]
            x1_tr, x1_val = fen.loc[folds["train"][f]], fen.loc[folds["val"][f]]
            x2_tr, x2_val = gen.loc[folds["train"][f]], gen.loc[folds["val"][f]]
            y_tr, y_val = ab.loc[folds["train"][f]], ab.loc[folds["val"][f]]

            x0_tr = np.vstack(x0_tr.values).astype(float)
            x0_val = np.vstack(x0_val.values).astype(float)

            # Concatenate the X fold of seen points and the unseen points
            x0 = np.vstack((x0_tr, x0_val)).astype(float)
            # Fenotype
            x1 = np.vstack((np.vstack(x1_tr.values), np.vstack(x1_val.values))).astype(float)
            # Genotype
            x2 = np.vstack((np.vstack(x2_tr.values), np.vstack(x2_val.values))).astype(float)

            # Familias
            y0 = np.vstack((np.vstack(y_tr[familias[familia]].values)))

            myModel_mul = ksshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], fs=1)
            print(x0.shape)
            X0 = myModel_mul.struct_data(x0, method="reg", V=x0, kernel="rbf", sparse_fs=1)
            X1 = myModel_mul.struct_data(x1, method="mult")
            X2 = myModel_mul.struct_data(x2, method="mult")
            Y0 = myModel_mul.struct_data(y0.astype(float), method="mult")

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
