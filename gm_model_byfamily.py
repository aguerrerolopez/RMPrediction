import pickle
import topf
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

data_path = "./data/hgm_data_mediansample_only2-12_TIC.pkl"
with open(data_path, 'rb') as pkl:
    hgm_data = pickle.load(pkl)

maldi = hgm_data['maldi']
fen = hgm_data['fen']
gen = hgm_data['gen']
cmi = hgm_data['cmi']
ab = hgm_data['binary_ab']

for asig in range(maldi.shape[0]):
    transformer= topf.PersistenceTransformer()
    print("Preproccessing TOPF "+str(asig)+"/"+str(maldi.shape[0]), end='\r')
    topf_signal = np.concatenate((np.arange(2000,12000).reshape(-1, 1), maldi[asig].reshape(-1,1)), axis=1)
    maldi[asig] = transformer.fit_transform(topf_signal)[:, 1]


hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 1e-1, "max_it": int(1500)}}

for fold in range(10):

    for familia in familias:
        print(familia)
        folds_path = "./data/HGM_10STRATIFIEDfolds_muestrascompensadas_"+familia+".pkl"
        store_path = "Results/topf_linear/HGM_10fold"+str(fold)+"_TOPF_muestrascompensadas_2-12maldi_"+familia+"_prun"+str(hyper_parameters['sshiba']["pruning_crit"])+".pkl"
        message = "CODIGO TERMINADO EN SERVIDOR: " +"\n Data used: " + data_path + "\n Folds used: " + folds_path +\
                "\n Storage name: "+store_path

        results = {}
    
        with open(folds_path, 'rb') as pkl:
            folds = pickle.load(pkl)

        c = 0
        for f in range(3):
            
            print("Training fold: ", c)
            x0_tr, x0_val = maldi.loc[folds["train"][fold]], maldi.loc[folds["val"][fold]]
            x1_tr, x1_val = fen.loc[folds["train"][fold]], fen.loc[folds["val"][fold]]
            # x2_tr, x2_val = gen.loc[folds["train"][fold]], gen.loc[folds["val"][fold]]
            y_tr, y_val = ab.loc[folds["train"][fold]], ab.loc[folds["val"][fold]]

            x0_tr = np.vstack(x0_tr.values).astype(float)
            x0_val = np.vstack(x0_val.values).astype(float)

            if len(np.unique(np.vstack((np.vstack(y_val[familias[familia]].values)))))<2:
                message = "ha petado porque no está stratified para la familia "+familias[familia]
                notify_ending(message)
                break

            # Concatenate the X fold of seen points and the unseen points
            x0 = np.vstack((x0_tr, x0_val)).astype(float)
            # Fenotype
            x1 = np.vstack((np.vstack(x1_tr.values), np.vstack(x1_val.values))).astype(float)
            # Genotype
            # x2 = np.vstack((np.vstack(x2_tr.values), np.vstack(x2_val.values))).astype(float)
            # Both together in one multilabel
            # x_fengen = np.hstack((x1, x2))
            # Familias
            y0 = np.vstack((np.vstack(y_tr[familias[familia]].values)))

            myModel_mul = ksshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], fs=1)
            print(x0.shape)
            X0 = myModel_mul.struct_data(x0, method="reg", V=x0, kernel="linear", sparse_fs=0)
            # X0 = myModel_mul.struct_data(x0, method="reg", sparse=1)
            X1 = myModel_mul.struct_data(x1, method="mult")
            # X2 = myModel_mul.struct_data(x2, method="mult")
            Y0 = myModel_mul.struct_data(y0.astype(float), method="mult")

            myModel_mul.fit(X0,
                            X1,
                            # X2,
                            Y0,
                            max_iter=hyper_parameters['sshiba']['max_it'],
                            pruning_crit=hyper_parameters['sshiba']['pruning_crit'],
                            verbose=1,
                            feat_crit=1e-2)

            model_name = "model_fold" + str(f)
            results[model_name] = myModel_mul
            c += 1

        with open(store_path, 'wb') as f:
            pickle.dump(results, f)

        notify_ending(message)
