import pickle
import topf
import sys
sys.path.append('../maldi_PIKE/maldi-learn/maldi_learn')
from data import MaldiTofSpectrum
import numpy as np
sys.path.insert(0, "./lib")
from lib import fast_fs_ksshiba_b_ord as ksshiba
import json
import telegram
import pandas as pd

# Telegram bot
def notify_ending(message):
    with open('./keys_file.json', 'r') as keys_file:
        k = json.load(keys_file)
        token = k['telegram_token']
        chat_id = k['telegram_chat_id']
    bot = telegram.Bot(token=token)
    bot.sendMessage(chat_id=chat_id, text=message)

##################### PARAMETERS SELECTION ####################
# Prune = to prune the latent space
# myKc = initial K dimension of latent space
# pruning crit to prune K dimensiosn
# max_it = maximum iterations to wait until convergence
hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 1e-1, "max_it": int(1500)}}
# If 1: PIKE kernel from Weis et al 2020 is used. If 0: you have to choose between linear or rbf.
weis_et_al = 0
# linear_or_rbf= 1 means linear, 0 means rbf
linear_or_rbf = 1
# PATH TO STORE RESULTS
store_path = "./Results/TrainHGM_predictRYC/TrainHGM_PredRyC_linear_gen4_prun"+str(hyper_parameters['sshiba']["pruning_crit"])+".pkl"
# MESSAGE TO MY TELEGRAM BOT
message = "CODIGO TERMINADO EN SERVIDOR: " +"\n Data used: " + data_path +"\n Storage name: "+store_path

########################## LOAD DATA ######################
data_path = "./data/hgm_data_mediansample_only2-12_TIC.pkl"
with open(data_path, 'rb') as pkl:
    hgm_data = pickle.load(pkl)
data_path = "./data/ryc_data_mediansample_only2-12_TIC.pkl"
with open(data_path, 'rb') as pkl:
    ryc_data = pickle.load(pkl)
folds_path = "./data/HGM_10STRATIFIEDfolds_muestrascompensada_pruebaloca.pkl"
with open(folds_path, 'rb') as pkl:
    folds = pickle.load(pkl)


# COLUMNS TO USE FROM HGM
ab_cols =  ['AMOXI/CLAV ', 'PIP/TAZO', 'CEFTAZIDIMA', 'CEFOTAXIMA', 'CEFEPIME', 'AZTREONAM', 'IMIPENEM', 'MEROPENEM', 'ERTAPENEM']

old_fen = hgm_data['fen']
old_fen = old_fen.drop(old_fen[old_fen['Fenotipo CP']==1].index)
hgm_data['fen'] = old_fen[['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]

hgm_data['maldi'] = hgm_data['maldi'].loc[hgm_data['fen'].index]
hgm_data['cmi'] = hgm_data['cmi'].loc[hgm_data['fen'].index]
hgm_data['binary_ab'] = hgm_data['binary_ab'][ab_cols].loc[hgm_data['fen'].index]

old_fen = ryc_data['fen']
old_fen = old_fen.drop(old_fen[old_fen['Fenotipo CP']==1].index)
ryc_data['fen'] = old_fen[['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]

# COLUMNS TO USE FROM HRC
ab_cols =  ['AMOXI/CLAV .1', 'PIP/TAZO.1', 'CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1', 'AZTREONAM.1', 'IMIPENEM.1', 'MEROPENEM.1', 'ERTAPENEM.1']

full_predict = pd.concat([ryc_data['fen'], ryc_data['binary_ab'][ab_cols].loc[ryc_data['fen'].index]], axis=1)
# REMOVE MISSING DATA FOR HRC DATA
full_predict = full_predict.dropna()

ryc_data['maldi'] = ryc_data['maldi'].loc[full_predict.index]
ryc_data['binary_ab'] = full_predict[ab_cols]
ryc_data['fen'] = full_predict[['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]
ryc_data['gen'] = ryc_data['gen'].loc[full_predict.index]
# REMOVE MISSING DATA FOR HGM DATA
nomissing = hgm_data['binary_ab'].loc[folds["train"][0]].dropna().index

############### CONSTRUCT TRAIN AND TEST DATA
# MALDI MS DATA FROM HGM
hgm_x0_tr, hgm_x0_tst = np.vstack(hgm_data['maldi'].loc[nomissing].values), np.vstack(hgm_data['maldi'].loc[folds["val"][0]].values)
# RESISTANCE MECHANISM DATA FROM HGM
hgm_x1_tr, hgm_x1_tst = hgm_data['fen'].loc[nomissing], hgm_data['fen'].loc[folds["val"][0]]
# AMR DATA FROM HGM
hgm_y_tr, hgm_y_tst = hgm_data['binary_ab'].loc[nomissing], hgm_data['binary_ab'].loc[folds["val"][0]]




x0 = np.vstack((hgm_x0_tr, hgm_x0_tst, np.vstack(ryc_data['maldi'].values)))
x1 = np.vstack((hgm_x1_tr.values, hgm_x1_tst.values))
# HERE WE ALSO ADD THE GEN DATA THAT ONLY EXISTS FOR HRC DATA
x2 = np.vstack((np.nan*np.zeros((x1.shape[0], 4)), ryc_data['gen'].values))
y0 = np.vstack((hgm_y_tr.values, hgm_y_tst.values))

# CONSTRUCT THE WEIS ET AL DATA
if weis_et_al:
    x0_new = [[] for i in range(x0.shape[0])]
    for asig in range(x0.shape[0]):
        transformer= topf.PersistenceTransformer(n_peaks=200)
        print("Preproccessing TOPF "+str(asig)+"/"+str(x0.shape[0]), end="\r")
        topf_signal = np.concatenate((np.arange(2000,12000).reshape(-1, 1), x0[asig,:].reshape(-1,1)), axis=1)
        signal_transformed = transformer.fit_transform(topf_signal)
        x0_new[asig] = MaldiTofSpectrum(signal_transformed[signal_transformed[:,1]>0])

    x0 = [MaldiTofSpectrum(x0_new[i]) for i in range(len(x0_new))]



################### DECLARE THE MODEL INPUT VIEWS ###########################################################33
myModel_mul = ksshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], fs=1)

# MALDI MS view: can be linear kernel, rbf kernel or pike kernel. You can also no use kernel but it is 10K features.
if linear_or_rbf:
    kernel = "linear"
elif weis_et_al:
    kernel = "pike"
else:
    kernel = "rbf"
X0 = myModel_mul.struct_data(x0, method="reg", V=x0, kernel=KERNEL, sparse_fs=0)

# GENETIC VIEW
GEN0 = myModel_mul.struct_data(x2, method="mult")
# Resistance mechanisms view: a multilabel view
RM1 = myModel_mul.struct_data(x1, method="mult")
# AMR view: a multilabel view one per AB
Y0 = myModel_mul.struct_data(y0[:, 0], method="mult")
Y1 = myModel_mul.struct_data(y0[:, 1], method="mult")
Y2 = myModel_mul.struct_data(y0[:, 2], method="mult")
Y3 = myModel_mul.struct_data(y0[:, 3], method="mult")
Y4 = myModel_mul.struct_data(y0[:, 4], method="mult")
Y5 = myModel_mul.struct_data(y0[:, 5], method="mult")
Y6 = myModel_mul.struct_data(y0[:, 6], method="mult")
Y7 = myModel_mul.struct_data(y0[:, 7], method="mult")
Y8 = myModel_mul.struct_data(y0[:, 8], method="mult")

##################### TRAIN THE MODEL ###################### 
# THE OTHER OF THE VIEWS DO NOT REALLY MATTER IF YOU REMEMBER THE ORDER ONCE YOU WANT TO SEE THE RESULTS
myModel_mul.fit(X0, GEN0,
                RM1,
                Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8,
                max_iter=hyper_parameters['sshiba']['max_it'],
                pruning_crit=hyper_parameters['sshiba']['pruning_crit'],
                verbose=1,
                feat_crit=1e-2)

model_name = "model_fold0"
results = {}
results[model_name] = myModel_mul
# DUMP THE RESULTS INTO A FILE
with open(store_path, 'wb') as f:
    pickle.dump(results, f)
# SEND A MESSAGE TO MY TELEGRAM BOT
notify_ending(message)