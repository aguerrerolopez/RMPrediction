#%%
import pickle
import sys
import numpy as np
from sklearn.metrics.pairwise import kernel_metrics
sys.path.insert(0, "./lib")
from lib import fast_fs_ksshiba_b_ord as ksshiba
import json
import telegram
from data import MaldiTofSpectrum
import topf
sys.path.append('../maldi_PIKE/maldi-learn/maldi_learn')

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
hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 100, "max_it": int(5000)}}
# If 1: PIKE kernel from Weis et al 2020 is used. If 0: you have to choose between linear or rbf.
weis_et_al = 0
# linear_or_rbf= 1 means linear, 0 means rbf
linear_or_rbf = 1

########################## LOAD DATA ######################
folds_path = "./data/HGM_10STRATIFIEDfolds_muestrascompensada_pruebaloca.pkl"
data_path = "./data/hgm_data_mediansample_only2-12_TIC.pkl"
store_path = "Results/normalizados/HGM_linear_normalizados_prun"+str(hyper_parameters['sshiba']["pruning_crit"])+".pkl"
message = "CODIGO TERMINADO EN SERVIDOR: " +"\n Data used: " + data_path + "\n Folds used: " + folds_path +\
              "\n Storage name: "+store_path
with open(data_path, 'rb') as pkl:
   gm_data = pickle.load(pkl)
with open(folds_path, 'rb') as pkl:
    folds = pickle.load(pkl)

######### PREDICT CARB AND ESBL
old_fen = gm_data['fen']
old_fen = old_fen.drop(old_fen[old_fen['Fenotipo CP']==1].index)
fen = old_fen[['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]

maldi = gm_data['maldi'].loc[fen.index]
cmi = gm_data['cmi'].loc[fen.index]
ab = gm_data['binary_ab'].loc[fen.index]

############ PREDICT OTHER AB
# maldi = gm_data['maldi']
# cols = ['GENTAMICINA', 'TOBRAMICINA', 'AMIKACINA', 'FOSFOMICINA', 'CIPROFLOXACINO', 'COLISTINA']
# ab = gm_data['binary_ab'][cols]

##################### TRAIN BASELINES AND PREDICT ######################
results = {}
for f in range(len(folds["train"])):

    print("Training fold: ", f)
    x0_tr, x0_val = maldi.loc[folds["train"][f]], maldi.loc[folds["val"][f]]
    x1_tr, x1_val = fen.loc[folds["train"][f]], fen.loc[folds["val"][f]]
    y_tr, y_val = ab.loc[folds["train"][f]], ab.loc[folds["val"][f]]

    for idx in y_val.index:
        print(idx in y_tr.index)

    x0_tr= np.vstack(x0_tr.values).astype(float)
    x0_val = np.vstack(x0_val.values).astype(float)
    
    # Concatenate the X fold of seen points and the unseen points
    x0 = np.vstack((x0_tr, x0_val)).astype(float)
    x0 = x0/np.mean(x0_tr)
    # # Fenotype
    x1 = np.vstack(x1_tr.values)
    # Familias
    y0 = np.vstack(y_tr.values)

#%%
    if weis_et_al:
        x0_new = [[] for i in range(x0.shape[0])]
        for asig in range(x0.shape[0]):
            transformer= topf.PersistenceTransformer(n_peaks=400)
            print("Preproccessing TOPF "+str(asig)+"/"+str(x0.shape[0]), end="\r")
            topf_signal = np.concatenate((np.arange(2000,12000).reshape(-1, 1), x0[asig,:].reshape(-1,1)), axis=1)
            signal_transformed = transformer.fit_transform(topf_signal)
            x0_new[asig] = MaldiTofSpectrum(signal_transformed[signal_transformed[:,1]>0])
            x0 = [MaldiTofSpectrum(x0_new[i]) for i in range(len(x0_new))]

    # DECLARE KSSHIBA MODEL TO DECLARE EVERY DATA VIEW
    myModel_mul = ksshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], fs=1)

    ##################### DECLARE EACH ONE OF THE DATA VIEW USED IN THIS PROBLEM ##########################
    # MALDI MS view: can be linear kernel, rbf kernel or pike kernel. You can also no use kernel but it is 10K features.
    if linear_or_rbf:
        kernel = "linear"
    elif weis_et_al:
        kernel = "pike"
    else:
        kernel = "rbf"
    # method: reg because is a continuous input (aka regression), V is the support vectors of a kernel, here we are using all data
    X0 = myModel_mul.struct_data(x0, method="reg", V=x0, kernel=kernel)
    # if you want to use no kernel: X0 = myModel_mul.struct_data(x0, method="reg")

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
    myModel_mul.fit(X0,
                    RM1,
                    Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8,
                    max_iter=hyper_parameters['sshiba']['max_it'],
                    pruning_crit=hyper_parameters['sshiba']['pruning_crit'],
                    verbose=1,
                    feat_crit=1e-2)

    model_name = "model_fold" + str(f)
    results[model_name] = myModel_mul
    
# DUMP THE RESULTS INTO A FILE
with open(store_path, 'wb') as f:
    pickle.dump(results, f)
# SEND A MESSAGE TO MY TELEGRAM BOT
notify_ending(message)