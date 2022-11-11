#%%
import pickle
import sys
import numpy as np
sys.path.insert(0, "./lib")
from lib import fast_fs_ksshiba_b_ord as ksshiba
import json
import telegram
from data import MaldiTofSpectrum
import topf
sys.path.append('../maldi_PIKE/maldi-learn/maldi_learn')

# Telegram bot: I have a telegram bot that sends me a message whenever the training has finished. Just dont pay attention to this if you dont want to use it.
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
# pruning crit to prune K dimensions
# max_it = maximum iterations to wait until convergence
hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 1e-2, "max_it": int(5000)}}
# KERNEL SELECTION: pike, linear or rbf
kernel = "pike" 

# Path to store the model. Then, we are going to use show_results.py to reload the model and see the results. In that way, you can launch as many options as your boss wants to.
store_path = "Results/normalizados/HGM_mkl2kernels_linpike_normalizados_prun"+str(hyper_parameters['sshiba']["pruning_crit"])+".pkl"

########################## LOAD DATA ######################
folds_path = "./data/GM_5STRATIFIEDfolds_paper.pkl"
data_path = "./data/gm_data_paper.pkl"
with open(data_path, 'rb') as pkl:
   gm_data = pickle.load(pkl)
with open(folds_path, 'rb') as pkl:
    folds = pickle.load(pkl)

# Msg for the telegram bot, remove this if you dont use it
message = "CÓDIGO TERMINADO EN SERVIDOR: " +"\n Data used: " + data_path + "\n Folds used: " + folds_path +\
              "\n Storage name: "+store_path

######### PREDICT ESBL AND CP
old_fen = gm_data['fen']
fen = old_fen[['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]
maldi = gm_data['maldi'].loc[fen.index]

##################### TRAIN MODEL AND PREDICT ######################
results = {}
sig = []
for f in range(len(folds["train"])):

    print("Training fold: ", f)
    # Select train and test data
    x0_tr, x0_val = maldi.loc[folds["train"][f]], maldi.loc[folds["val"][f]]
    x1_tr, x1_val = fen.loc[folds["train"][f]], fen.loc[folds["val"][f]]

    x0_tr= np.vstack(x0_tr.values).astype(float)
    x0_val = np.vstack(x0_val.values).astype(float)
    
    # Concatenate the X fold of train and test data
    x0 = np.vstack((x0_tr, x0_val)).astype(float)
    x0 = x0/np.mean(x0_tr)
    # AST
    x1 = np.vstack(x1_tr.values)

    # If you want to use the PIKE kernel, the data has to be transformed into their required shape:
    if kernel=="pike":
        x0_new = [[] for i in range(x0.shape[0])]
        for asig in range(x0.shape[0]):
            transformer= topf.PersistenceTransformer(n_peaks=200)
            print("Preproccessing TOPF "+str(asig)+"/"+str(x0.shape[0]), end="\r")
            topf_signal = np.concatenate((np.arange(2000,12000).reshape(-1, 1), x0[asig,:].reshape(-1,1)), axis=1)
            signal_transformed = transformer.fit_transform(topf_signal)
            x0_new[asig] = MaldiTofSpectrum(signal_transformed[signal_transformed[:,1]>0])
        x0 = [MaldiTofSpectrum(x0_new[i]) for i in range(len(x0_new))]

    # DECLARE KSSHIBA MODEL TO DECLARE EVERY DATA VIEW
    myModel_mul = ksshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], fs=1)

    ##################### DECLARE EACH ONE OF THE DATA VIEW USED IN THIS PROBLEM ##########################
    # MALDI MS view: can be linear kernel, rbf kernel or pike kernel. You can also no use kernel but it is 10K features.
    X0 = myModel_mul.struct_data(x0, method="reg", V=x0, kernel=kernel)
    # if you want to use no kernel: X0 = myModel_mul.struct_data(x0, method="reg")

    # Antibiotic Resistance mechanisms view: a multilabel view
    AST = myModel_mul.struct_data(x1, method="mult")


    ##################### TRAIN THE MODEL ###################### 
    myModel_mul.fit(X0,
                    AST,
                    max_iter=hyper_parameters['sshiba']['max_it'],
                    pruning_crit=hyper_parameters['sshiba']['pruning_crit'],
                    verbose=1,
                    feat_crit=1e-2)
    sig.append(myModel_mul.sig[0])
    model_name = "model_fold" + str(f)
    results[model_name] = myModel_mul

print(np.mean(sig))
# DUMP THE RESULTS INTO A pkl FILE
# with open(store_path, 'wb') as f:
#     pickle.dump(results, f)

# SEND A MESSAGE TO MY TELEGRAM BOT
notify_ending(message)