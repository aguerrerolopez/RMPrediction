import pickle
import numpy as np
import pandas as pd
import os
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pyteomics import mzml

# PATH TO MZML FILES
ryc_path = "./data/Klebsiellas_RyC/"
# Columns to read from excel data
cols = ['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL', 'AMOXI/CLAV .1', 'PIP/TAZO.1', 'CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1', 'AZTREONAM.1', 'IMIPENEM.1', 'MEROPENEM.1', 'ERTAPENEM.1']
# BOOLEAN TO NORMALIZE BY TIC
tic_norm=True

######################## READ AND PROCESS DATA ############################
# LOAD RYC MALDI-TOF
listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk(ryc_path):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]
# ============= READ FEN/GEN/AB INFO ============
full_data = pd.read_excel("./data/DB_conjunta.xlsx", engine='openpyxl')

# READ DATA FROM FOLDS
data_int = []
id = []
letter = ["A", "B", "BTS", "C", "D", "E", "F", "G", "H"]
# CONVERT TO A PANDAS DATAFRAME
for file in listOfFiles:
    print(file)
    t = mzml.read(file)
    a = next(t)
    data_int.append(a["intensity array"][2000:12000])
    filename = file.split("/")[4]
    erase_end = filename.split(".")[0]
    if erase_end.split("_")[0] in letter:
        id.append(erase_end.split("_")[0] + erase_end.split("_")[1])
    else:
        id.append(erase_end.split("_")[0] + "-" + erase_end.split("_")[1])

ryc_data = pd.DataFrame(data=np.empty((len(data_int), 2)), columns=["Número de muestra", "maldi"])
ryc_data["maldi"] = data_int
ryc_data["Número de muestra"] = id

# AS EVERY BACTERIA HAS MORE THAN ONE MALDI MS WE SELECT THE MORE SIMILAR TO THE MEDIAN
ryc_median = pd.DataFrame(data=np.vstack(data_int))
ryc_median['id'] = id
ryc_median = ryc_median.set_index('id')
median_sample = ryc_median.groupby('id').median()

ryc_data_1s = pd.DataFrame(data=np.empty((len(median_sample), 2)), columns=["Número de muestra", "maldi"])
ryc_data_1s['Número de muestra'] = median_sample.index
ryc_data_1s = ryc_data_1s.set_index('Número de muestra')
data_median = []
for s in median_sample.index:
    print(s)
    if ryc_median.loc[s].shape[0] == 10000:
        data_median.append(ryc_median.loc[s].values)
    else:
        data_median.append(ryc_median.loc[s].iloc[(median_sample.loc[s]-ryc_median.loc[s]).mean(axis=1).abs().argmin(), :].values)
ryc_data_1s['maldi'] = data_median

# RELEASE MEMORY
del data_int, a, t, file, filename, id, filenames, letter, listOfFiles, erase_end, dirpath

# RYC FEN/GEN/AB
aux_ryc = full_data.loc[full_data['Centro'] == 'RyC'].copy().set_index("Número de muestra").drop('E11')
aux_ryc = aux_ryc.replace("R", np.float(1)).replace("I", "S").replace("S", np.float(0)).replace("-", np.nan)
ryc_full_data = pd.merge(how='outer', left=ryc_data_1s, right=aux_ryc, left_on='Número de muestra', right_on='Número de muestra')

################################ FOLDS CREATION ##########################
mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=0)
# GET RID OF MISSING SAMPLES FOR FOLD CREATION
complete_samples = np.unique(ryc_full_data[~ryc_full_data[cols].isna().any(axis=1)].index)
missing_samples = np.unique(ryc_full_data[ryc_full_data[cols].isna().any(axis=1)].index).tolist()
gm_complete_y = ryc_full_data[cols].loc[complete_samples]

# PATH TO SAVE FOLDS
fold_storage_name = "data/RYC_10STRATIFIEDfolds_muestrascompensada_cpblee.pkl"
gm_folds = {"train": [], "val": []}

for tr_idx, tst_idx in mskf.split(complete_samples, gm_complete_y):
    train_samples = complete_samples[tr_idx].tolist()
    test_samples = complete_samples[tst_idx].tolist()
    for col in gm_complete_y.columns:
        # HERE WE CHECK FOR UNBALANCE IN TRAINING
        low_appear_value = int(gm_complete_y[col].loc[train_samples].value_counts().index[1])
        difference = abs(np.diff(gm_complete_y[col].loc[train_samples].value_counts()))
        # IF THE UNBALANCE IS HIGHER THAN 40 WE OVERSAMPLE THE MINORITY CLASS
        if int(difference) > int(40) and (gm_complete_y[col].loc[train_samples].value_counts().values<50).any():
            low_appear_rep_idx = gm_complete_y[col].loc[train_samples][gm_complete_y[col].loc[train_samples]==low_appear_value].sample(50, replace=1).index.tolist()
            train_samples = train_samples + low_appear_rep_idx
        # CHECK IF I HAVE BOTH CLASSES IN TEST DATA, IF NOT WE MOVE A SAMPLE FROM TRAINING TO TEST THAT SATISFY THIS CONDITION
        if gm_complete_y[col].loc[test_samples].value_counts().shape[0]<2:
            print("NO STRATIFIED FOR AB: "+col)
            print(gm_complete_y[col].loc[test_samples].value_counts())
            # CHECK WHICH CLASS IS THE UNCOMPLETE
            no_samples_value = int(np.logical_not(gm_complete_y[col].loc[test_samples].value_counts().index[0]))
            # LOOK FOR A RANDOM SAMPLE OF THE UNCOMPLETE CLASS IN TRAIN
            new_test_sample = gm_complete_y[col].loc[train_samples][gm_complete_y[col].loc[train_samples]==no_samples_value].sample(1).index.values.tolist()
            # ADD TO TEST DATA AND REMOVE FROM TRAIN DATA
            test_samples = test_samples+new_test_sample
            train_samples = [idx for idx in train_samples if idx not in test_samples]
            print("Ahora debería estar correcto "+col)
            print(gm_complete_y[col].loc[test_samples].value_counts())  
    # CHECK THAT ANY TEST SAMPLES IS ALSO IN TRAIN
    for idx in test_samples:
        print(idx in train_samples) 
    train_samples = train_samples + missing_samples
    gm_folds["train"].append(train_samples)
    gm_folds["val"].append(test_samples)

# STORE FOLDS
with open(fold_storage_name, 'wb') as f:
        pickle.dump(gm_folds, f)

############################### NORMALIZE DATA AND STORE IT ##########################
if tic_norm:
    print("TIC NORMALIZING RYC DATA...")
    for i in range(ryc_full_data["maldi"].shape[0]):
        TIC = np.sum(ryc_full_data["maldi"][i])
        ryc_full_data["maldi"][i] /= TIC

else:
    print("NO TIC NORMALIZATION PERFORMED")

ryc_dict = {"full": ryc_full_data,
            "maldi": ryc_full_data['maldi'].copy(),
            "fen": ryc_full_data.loc[:, 'Fenotipo CP':'Fenotipo noCP noESBL'].copy(),
            "gen": ryc_full_data.loc[:, 'Genotipo CP':'Genotipo noCP noESBL'].copy(),
            "cmi": ryc_full_data.iloc[:, np.arange(13, len(ryc_full_data.columns) - 5, 2)].copy(),
            "binary_ab": ryc_full_data.iloc[:, np.arange(14, len(ryc_full_data.columns) - 5, 2)].copy()}

with open("data/ryc_data_mediansample_only2-12_TIC.pkl", 'wb') as f:
    pickle.dump(ryc_dict, f)