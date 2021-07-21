import pickle
import numpy as np
import pandas as pd
import os
from pyteomics import mzml
from sklearn.model_selection import KFold, StratifiedKFold

tic_norm=1
ryc_path = "./data/Klebsiellas_RyC/"

# LOAD RYC MALDI-TOF
listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk(ryc_path):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]

data_int = []
id = []
letter = ["A", "B", "BTS", "C", "D", "E", "F", "G", "H"]
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

######### ESTO SE QUEDA CON LA MUESTRA MÁS CERCANA A LA MEDIANA
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
#########

######### ESTO SACA UNA MUESTRA ALEATORIA DE TODAS LAS POSIBLES
#ryc_data_1s = ryc_data.groupby('Número de muestra').sample(n=1, random_state=0).set_index('Número de muestra')
######### 
# RELEASE MEMORY
del data_int, a, t, file, filename, id, filenames, letter, listOfFiles, erase_end, dirpath

# ============= READ FEN/GEN/AB INFO ============
full_data = pd.read_excel("./data/DB_conjunta.xlsx", engine='openpyxl')

# RYC FEN/GEN/AB
print("ELIMINAMOS MUESTRA E11 DEL EXCEL PORQUE NO TENEMOS MALDI")
aux_ryc = full_data.loc[full_data['Centro'] == 'RyC'].copy().set_index("Número de muestra").drop('E11')
aux_ryc = aux_ryc.replace("R", np.float(1)).replace("I", "S").replace("S", np.float(0)).replace("-", np.nan)
ryc_full_data = pd.merge(how='outer', left=ryc_data_1s, right=aux_ryc, left_on='Número de muestra', right_on='Número de muestra')

familias = {"penicilinas": ['AMOXI/CLAV .1', 'PIP/TAZO.1'],
          "cephalos": ['CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1'],
          "monobactams": ['AZTREONAM.1'],
          "carbapenems": ['IMIPENEM.1', 'MEROPENEM.1', 'ERTAPENEM.1'],
          "aminos": ['GENTAMICINA.1', 'TOBRAMICINA.1'],
          "fluoro":['CIPROFLOXACINO.1'],
          "otros":['COLISTINA.1']}

for f in familias:
    print("Familia: ", f)
    for ab in familias[f]:
        print(ryc_full_data[ab].value_counts())


# # REMOVE unbalanced
ryc_full_data = ryc_full_data.drop(['FOSFOMICINA.1', 'AMIKACINA.1', 'FOSFOMICINA', 'AMIKACINA'], axis=1)

# complete_samples = np.unique(ryc_full_data[~ryc_full_data.iloc[:, np.arange(14, len(ryc_full_data.columns) - 5, 2)].isna(
#             ).any(axis=1)].index)
# missing_samples = np.unique(ryc_full_data[ryc_full_data.iloc[:, np.arange(14, len(ryc_full_data.columns) - 5, 2)].isna(
#             ).any(axis=1)].index)

# familias = {"penicilinas": [ 'PIP/TAZO.1'],
#           "cephalos": ['CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1'],
#           "monobactams": ['AZTREONAM.1'],
#           "carbapenems": ['IMIPENEM.1', 'MEROPENEM.1', 'ERTAPENEM.1'],
#           "aminos": ['GENTAMICINA.1', 'TOBRAMICINA.1'],
#           "fluoro":['CIPROFLOXACINO.1'],
#           "otros":['COLISTINA.1']}

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
for fam in familias:
    complete_samples = np.unique(ryc_full_data[~ryc_full_data[familias[fam]].isna().any(axis=1)].index)
    missing_samples = np.unique(ryc_full_data[ryc_full_data[familias[fam]].isna().any(axis=1)].index)
    fold_storage_name = "data/RyC_5STRATIFIEDfolds_noamika_nofosfo_"+fam+".pkl"
    ryc_complete_y= ryc_full_data[familias[fam]].loc[complete_samples]
    if ryc_complete_y.shape[1]>1:
        mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    else: 
        mskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    ryc_folds = {"train": [], "val": []}
    for tr_idx, tst_idx in mskf.split(complete_samples, ryc_complete_y):
        train_with_missing = np.concatenate([complete_samples[tr_idx], missing_samples])
        ryc_folds["train"].append(train_with_missing)
        ryc_folds["val"].append(complete_samples[tst_idx])
        for ab in familias[fam]:
            if ryc_complete_y[ab].loc[complete_samples[tst_idx]].value_counts().shape[0]<2:
                print("NO STRATIFIED FOR AB: "+ab)
                print(ryc_complete_y[ab].loc[complete_samples[tst_idx]].value_counts())
    with open(fold_storage_name, 'wb') as f:
        pickle.dump(ryc_folds, f)


# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# ryc_folds = {"train": [], "val": []}
# ryc_y_complete = ryc_full_data.loc[complete_samples].iloc[:, np.arange(14, len(ryc_full_data.columns) - 5, 2)]
# for tr_idx, tst_idx in mskf.split(complete_samples, ryc_y_complete):
#     train_with_missing = np.concatenate([complete_samples[tr_idx], missing_samples])
#     ryc_folds["train"].append(train_with_missing)
#     ryc_folds["val"].append(complete_samples[tst_idx])

# from skmultilearn.model_selection import IterativeStratification
# ryc_y_complete = ryc_full_data.loc[complete_samples].iloc[:, np.arange(14, len(ryc_full_data.columns) - 5, 2)]
# kf = IterativeStratification(n_splits=10, order=1)
# ryc_folds = {"train": [], "val": []}
# for train_idx, test_idx in kf.split(complete_samples, ryc_y_complete):
#     train_with_missing = np.concatenate([complete_samples[train_idx], missing_samples])
#     ryc_folds["train"].append(train_with_missing)
#     ryc_folds["val"].append(complete_samples[test_idx])


# # for train_idx, test_idx in kf.split(range(len(complete_samples))):
# #     train_with_missing = np.concatenate([complete_samples[train_idx], missing_samples])
# #     ryc_folds["train"].append(train_with_missing)
# #     ryc_folds["val"].append(complete_samples[test_idx])


with open("data/ryc_5STRATIFIEDfolds_noamika_nofosfo_fullAB.pkl", 'wb') as f:
    pickle.dump(ryc_folds, f)

del ryc_folds

if tic_norm:
    print("TIC NORMALIZING RYC DATA...")
    for i in range(ryc_full_data["maldi"].shape[0]):
        TIC = np.sum(ryc_full_data["maldi"][i])
        ryc_full_data["maldi"][i] /= TIC

else:
    print("NO TIC NORMALIZATION PERFORMED")

# print("Standarizing RYC data alone...")
# ryc_data = np.vstack(ryc_full_data["maldi"].values)
# scaler = StandardScaler()
# scaler.fit(ryc_data)
# for i in range(ryc_full_data["maldi"].shape[0]):
#     ryc_full_data["maldi"][i] = scaler.transform(ryc_data[i, :][np.newaxis, :])[0, :]

ryc_dict = {"full": ryc_full_data,
            "maldi": ryc_full_data['maldi'].copy(),
            "fen": ryc_full_data.loc[:, 'Fenotipo CP':'Fenotipo noCP noESBL'].copy(),
            "gen": ryc_full_data.loc[:, 'Genotipo CP':'Genotipo noCP noESBL'].copy(),
            "cmi": ryc_full_data.iloc[:, np.arange(13, len(ryc_full_data.columns) - 5, 2)].copy(),
            "binary_ab": ryc_full_data.iloc[:, np.arange(14, len(ryc_full_data.columns) - 5, 2)].copy()}

with open("data/ryc_data_mediansample_only2-12_noamika_nofosfo_TIC.pkl", 'wb') as f:
    pickle.dump(ryc_dict, f)