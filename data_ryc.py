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

familias = { "penicilinas": ['AMOXI/CLAV .1', 'PIP/TAZO.1'],
                    "cephalos": ['CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1'],
                    "monobactams": ['AZTREONAM.1'],
                    "carbapenems": ['IMIPENEM.1', 'MEROPENEM.1', 'ERTAPENEM.1'],
                    "aminos": ['GENTAMICINA.1', 'TOBRAMICINA.1', 'AMIKACINA.1', 'FOSFOMICINA.1'],
                    "fluoro":['CIPROFLOXACINO.1'],
                    "otros":['COLISTINA.1']
                    }

# for f in familias:
#     print("Familia: ", f)
#     for ab in familias[f]:
#         print(ryc_full_data[ab].value_counts())


# # # REMOVE unbalanced
# ryc_full_data = ryc_full_data.drop(['FOSFOMICINA.1', 'AMIKACINA.1', 'FOSFOMICINA', 'AMIKACINA'], axis=1)

# # complete_samples = np.unique(ryc_full_data[~ryc_full_data.iloc[:, np.arange(14, len(ryc_full_data.columns) - 5, 2)].isna(
# #             ).any(axis=1)].index)
# # missing_samples = np.unique(ryc_full_data[ryc_full_data.iloc[:, np.arange(14, len(ryc_full_data.columns) - 5, 2)].isna(
# #             ).any(axis=1)].index)

# # familias = {"penicilinas": [ 'PIP/TAZO.1'],
# #           "cephalos": ['CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1'],
# #           "monobactams": ['AZTREONAM.1'],
# #           "carbapenems": ['IMIPENEM.1', 'MEROPENEM.1', 'ERTAPENEM.1'],
# #           "aminos": ['GENTAMICINA.1', 'TOBRAMICINA.1'],
# #           "fluoro":['CIPROFLOXACINO.1'],
# #           "otros":['COLISTINA.1']}
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=0)

cols = ['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL', 'AMOXI/CLAV .1', 'PIP/TAZO.1', 'CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1', 'AZTREONAM.1', 'IMIPENEM.1', 'MEROPENEM.1', 'ERTAPENEM.1']
# Prueba loca: dejar solo blee y carbas
# cols = ['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL', 'AMOXI/CLAV ', 'PIP/TAZO', 'CEFTAZIDIMA', 'CEFOTAXIMA', 'CEFEPIME', 'AZTREONAM', 'IMIPENEM', 'MEROPENEM', 'ERTAPENEM']
# cols = ['GENTAMICINA', 'TOBRAMICINA', 'AMIKACINA', 'FOSFOMICINA', 'CIPROFLOXACINO', 'COLISTINA']
# gm_full_data = gm_full_data.drop(gm_full_data[gm_full_data['Fenotipo CP']==1].index)
# gm_full_data['Fenotipo CP+ESBL'][gm_full_data['Fenotipo CP']==1] = 1

complete_samples = np.unique(ryc_full_data[~ryc_full_data[cols].isna().any(axis=1)].index)
missing_samples = np.unique(ryc_full_data[ryc_full_data[cols].isna().any(axis=1)].index).tolist()

fold_storage_name = "data/RYC_10STRATIFIEDfolds_muestrascompensada_cpblee.pkl"
gm_folds = {"train": [], "val": []}

gm_complete_y = ryc_full_data[cols].loc[complete_samples]


for tr_idx, tst_idx in mskf.split(complete_samples, gm_complete_y):
    train_samples = complete_samples[tr_idx].tolist()
    # train_samples = [idx for idx in train_samples if idx != '675']
    # test_samples = complete_samples[tst_idx].tolist() + ['675']
    test_samples = complete_samples[tst_idx].tolist()
    for col in gm_complete_y.columns:
        # COMPROBAR SI ESTÁN DESCOMPENSADAS EN ENTRENAMIENTO
        low_appear_value = int(gm_complete_y[col].loc[train_samples].value_counts().index[1])
        difference = abs(np.diff(gm_complete_y[col].loc[train_samples].value_counts()))
        if int(difference) > int(40) and (gm_complete_y[col].loc[train_samples].value_counts().values<50).any():
            low_appear_rep_idx = gm_complete_y[col].loc[train_samples][gm_complete_y[col].loc[train_samples]==low_appear_value].sample(50, replace=1).index.tolist()
            train_samples = train_samples + low_appear_rep_idx
        # COMPROBAR SI TENGO MUESTRAS DE LOS DOS TIPOS EN TEST
        if gm_complete_y[col].loc[test_samples].value_counts().shape[0]<2:
            print("NO STRATIFIED FOR AB: "+col)
            print(gm_complete_y[col].loc[test_samples].value_counts())
            # Mirar cuale s el valor que no tenemos en test
            no_samples_value = int(np.logical_not(gm_complete_y[col].loc[test_samples].value_counts().index[0]))
            # Buscamos el valor entre los de training
            new_test_sample = gm_complete_y[col].loc[train_samples][gm_complete_y[col].loc[train_samples]==no_samples_value].sample(1).index.values.tolist()
            # Lo metemos en test y lo eliminamos de train
            test_samples = test_samples+new_test_sample
            train_samples = [idx for idx in train_samples if idx not in test_samples]
            print("Ahora debería estar correcto "+col)
            print(gm_complete_y[col].loc[test_samples].value_counts())  
    for idx in test_samples:
        print(idx in train_samples) 
    train_samples = train_samples + missing_samples
    gm_folds["train"].append(train_samples)
    gm_folds["val"].append(test_samples)

with open(fold_storage_name, 'wb') as f:
        pickle.dump(gm_folds, f)

print("hola")
print("a ver")
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


# del ryc_folds

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

with open("data/ryc_data_mediansample_only2-12_TIC.pkl", 'wb') as f:
    pickle.dump(ryc_dict, f)