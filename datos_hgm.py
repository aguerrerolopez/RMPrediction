import pickle
import numpy as np
import pandas as pd
import os
from pyteomics import mzml
from sklearn.model_selection import KFold, StratifiedKFold

hgm_rep_mzml_path = './data/Reproducibilidad/mzml'
excel_path = "./data/Reproducibilidad/Klebsiellas_Estudio_Reproducibilidad_rev.xlsx"

tic_norm=True

listOfFiles = list()
for (dirpath, dirnames, filenames_rep) in os.walk(hgm_rep_mzml_path):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames_rep]

id_samples_rep = []
maldis = []
for filepath in listOfFiles:
    file = filepath.split("/")[-1]
    if file == ".DS_Store" or file.split('_')[2] == '1988':
        continue
    print(file)
    t = mzml.read(filepath)
    a = next(t)
    maldis.append(a["intensity array"][2000:12000])
    id_samples_rep.append(file.split('_')[2])

gm_data = pd.DataFrame(data=np.empty((len(maldis), 2)), columns=["Nº Espectro", "maldi"])
gm_data["maldi"] = maldis
gm_data["Nº Espectro"] = id_samples_rep
gm_x = gm_data.set_index("Nº Espectro")

# gm_random_1s = gm_data.groupby('Nº Espectro').sample(n=1, random_state=0).set_index('Nº Espectro')
hgm_median = pd.DataFrame(data=np.vstack(maldis))
hgm_median['id'] = id_samples_rep
hgm_median = hgm_median.set_index('id')
median_sample = hgm_median.groupby('id').median()

gm_median_1s = pd.DataFrame(data=np.empty((len(median_sample), 2)), columns=["Nº Espectro", "maldi"])
gm_median_1s['Nº Espectro'] = median_sample.index
gm_random_1s = gm_median_1s.set_index('Nº Espectro')
data_median = []
for s in median_sample.index:
    print(s)
    if hgm_median.loc[s].shape[0] == 10000:
        data_median.append(hgm_median.loc[s].values)
    else:
        data_median.append(hgm_median.loc[s].iloc[(median_sample.loc[s]-hgm_median.loc[s]).mean(axis=1).abs().argmin(), :].values)
gm_median_1s['maldi'] = data_median

excel_data = pd.read_excel(excel_path, engine='openpyxl', dtype={'Nº Espectro': str})
excel_data = excel_data.replace("R", np.float(1)).replace("BLEE", np.float(1)).replace("I", "S").replace("S", np.float(0)).replace("-", np.nan)
excel_samples = np.unique(excel_data['Nº Espectro'])

gm_full_data = pd.merge(how='outer', left=gm_median_1s, right=excel_data, left_on='Nº Espectro',
                        right_on='Nº Espectro').set_index("Nº Espectro")

# gm_full_data = gm_full_data.drop(['MEROPENEM', 'MEROPENEM.1', 'COLISTINA', 'COLISTINA.1', 'IMIPENEM', 'IMIPENEM.1'], axis=1)

# complete_samples = np.unique(
#     gm_full_data[~gm_full_data.iloc[:, np.arange(8, len(gm_full_data.columns) - 2, 2)].isna(
#     ).any(axis=1)].index)
# missing_samples = np.unique(gm_full_data[gm_full_data.iloc[:, np.arange(8, len(gm_full_data.columns) - 2, 2)].isna(
# ).any(axis=1)].index)

familias = {
                    "penicilinas": ['AMOXI/CLAV ', 'PIP/TAZO'],
                    "cephalos": ['CEFTAZIDIMA', 'CEFOTAXIMA', 'CEFEPIME'],
                    "monobactams": ['AZTREONAM'],
                    "carbapenems": ['IMIPENEM', 'MEROPENEM', 'ERTAPENEM'],
                    "aminos": ['GENTAMICINA', 'TOBRAMICINA', 'AMIKACINA', 'FOSFOMICINA'],
                    "fluoro":['CIPROFLOXACINO'],
                    "otros":['COLISTINA']
                    }

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
for fam in familias:
    complete_samples = np.unique(gm_full_data[~gm_full_data[familias[fam]].isna().any(axis=1)].index)
    missing_samples = np.unique(gm_full_data[gm_full_data[familias[fam]].isna().any(axis=1)].index)
    fold_storage_name = "data/HGM_10STRATIFIEDfolds_muestrascompensadas_"+fam+".pkl"
    gm_complete_y= gm_full_data[familias[fam]].loc[complete_samples]
    if gm_complete_y.shape[1]>1:
        mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    else: 
        mskf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    gm_folds = {"train": [], "val": []}
    for tr_idx, tst_idx in mskf.split(complete_samples, gm_complete_y):
        train_samples = complete_samples[tr_idx]
        test_samples = complete_samples[tst_idx]
        for ab in familias[fam]:
            # COMPROBAR SI ESTÁN DESCOMPENSADAS EN ENTRENAMIENTO
            low_appear_value = int(gm_complete_y[ab].loc[train_samples].value_counts().index[1])
            n_repeats = abs(np.diff(gm_complete_y[ab].loc[train_samples].value_counts()))
            if int(n_repeats) > int(50):
                low_appear_rep_idx = gm_complete_y[ab].loc[train_samples][gm_complete_y[ab].loc[train_samples]==low_appear_value].sample(n_repeats, replace=1).index
                train_samples = np.concatenate([train_samples, low_appear_rep_idx])
            
            # COMPROBAR SI TENGO MUESTRAS DE LOS DOS TIPOS EN TEST
            if gm_complete_y[ab].loc[test_samples].value_counts().shape[0]<2:
                print("NO STRATIFIED FOR AB: "+ab)
                print(gm_complete_y[ab].loc[test_samples].value_counts())
                # Mirar cuale s el valor que no tenemos en test
                no_samples_value = int(np.logical_not(gm_complete_y[ab].loc[test_samples].value_counts().index[0]))
                # Buscamos el valor entre los de training
                new_test_sample = gm_complete_y[ab].loc[train_samples][gm_complete_y[ab].loc[train_samples]==no_samples_value].sample(1).index
                # Lo metemos en test y lo eliminamos de train
                test_samples = np.concatenate([test_samples, new_test_sample])
                train_samples = train_samples[train_samples != new_test_sample[0]]
                print("Ahora debería estar correcto "+ab)
                print(gm_complete_y[ab].loc[test_samples].value_counts())

        train_samples = np.concatenate([train_samples, missing_samples])   
        gm_folds["train"].append(train_samples)
        gm_folds["val"].append(test_samples)
    
    with open(fold_storage_name, 'wb') as f:
        pickle.dump(gm_folds, f)

# gm_folds = {"train": [], "val": []}
# gm_y_complete = gm_full_data.loc[complete_samples].iloc[:, np.arange(8, len(gm_full_data.columns) - 2, 2)]
# for tr_idx, tst_idx in mskf.split(complete_samples, gm_y_complete):
#     train_with_missing = np.concatenate([complete_samples[tr_idx], missing_samples])
#     gm_folds["train"].append(train_with_missing)
#     gm_folds["val"].append(complete_samples[tst_idx])

# kf = KFold(n_splits=5, random_state=32, shuffle=True)
# gm_folds = {"train": [], "val": []}

# for train_idx, test_idx in kf.split(range(len(complete_samples))):
#     train_with_missing = np.concatenate([complete_samples[train_idx], missing_samples])
#     gm_folds["train"].append(train_with_missing)
#     gm_folds["val"].append(complete_samples[test_idx])

# with open("data/gm_5folds_sinreplicados.pkl", 'wb') as f:
#     pickle.dump(gm_folds, f)

# if tic_norm:
#     print("TIC NORMALIZING gm DATA...")
#     for i in range(gm_full_data["maldi"].shape[0]):
#         TIC = np.sum(gm_full_data["maldi"][i])
#         gm_full_data["maldi"][i] /= TIC

# gm_dict = {"full": gm_full_data,
#             "maldi": gm_full_data['maldi'].copy(),
#             "fen": gm_full_data.loc[:, 'Fenotipo CP':'Fenotipo noCP noESBL'].copy(),
#             "gen": None,
#             "cmi": gm_full_data.iloc[:, np.arange(9, len(gm_full_data.columns) - 1, 2)].copy(),
#             "binary_ab": gm_full_data.iloc[:, np.arange(8, len(gm_full_data.columns) - 1, 2)].copy()}

# with open("data/hgm_data_mediansample_only2-12_TIC.pkl", 'wb') as f:
#     pickle.dump(gm_dict, f)
