import pickle
import numpy as np
import pandas as pd
import os
from pyteomics import mzml
from sklearn.model_selection import KFold


def load_gm(excel_path = "/Users/alexjorguer/Downloads/Reproducibilidad/Klebsiellas_Estudio_Reproducibilidad_rev.xlsx",
            hgm_rep_mzml_path = '/Users/alexjorguer/Downloads/Reproducibilidad/mzml', drop=False, tic_norm=True):
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
        maldis.append(a["intensity array"][0:20000])
        id_samples_rep.append(file.split('_')[2])

    gm_data = pd.DataFrame(data=np.empty((len(maldis), 2)), columns=["Nº Espectro", "maldi"])
    gm_data["maldi"] = maldis
    gm_data["Nº Espectro"] = id_samples_rep
    gm_x = gm_data.set_index("Nº Espectro")

    unique_samples = np.unique(id_samples_rep)

    excel_data = pd.read_excel(excel_path, engine='openpyxl', dtype={'Nº Espectro': str})
    excel_data = excel_data.replace("R", 1).replace("BLEE", 1).replace("I", "S").replace("S", 0).replace("-", np.nan)
    excel_samples = np.unique(excel_data['Nº Espectro'])

    gm_full_data = pd.merge(how='outer', left=gm_data, right=excel_data, left_on='Nº Espectro',
                            right_on='Nº Espectro').set_index("Nº Espectro")

    if drop:
        gm_full_data = gm_full_data.drop(['MEROPENEM', 'MEROPENEM.1', 'COLISTINA', 'COLISTINA.1'], axis=1)

    complete_samples = np.unique(
        gm_full_data[~gm_full_data.iloc[:, np.arange(8, len(gm_full_data.columns) - 2, 2)].isna(
        ).any(axis=1)].index)
    missing_samples = np.unique(gm_full_data[gm_full_data.iloc[:, np.arange(8, len(gm_full_data.columns) - 2, 2)].isna(
    ).any(axis=1)].index)

    kf = KFold(n_splits=5, random_state=32, shuffle=True)
    gm_folds = {"train": [], "val": []}

    for train_idx, test_idx in kf.split(range(len(complete_samples))):
        train_with_missing = np.concatenate([complete_samples[train_idx], missing_samples])
        gm_folds["train"].append(train_with_missing)
        gm_folds["val"].append(complete_samples[test_idx])

    with open("data/gm_5folds_sinreplicados.pkl", 'wb') as f:
        pickle.dump(gm_folds, f)

    if tic_norm:
        print("TIC NORMALIZING gm DATA...")
        for i in range(gm_full_data["maldi"].shape[0]):
            TIC = np.sum(gm_full_data["maldi"][i])
            gm_full_data["maldi"][i] /= TIC

    gm_dict = {"full": gm_full_data,
               "maldi": gm_full_data['maldi'].copy(),
               "fen": gm_full_data.loc[:, 'Fenotipo CP':'Fenotipo noCP noESBL'].copy(),
               "gen": None,
               "cmi": gm_full_data.iloc[:, np.arange(9, len(gm_full_data.columns) - 1, 2)].copy(),
               "binary_ab": gm_full_data.iloc[:, np.arange(8, len(gm_full_data.columns) - 1, 2)].copy()}

    with open("data/gm_data_sinreplicados_TIC.pkl", 'wb') as f:
        pickle.dump(gm_dict, f)

def load_ryc( ryc_path='./data/Klebsiellas_RyC/', tic_norm=True):

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
        data_int.append(a["intensity array"][0:20000])
        filename = file.split("/")[4]
        erase_end = filename.split(".")[0]
        if erase_end.split("_")[0] in letter:
            id.append(erase_end.split("_")[0] + erase_end.split("_")[1])
        else:
            id.append(erase_end.split("_")[0] + "-" + erase_end.split("_")[1])

    ryc_data = pd.DataFrame(data=np.empty((len(data_int), 2)), columns=["Número de muestra", "maldi"])
    ryc_data["maldi"] = data_int
    ryc_data["Número de muestra"] = id

    # RELEASE MEMORY
    del data_int, a, t, file, filename, id, filenames, letter, listOfFiles, erase_end, dirpath

    # ============= READ FEN/GEN/AB INFO ============
    full_data = pd.read_excel("./data/DB_conjunta.xlsx", engine='openpyxl')

    # RYC FEN/GEN/AB
    print("ELIMINAMOS MUESTRA E11 DEL EXCEL PORQUE NO TENEMOS MALDI")
    aux_ryc = full_data.loc[full_data['Centro'] == 'RyC'].copy().set_index("Número de muestra").drop('E11')
    aux_ryc = aux_ryc.replace("R", 1).replace("I", "S").replace("S", 0).replace("-", np.nan)
    ryc_full_data = pd.merge(how='outer', left=ryc_data, right=aux_ryc, left_on='Número de muestra', right_on='Número de muestra').set_index("Número de muestra")

    # REMOVE ERTAPENEM BECAUSE IS HALF MISSING
    ryc_full_data = ryc_full_data.drop(['ERTAPENEM', 'ERTAPENEM.1'], axis=1)

    complete_samples = np.unique(ryc_full_data[~ryc_full_data.iloc[:, np.arange(14, len(ryc_full_data.columns) - 5, 2)].isna(
                ).any(axis=1)].index)
    missing_samples = np.unique(ryc_full_data[ryc_full_data.iloc[:, np.arange(14, len(ryc_full_data.columns) - 5, 2)].isna(
                ).any(axis=1)].index)

    kf = KFold(n_splits=5, random_state=32, shuffle=True)
    ryc_folds = {"train": [], "val": []}

    for train_idx, test_idx in kf.split(range(len(complete_samples))):
        train_with_missing = np.concatenate([complete_samples[train_idx], missing_samples])
        ryc_folds["train"].append(train_with_missing)
        ryc_folds["val"].append(complete_samples[test_idx])
    with open("data/ryc_5folds_NOERTAPENEM.pkl", 'wb') as f:
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

    with open("data/ryc_data_TIC.pkl", 'wb') as f:
        pickle.dump(ryc_dict, f)


