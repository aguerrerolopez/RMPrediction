import pickle
import numpy as np
from pyteomics import mzml
import pandas as pd
import os
import preprocess as pp
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler



def process_hospital_data(gm_path = "./data/old_data/", gm_n_labels = 18, ryc_path = 'data/Klebsiellas_RyC/',
                          keep_common = True, return_GM = True, return_RYC = True , norm_both = True, save_data=True):
    """Method to process both hospital data.
    :param gm_path: String path. Path to Gregorio M. hospital data.
    :param ryc_path: String path. Path to Ramon y Cajal hospital data
    :param gm_n_labels: int. Number of antibiotic targets that exists in GM data.
    :param keep_common: bool. If True only antibiotic that exists in BOTH hospital is kept.
    :param return_GM: bool. If True the GM processed data is returned.
    :param return_RYC: bool. If True the RYC processed data is returned.
    :param norm_both: bool. If True both hospitals are normalized together. If False, each hospital data is normalized
    with its own data.
    :param save_data: bool. If True, hospital data is stored.
    :return: tuple. List of two ndarray in the first dimension (GM standarized data) and a Dataframe in the second
    dimension with the RYC standarized data."""

    # ============= GREGORIO MARANYON DATA ============
    # Load GM data to choose the correct labels from RyC
    gm_data = pp.load_data(path=gm_path, format="zip")
    gm_labels = gm_data.columns[-(gm_n_labels + 1):-1]

    # Y dataframe with ALL GM antibiotics
    gm_y = gm_data[gm_labels].drop(columns=["AMPICILINA"])
    gm_labels = gm_y.columns


    gm_x = np.zeros((gm_data["intensity"].shape[0], 20000))
    aux = 0
    for sample in gm_data["intensity"].values:
        gm_x[aux, :] = sample[0:20000]
        aux += 1
    del aux, sample, gm_data

    # =================  RYC DATA  ===================
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
    ryc_x = ryc_data.set_index("Número de muestra")

    samples_in_X = np.unique(id).astype(object)
    # RELEASE MEMORY
    del data_int, a, t, file, filename, id, filenames, letter, listOfFiles, erase_end, dirpath


    # Read Y from Excel
    ryc_y_raw = pd.read_excel("data/Sensibilidad_Klebsiellas_RyC_revisado.xlsx", sheet_name="RyC",
                              engine='openpyxl')

    # Keep only the common labels to both hospitals
    if keep_common:
        # GET ONLY THE COMMON LABELS FROM BOTH HOSPITALS
        common_labels = ['Número de muestra']
        common_for_gm = []
        for label in gm_labels:
            aux = label + ".1"
            if aux in ryc_y_raw.columns:
                print(label + " exists in both hospitals")
                common_labels.append(aux)
                common_for_gm.append(label)
        del aux
        gm_y = gm_y[common_for_gm].to_numpy()
        ryc_y_com = ryc_y_raw[common_labels].set_index("Número de muestra")
    else:
        ryc_y_com = ryc_y_raw.set_index("Número de muestra")

    # Erase errors on data if they don't exist in X and Y simultaneously
    non_exists_in_X = []
    for sample in samples_in_X:
        if sample in ryc_y_com.index.values:
            continue
        else:
            non_exists_in_X.append(sample)

    non_exists_in_Y = []
    for sample in ryc_y_com.index.values:
        if sample in samples_in_X:
            continue
        else:
            non_exists_in_Y.append(sample)

    ryc_y_com = ryc_y_com.drop(non_exists_in_Y)
    ryc_x = ryc_x.drop(non_exists_in_X)
    ryc_full_data = pd.merge(left=ryc_x, right=ryc_y_com, left_on='Número de muestra', right_on='Número de muestra')

    with open("data/ryc_data.pkl", 'wb') as f:
        pickle.dump(ryc_full_data, f)
    # Create folds
    # if create_gm_folds:
    #     n_samples = range(0, len(gm_y))
    #     kf = KFold(n_splits=10, random_state=32, shuffle=True)
    #     gm_folds = {"train": [], "test": []}
    #     for train_idx, test_idx in kf.split(range(len(n_samples))):
    #         gm_folds["train"].append(train_idx)
    #         gm_folds["test"].append(test_idx)
    #     with open("data/gm_folds.pkl", 'wb') as f:
    #         pickle.dump(gm_folds, f)
    #
    #     del gm_folds

    # if create_ryc_folds:
    #     n_samples = np.unique(ryc_full_data.index.values)
    #     kf = KFold(n_splits=10, random_state=32, shuffle=True)
    #     ryc_folds = {"train": [], "test": []}
    #
    #     for train_idx, test_idx in kf.split(range(len(n_samples))):
    #         ryc_folds["train"].append(n_samples[train_idx])
    #         ryc_folds["test"].append(n_samples[test_idx])
    #     with open("data/ryc_folds.pkl", 'wb') as f:
    #         pickle.dump(ryc_folds, f)
    #
    #     del ryc_folds


    print("TIC NORMALIZING RYC DATA...")
    for i in range(ryc_full_data["maldi"].shape[0]):
        TIC = np.sum(ryc_full_data["maldi"][i])
        ryc_full_data["maldi"][i] /= TIC

    print("TIC NORMALIZING GM DATA...")
    for i in range(gm_x.shape[0]):
        TIC = np.sum(gm_x[i])
        gm_x[i] /= TIC

    if norm_both:
        print("Standarizing data from both hospitals...")
        ryc_data = np.vstack(ryc_full_data["maldi"].values)
        both_data = np.concatenate((gm_x, ryc_data))
        scaler = StandardScaler()
        scaler.fit(both_data)
        del both_data

        gm_x = scaler.transform(gm_x)
        for i in range(ryc_full_data["maldi"].shape[0]):
            ryc_full_data["maldi"][i] = scaler.transform(ryc_data[i, :][np.newaxis, :])[0, :]

    else:
        print("Standarizing GM data alone...")
        scaler = StandardScaler()
        gm_x = scaler.fit_transform(gm_x)

        ryc_data = np.vstack(ryc_full_data["maldi"].values)
        print("Standarizing RYC data alone...")
        scaler = StandardScaler()
        scaler.fit(ryc_data)
        for i in range(ryc_full_data["maldi"].shape[0]):
            ryc_full_data["maldi"][i] = scaler.transform(ryc_data[i, :][np.newaxis, :])[0, :]

    if return_RYC and return_GM:
        gm_data = [gm_x, gm_y]
        with open("data/ryc_data_both.pkl", 'wb') as f:
            pickle.dump(ryc_full_data, f)
        with open("data/gm_data_both.pkl", 'wb') as f:
            pickle.dump(gm_data, f)
        return gm_data, ryc_full_data
    elif return_RYC:
        with open("data/ryc_data.pkl", 'wb') as f:
            pickle.dump(ryc_full_data, f)
        return ryc_full_data
    elif return_GM:
        gm_data = [gm_x, gm_y]
        with open("data/gm_data.pkl", 'wb') as f:
            pickle.dump(gm_data, f)
        return gm_x, gm_y, common_for_gm
