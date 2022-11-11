import pickle
import numpy as np
import pandas as pd
import os
from pyteomics import mzml
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# PATH TO MZML FILES
hgm_rep_mzml_path = './data/GM/mzml'
# PATH TO THE EXCEL FILE WITH ANTIBIOTIC RESISTANCE DATA
excel_path = "./data/GM/GM_AST.xlsx"

# BOOLEAN TO NORMALIZE BY TIC
tic_norm=True
# COLS TO STORE FROM THE EXCEL
cols = ['CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']

######################## READ AND PROCESS DATA ############################
# READ DATA MZML
listOfFiles = list()
for (dirpath, dirnames, filenames_rep) in os.walk(hgm_rep_mzml_path):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames_rep]

id_samples_rep = []
maldis = []
# CONVERT TO A PANDAS DATAFRAME
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

# AS EVERY BACTERIA HAS MORE THAN ONE MALDI MS WE SELECT THE MORE SIMILAR TO THE MEDIAN
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

# READ EXCEL DATA
excel_data = pd.read_excel(excel_path, engine='openpyxl', dtype={'Nº Espectro': str})

# FINALLY JOIN THE MALDIs + ANTIBIOTIC RESISTANCE DATA INTO ONE DATAFRAME
gm_full_data = pd.merge(how='outer', left=gm_median_1s, right=excel_data, left_on='Nº Espectro',
                        right_on='Nº Espectro').set_index("Nº Espectro")

#%%
################################ FOLDS CREATION ##########################

# GET RID OF MISSING SAMPLES FOR FOLD CREATION (IF EXISTS)
complete_samples = np.unique(gm_full_data[~gm_full_data[cols].isna().any(axis=1)].index)
missing_samples = np.unique(gm_full_data[gm_full_data[cols].isna().any(axis=1)].index).tolist()
gm_complete_y = gm_full_data[cols].loc[complete_samples]

# PATH TO SAVE THE STRATIFIED FOLDS
fold_storage_name = "data/GM_5STRATIFIEDfolds_paper.pkl"
gm_folds = {"train": [], "val": []}
# MULTILABEL STRATIFICATION FOLDS
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for tr_idx, tst_idx in mskf.split(complete_samples, gm_complete_y):
    train_samples = complete_samples[tr_idx].tolist()
    test_samples = complete_samples[tst_idx].tolist()
    for col in gm_complete_y.columns:
        # HERE WE CHECK FOR UNBALANCE IN TRAINING FOLDS
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
            print("IT SHOULD BE CORRECTED NOW FOR "+col)
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

# ############################### NORMALIZE DATA AND STORE IT ##########################
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

with open("data/gm_data_paper.pkl", 'wb') as f:
    pickle.dump(gm_dict, f)
