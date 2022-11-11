#%%
import pickle
import numpy as np
import pandas as pd
import os
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pyteomics import mzml

# PATH TO MZML FILES
ryc_path = "./data/RyC/mzml"
# PATH TO EXCEL FILE WHERE THE AST IS
excel_path = "./data/RyC/RyC_AST.xlsx"
# Columns to read from excel data
cols = ['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']
# BOOLEAN TO NORMALIZE BY TIC
tic_norm=True

######################## READ AND PROCESS DATA ############################
# LOAD RYC MALDI-TOF
listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk(ryc_path):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]

# ============= READ AST INFO ============
full_data = pd.read_excel(excel_path, engine='openpyxl')

# READ DATA FROM FOLDS
data_int = []
id = []
hosp = []
# TAGS FOR SPANISH HOSPITALS
letter = np.array(["A", "B", "C", "D", "E", "F", "G", "H"])
# TAGS FOR PORTUGUESE HOSPITALS
number = np.array(["1", "2", "3", "4", "5", "6","7" , "8", "9", "10", "11"])
hospital_procedence = np.array(np.arange(19))
m = 5
hospital_labels = (((hospital_procedence[:,None] & (1 << np.arange(m)))) > 0).astype(int)
# CONVERT TO A PANDAS DATAFRAME
for file in listOfFiles:
    print(file)
    t = mzml.read(file)
    a = next(t)
    filename = file.split("/")[-1]
    erase_end = filename.split(".")[0]
    if erase_end.split("_")[0] in letter:
        data_int.append(a["intensity array"][2000:12000])
        id.append(erase_end.split("_")[0] + erase_end.split("_")[1])
        hosp.append(hospital_labels[11+np.where(letter == erase_end.split("_")[0])[0][0]])
    elif erase_end.split("_")[0] in number:
        data_int.append(a["intensity array"][2000:12000])
        id.append(erase_end.split("_")[0] + "-" + erase_end.split("_")[1])
        hosp.append(hospital_labels[np.where(number == erase_end.split("_")[0])[0][0]])
        
# CREATE A TAG FOR EACH HOSPITAL IN CASE YOU WANT TO IDENFITY THEM BY EACH HOSPITAL
ryc_data = pd.DataFrame(data=np.empty((len(data_int), 3)), columns=["Número de muestra", "maldi", "Hospital"])
ryc_data["maldi"] = data_int
ryc_data["Número de muestra"] = id
ryc_data["Hospital"] = hosp

# AS EVERY BACTERIA HAS MORE THAN ONE MALDI MS WE SELECT THE MORE SIMILAR TO THE MEDIAN
ryc_median = pd.DataFrame(data=np.vstack(data_int))
ryc_median['id'] = id
ryc_median["Hospital"] = hosp
ryc_median = ryc_median.set_index('id')

median_sample = ryc_median.groupby('id').median()

ryc_data_1s = pd.DataFrame(data=np.empty((len(median_sample), 3)), columns=["Número de muestra", "maldi", "Hospital"])
ryc_data_1s['Número de muestra'] = median_sample.index
ryc_data_1s = ryc_data_1s.set_index('Número de muestra')
data_median = []
hosp_unique = []
for s in median_sample.index:
    print(s)
    hosp_unique.append(ryc_median.loc[s]["Hospital"].sample(1).values[0])
    if ryc_median.loc[s].shape[0] == 10000:
        data_median.append(ryc_median.loc[s].iloc[:, :-1].values)
    else:
        data_median.append(ryc_median.iloc[:, :-1].loc[s].iloc[(median_sample.loc[s]-ryc_median.loc[s].iloc[:, :-1]).mean(axis=1).abs().argmin(), :].values)
ryc_data_1s['maldi'] = data_median
ryc_data_1s['Hospital'] = hosp_unique

# RELEASE MEMORY
del data_int, a, t, file, filename, id, filenames, letter, listOfFiles, erase_end, dirpath

# RYC AST DATA
aux_ryc = full_data.loc[full_data['Centro'] == 'RyC'].copy().set_index("Número de muestra").drop('E11')
ryc_full_data = pd.merge(how='outer', left=ryc_data_1s, right=aux_ryc, left_on='Número de muestra', right_on='Número de muestra')


#%%
################################ FOLDS CREATION ##########################
ryc_full_data['Fenotipo ESBL']
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# GET RID OF MISSING SAMPLES FOR FOLD CREATION(if exists)
complete_samples = np.unique(ryc_full_data[~ryc_full_data[cols].isna().any(axis=1)].index)
missing_samples = np.unique(ryc_full_data[ryc_full_data[cols].isna().any(axis=1)].index).tolist()
gm_complete_y = ryc_full_data[cols].loc[complete_samples]

# PATH TO SAVE FOLDS
fold_storage_name = "data/RYC_5STRATIFIEDfolds_paper.pkl"
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
            print("It should be corrected now "+col)
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


###################### Categorizes the types ###############
ryc_full_data['CP gene'][ryc_full_data['CP gene'].isna()]=0
ryc_full_data['CP gene'][ryc_full_data['CP gene']=='KPC-3']=0
ryc_full_data['CP gene'][ryc_full_data['CP gene']=='OXA-181']=0
ryc_full_data['CP gene'][ryc_full_data['CP gene']=='OXA-48']=1
ryc_full_data['CP gene'][ryc_full_data['CP gene']=='OXA-48+VIM-2']=1
ryc_full_data['CP gene'][ryc_full_data['CP gene']=='KPC-3+VIM-2']=0
ryc_full_data['CP gene'][ryc_full_data['CP gene']=='NDM-1']=0

cols = ['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL', 'CP gene']
ryc_dict = {"maldi": ryc_full_data['maldi'].copy(),
            "full": ryc_full_data.copy()}

with open("data/ryc_data_expAug.pkl", 'wb') as f:
    pickle.dump(ryc_dict, f)