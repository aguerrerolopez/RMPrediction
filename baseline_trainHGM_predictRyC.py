import pickle
import numpy as np
import json
import telegram
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as auc


def notify_ending(message):
    with open('./keys_file.json', 'r') as keys_file:
        k = json.load(keys_file)
        token = k['telegram_token']
        chat_id = k['telegram_chat_id']
    bot = telegram.Bot(token=token)
    bot.sendMessage(chat_id=chat_id, text=message)


# FAMILIAS
familias_hgm = { "penicilinas": ['AMOXI/CLAV ', 'PIP/TAZO'],
                    "cephalos": ['CEFTAZIDIMA', 'CEFOTAXIMA', 'CEFEPIME'],
                    "monobactams": ['AZTREONAM'],
                    "carbapenems": ['IMIPENEM', 'MEROPENEM', 'ERTAPENEM'],
                    "aminos": ['GENTAMICINA', 'TOBRAMICINA', 'AMIKACINA', 'FOSFOMICINA'],
                    "fluoro":['CIPROFLOXACINO'],
                    "otros":['COLISTINA']
                    }
familias_ryc = { "penicilinas": ['AMOXI/CLAV .1', 'PIP/TAZO.1'],
                    "cephalos": ['CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1'],
                    "monobactams": ['AZTREONAM.1'],
                    "carbapenems": ['IMIPENEM.1', 'MEROPENEM.1', 'ERTAPENEM.1'],
                    "aminos": ['GENTAMICINA.1', 'TOBRAMICINA.1', 'AMIKACINA.1', 'FOSFOMICINA.1'],
                    "fluoro":['CIPROFLOXACINO.1'],
                    "otros":['COLISTINA.1']
                    }
familias = familias_ryc

full = True

hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 1e-1, "max_it": int(1500)}}

data_path = "./data/hgm_data_mediansample_only2-12_TIC.pkl"
with open(data_path, 'rb') as pkl:
    hgm_data = pickle.load(pkl)

data_path = "./data/ryc_data_mediansample_only2-12_TIC.pkl"
with open(data_path, 'rb') as pkl:
    ryc_data = pickle.load(pkl)

results = np.zeros((1,15))

for i, familia in enumerate(familias):
    if familia=="cephalos":
        delay=len(familias["penicilinas"])
    elif familia=="monobactams":
        delay=len(familias["penicilinas"])+len(familias["cephalos"])
    elif familia=="carbapenems":
        delay=len(familias["penicilinas"])+len(familias["cephalos"])+len(familias["monobactams"])
    elif familia=="aminos":
        delay=len(familias["penicilinas"])+len(familias["cephalos"])+len(familias["monobactams"])+len(familias["carbapenems"])
    elif familia=="fluoro":
        delay=len(familias["penicilinas"])+len(familias["cephalos"])+len(familias["monobactams"])+len(familias["carbapenems"])+len(familias["aminos"])
    elif familia=="otros":
        delay=len(familias["penicilinas"])+len(familias["cephalos"])+len(familias["monobactams"])+len(familias["carbapenems"])+len(familias["aminos"])+len(familias["fluoro"])
    else:
        delay=0
    
    folds_path = "./data/HGM_10STRATIFIEDfolds_muestrascompensadas_"+familia+".pkl"
    with open(folds_path, 'rb') as pkl:
            folds = pickle.load(pkl)


    ryc_complete_samples = np.unique(ryc_data['binary_ab'][familias_ryc[familia]][~ryc_data['binary_ab'][familias_ryc[familia]].isna().any(axis=1)].index)
    hgm_x_tr, hgm_x_tst = hgm_data['maldi'].loc[folds["train"][0]], hgm_data['maldi'].loc[folds["val"][0]]
    hgm_y_tr, hgm_y_tst = hgm_data['binary_ab'].loc[folds["train"][0]], hgm_data['binary_ab'].loc[folds["val"][0]]
    
    x_tr = np.vstack((hgm_x_tr.values, hgm_x_tst.values))
    x_tst = np.vstack(ryc_data['maldi'].loc[ryc_complete_samples].values)

    y_tr = np.vstack((hgm_y_tr.values, hgm_y_tst.values))
    y_tst = ryc_data['binary_ab'][familias_ryc[familia]].loc[ryc_complete_samples].values

    rfc=RFC(random_state=42)
    param_grid = {'n_estimators': [100],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'criterion' :['gini']}

    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1)

    if y_tst.shape[1]==1:
        y_tr = y_tr.ravel()


    CV_rfc.fit(x_tr, y_tr)

    y_pred_clf = CV_rfc.predict_proba(x_tst)
    y_pred = np.zeros(y_tst.shape)
    if y_tst.shape[1]==1:
        y_pred = y_pred_clf[:, 1]
        results[0, delay] = auc(y_tst, y_pred)
        print(results)
    else:    
        for c in range(len(y_pred_clf)):
            results[0, c+delay] = auc(y_tst[:, c], y_pred_clf[c][:, 1])
            print(results)
print(results)







