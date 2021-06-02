import pickle
import numpy as np
import json
import telegram
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as auc
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC


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

coef_by_fam = np.zeros((7, 10000))
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
    
    folds_path = "./data/HGM_10STRATIFIEDfolds_nomissing_muestrascompensadas_"+familia+".pkl"
    with open(folds_path, 'rb') as pkl:
            folds = pickle.load(pkl)


    hgm_complete = np.unique(hgm_data['binary_ab'][familias_hgm[familia]][~hgm_data['binary_ab'][familias_hgm[familia]].isna().any(axis=1)].index)
    ryc_complete_samples = np.unique(ryc_data['binary_ab'][familias_ryc[familia]][~ryc_data['binary_ab'][familias_ryc[familia]].isna().any(axis=1)].index)
    
    x_tr = np.vstack(hgm_data['maldi'].loc[hgm_complete].values) 
    y_tr = hgm_data['binary_ab'][familias_hgm[familia]].loc[hgm_complete].values

    x_tst = np.vstack(ryc_data['maldi'].loc[ryc_complete_samples].values)
    y_tst = ryc_data['binary_ab'][familias_ryc[familia]].loc[ryc_complete_samples].values

    # clf=RFC(random_state=42)
    # param_grid = {'n_estimators': [100],
    #                 'max_features': ['auto', 'sqrt', 'log2'],
    #                 'criterion' :['gini']}
    if y_tst.shape[1]==1:
        y_tr = y_tr.ravel()
        clf=SVC(probability=1, kernel="linear")
        param_grid = {'C': [0.01, 0.1 , 1, 10]}
    else:
        clf = MultiOutputClassifier(SVC(probability=True, kernel="linear"))
        param_grid = {'estimator__C': [0.01, 0.1 , 1, 10]}     

    CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)

    if y_tst.shape[1]==1:
        y_tr = y_tr.ravel()


    CV_rfc.fit(x_tr, y_tr)
    

    y_pred_clf = CV_rfc.predict_proba(x_tst)
    y_pred = np.zeros(y_tst.shape)
    if y_tst.shape[1]==1:
        y_pred = y_pred_clf[:, 1]
        results[0, delay] = auc(y_tst, y_pred)
        print(results)
        coef_by_fam[i, :]=CV_rfc.best_estimator_.coef_.ravel()
    else:    
        coef_mean = 0
        for c in range(y_tst.shape[1]):
            results[0, c+delay] = auc(y_tst[:, c], y_pred_clf[c][:, 1])
            coef_mean += CV_rfc.best_estimator_.estimators_[c].coef_.ravel()
            print(results)
        coef_by_fam[i, :]= coef_mean/y_tst.shape[1]
    
    
    
print(results)

from matplotlib import pyplot as plt


for i, fam in enumerate(familias):
    plt.figure(figsize=[20, 10])
    plt.title("Feature importance in mean and std in "+fam)
    plt.errorbar(x=range(0,10000), y=coef_by_fam[i,:], fmt='o', color='black',
             ecolor='lightgray',  alpha=0.3, label=fam)
    plt.show()


# JUST A SAMPLE
familia="carbapenems"
for ab in familias_ryc[familia]:
    plt.figure(figsize=[20, 10])
    plt.title("Resistant and sensible sample for "+ab)
    plt.plot(coef_by_fam[3, :],marker='o', color="black", alpha=0.2, label="SVM coeficients of "+familia)
    pos_sample = np.vstack(ryc_data['maldi'].loc[ryc_data['binary_ab'][ab][ryc_data['binary_ab'][ab]==1].sample(1).index]).ravel()
    neg_sample = np.vstack(ryc_data['maldi'].loc[ryc_data['binary_ab'][ab][ryc_data['binary_ab'][ab]==0].sample(1).index]).ravel()
    plt.plot(pos_sample, color='green', label=ab+": Resistent sample")
    plt.plot(neg_sample, color='orange', label=ab+": Sensible sample")
    plt.legend()
    plt.show()

# MEAN OF THE POS AND NEG SAMPLE

for ab in familias[familia]:
    plt.figure(figsize=[20, 10])
    plt.title("Resistant and sensible mean for "+ab)
    plt.plot(coef_by_fam[3, :],marker='o', color="black", alpha=0.2, label="SVM coeficients of  "+familia)
    pos_sample = np.mean(np.vstack(ryc_data['maldi'].loc[ryc_data['binary_ab'][ab][ryc_data['binary_ab'][ab]==1].index]), axis=0)
    neg_sample = np.mean(np.vstack(ryc_data['maldi'].loc[ryc_data['binary_ab'][ab][ryc_data['binary_ab'][ab]==0].index]), axis=0)
    plt.plot(pos_sample, color='green', label=ab+": Resistent MEAN")
    plt.plot(neg_sample, color='orange', label=ab+": Sensible MEAN")
    plt.legend()
    plt.show()





