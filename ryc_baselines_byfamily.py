import pickle
import numpy as np
import json
import telegram
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as auc
import matplotlib.pyplot as plt
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
familias = {"penicilinas": ['AMOXI/CLAV .1', 'PIP/TAZO.1'],
                    "cephalos": ['CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1'],
                    "monobactams": ['AZTREONAM.1'],
                    "carbapenems": ['IMIPENEM.1', 'MEROPENEM.1', 'ERTAPENEM.1'],
                    "fluoro":['CIPROFLOXACINO.1'],
                    "aminos": ['GENTAMICINA.1', 'TOBRAMICINA.1', 'AMIKACINA.1', 'FOSFOMICINA.1'],
                    "otros":['COLISTINA.1']
                    }

full = True

hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 1e-1, "max_it": int(1500)}}

data_path = "./data/ryc_data_mediansample_only2-12_TIC.pkl"
with open(data_path, 'rb') as pkl:
        ryc_data = pickle.load(pkl)

random_states = [0, 10, 20, 30, 40]
results = np.zeros((10,15))

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
    
    folds_path = "./data/RYC_10STRATIFIEDfolds_nomissing_muestrascompensadas_"+familia+".pkl"
    with open(folds_path, 'rb') as pkl:
        ryc_folds = pickle.load(pkl)

    plt.figure()
    for fold in range(10):
        x_tr = np.vstack(ryc_data['maldi'].loc[ryc_folds["train"][fold]].values)
        x_tst = np.vstack(ryc_data['maldi'].loc[ryc_folds["val"][fold]].values)

        y_tr = ryc_data['binary_ab'][familias[familia]].loc[ryc_folds["train"][fold]].values
        y_tst = ryc_data['binary_ab'][familias[familia]].loc[ryc_folds["val"][fold]].values

        if y_tst.shape[1]==1:
            y_tr = y_tr.ravel()
            clf=SVC(probability=1, kernel="linear")
            param_grid = {'C': [0.01, 0.1 , 1, 10]}
        else:
            clf = MultiOutputClassifier(SVC(probability=True, kernel="linear"))
            param_grid = {'estimator__C': [0.01, 0.1 , 1, 10]}     
            # knn = KNeighborsClassifier(n_jobs=-1)
            # param_grid = {'n_neighbors': range(1,20)}
            # rfc=RFC(random_state=42)
            # param_grid = {'n_estimators': [100],
                            # 'max_features': ['auto', 'sqrt', 'log2'],
                            # 'criterion' :['gini']}

        CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
        CV_rfc.fit(x_tr, y_tr)

        print(CV_rfc.best_params_)

        y_pred_clf = CV_rfc.predict_proba(x_tst)
        y_pred = np.zeros(y_tst.shape)
        if y_tst.shape[1]==1:
            y_pred = y_pred_clf[:, 1]
            results[fold, delay] = auc(y_tst, y_pred)
        else:    
            for c in range(len(y_pred_clf)):
                results[fold, c+delay] = auc(y_tst[:, c], y_pred_clf[c][:, 1])

        # plt.plot(CV_rfc.best_estimator_.feature_importances_, label="| Fold:" +str(fold))
    plt.legend()
    plt.title("Familia: "+familia)
    plt.show()

print(results)
print(np.mean(results, axis=0))
print(np.std(results, axis=0))







