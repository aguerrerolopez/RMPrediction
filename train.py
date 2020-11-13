import os
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import numpy as np
import utils
from datetime import datetime
import pickle


def train(model, model_name, x, y, labels_list, save_path=None, folds=10, th=0.5):
    """
    Train method. Given a model to train and the data to use, this method trains the model for "k" stratified folds. Then
    it saves the results in the path given.
    :param model: model class. Expects a model class, i.e. GPC, SVC, LR.
    :param model_name: str. Name used to save the results and the model itself.
    :param x: numpy array, size NxD. Expects a numpy array with the spectra data of size NxD.
    :param y: numpy array, size NxC. Expects a numpy array of multilabel targets of size NxC being C the number of
    categories to predict.
    :param labels_list: list, size 1xC. Expects a list with the name of each category to classify.
    :param save_path: str or PathLike. Path to the directory where do you want to save the model.
    :param folds: int, default 10. Number of folds to do the cross-validation.
    :param th: int, default 0.5. Threshold to do the classification. If the probability of belong to a C_i class is above
    the threshold, this class is the predicted.
    :return: None.
    """
    if save_path is None:
        save_path = "./Results/" + model_name
    else:
        save_path += model_name
    try:
        os.mkdir(save_path)
        print("Folder to store the model created at: ", save_path)
    except OSError:
        print("The folder already exists at: ", save_path)

    scores = {"AUC": np.zeros([folds, len(labels_list)]),
              "AUPRC": np.zeros([folds, len(labels_list)]),
              "FNR": np.zeros([folds, len(labels_list)]),
              "FPR": np.zeros([folds, len(labels_list)]),
              "DR": np.zeros([folds, len(labels_list)]),
              # TODO: cambiar la dimensión, no es la dimensión de Y, sino la del fold
              "Y_true": np.zeros([folds, len(labels_list), y.shape[0]]),
              "Y_pred": np.zeros([folds, len(labels_list), y.shape[0]]),
              "feat_rank": np.zeros([folds, len(labels_list), x.shape[1]])}


    c = 0
    # TODO: girar los fors para poder hacer stratified
    skf = KFold(n_splits=folds, shuffle=True, random_state=0)
    for train_idx, test_idx in skf.split(x, y):

        print("=========Training ", c, " fold ==========")
        x_tr, y_tr, x_tst, y_tst = x[train_idx], y[train_idx], x[test_idx], y[test_idx]

        # We train the model for each antibiotic
        for i, cat in enumerate(labels_list):
            x_tr_1 = x_tr
            y_tr_1 = y_tr[:, i]

            # If missing weren't imputed we erase the samples than has a NaN for this antibiotic
            pos_nonan = np.where(np.isnan(y_tr_1) == False)[0]
            x_tr_1 = x_tr_1[pos_nonan, :]
            y_tr_1 = y_tr_1[pos_nonan]
            # In test set
            pos_nonan = np.where(np.isnan(y_tst[:, i]) == False)[0]
            x_tst_1 = x_tst[pos_nonan, :]
            y_tst_1 = y_tst[pos_nonan, i]

            # Train the model:
            model.fit(x_tr_1, y_tr_1)
            y_pred_1 = model.predict_proba(x_tst_1)[:, 1]

            # Calculate and store the results
            # scores["Y_pred"][c, i, :] = y_pred_1
            # scores["Y_true"][c, i, :] = y_tst_1
            if model_name != "GP":
                scores["feat_rank"][c, i, :] = model.coef_
            if len(set(y_tst_1))>1:
                scores["AUC"][c, i] = roc_auc_score(y_tst_1, y_pred_1)
                y_pred_b = y_pred_1 > th
                scores["FNR"][c, i] = utils.false_negative_rate_one_cat(y_tst_1, y_pred_b)
                scores["FPR"][c, i] = utils.false_positive_rate_one_cat(y_tst_1, y_pred_b)
                scores["DR"][c, i] = 1-scores["FNR"][c, i]
                print("AUC for fold ", c, " is:"  ,scores["AUC"][c, i])
            else:
                scores["AUC"][c, i] = float("NaN")
                scores["FNR"][c, i] = float("NaN")
                scores["FPR"][c, i] = float("NaN")
                scores["DR"][c, i] = float("NaN")

        c+=1

    scores_name = model_name + '_at_' + datetime.now().strftime('%d-%m-%Y_%H-%Mh') + '.pkl'
    with open(save_path+"/"+scores_name, 'wb') as f:
        pickle.dump(scores, f)
