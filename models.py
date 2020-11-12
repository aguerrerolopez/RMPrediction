from sklearn.linear_model import LogisticRegressionCV


def LR_CV(penalty, solver, class_weight, random_state, max_iter, refit, l1_ratios=None):

    model = LogisticRegressionCV(penalty=penalty,
                                 solver=solver, class_weight=class_weight,
                                 random_state=random_state, n_jobs=-1,
                                 max_iter=max_iter, refit=refit, l1_ratios=l1_ratios)

    return model


def GP_Classifier():
    return None


# train(model=model, model_name="LR_L2", x=x, y=y, labels_list=labels_name.tolist(), folds=cv_folds, th=th)

# model = LogisticRegressionCV(penalty='elasticnet',
#                              solver='saga', class_weight='balanced',
#                              random_state = 32, n_jobs = -1, max_iter=10, refit=True, l1_ratios=[0.3,0.5,0.8])
#
# train(model=model, model_name="LR_enet", x=x, y=y, labels_list=labels_name.tolist(), folds=cv_folds, th=th)

