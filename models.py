


def LR_CV(penalty, solver, class_weight, random_state, max_iter, refit, l1_ratios=None):
    from sklearn.linear_model import LogisticRegressionCV
    model = LogisticRegressionCV(penalty=penalty,
                                 solver=solver, class_weight=class_weight,
                                 random_state=random_state, n_jobs=-1,
                                 max_iter=max_iter, refit=refit, l1_ratios=l1_ratios)

    return model


def GP_Classifier():
    from sklearn.model_selection import GridSearchCV
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel

    grid = {'kernel': [1 * RBF(), 1 * DotProduct(), 1 * Matern(), 1 * RationalQuadratic(), 1 * WhiteKernel()]
            }
    search = GridSearchCV(GaussianProcessClassifier(), grid, scoring='roc_auc', n_jobs=-1)
    return search


# train(model=model, model_name="LR_L2", x=x, y=y, labels_list=labels_name.tolist(), folds=cv_folds, th=th)

# model = LogisticRegressionCV(penalty='elasticnet',
#                              solver='saga', class_weight='balanced',
#                              random_state = 32, n_jobs = -1, max_iter=10, refit=True, l1_ratios=[0.3,0.5,0.8])
#
# train(model=model, model_name="LR_enet", x=x, y=y, labels_list=labels_name.tolist(), folds=cv_folds, th=th)

