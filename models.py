
def LR_CV(penalty, solver, class_weight, random_state, max_iter, refit, l1_ratios=None):
    from sklearn.linear_model import LogisticRegressionCV
    model = LogisticRegressionCV(penalty=penalty,
                                 solver=solver, class_weight=class_weight,
                                 random_state=random_state, n_jobs=-1,
                                 max_iter=max_iter, refit=refit, l1_ratios=l1_ratios)

    return model


def GP_Classifier(grid):
    """
    Model to define the GP classifier to use.
    :param grid: dict, default RBF, DotProduct, Matern, RationalQuadratic and WhiteKernel. A grid with the parameter to
    cross-validate the GP.
    :return: model object. Returns the model object.
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel

    if grid is None:
        grid = {'kernel': [1 * RBF(), 1 * DotProduct(), 1 * Matern(), 1 * RationalQuadratic(), 1 * WhiteKernel()]}

    search = GridSearchCV(GaussianProcessClassifier(), grid, scoring='roc_auc', n_jobs=-1)
    return search



