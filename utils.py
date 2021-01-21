from matplotlib import pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import roc_curve, auc
import os
from sklearn.model_selection import KFold


def create_folds(x, y, n_folds=10, path="folds.pkl"):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    folds = {'train': [], 'test': []}
    for train_index, test_index in kf.split(x, y):
        folds['train'].append(train_index)
        folds['test'].append(test_index)
    with open(path, 'wb') as f:
        pickle.dump(folds, f)




def plot_sample(data_mz, data_int):
    plt.figure(figsize=(12, 4))
    plt.plot(data_mz[1], data_int[1])
    plt.title('Example of MALDI espectra')
    plt.xlabel('MZ')
    plt.ylabel('Intensity')


def save_results(model_name, model, y_pred, y_true, scores):
    path_to_save="./Results/"+model_name
    os.mkdir(path_to_save)
    with open(path_to_save + '/y_true.pkl', 'wb') as f:
        pickle.dump(y_true, f)
    with open(path_to_save + '/y_pred.pkl', 'wb') as f:
        pickle.dump(y_pred, f)
    with open(path_to_save + '/scores.pkl', 'wb') as f:
        pickle.dump(scores, f)


def load_results(main_folder, labels, file, print_r=True, plot=True, store=True):
    with open(main_folder, 'rb') as pkl:
        results = pickle.load(pkl)

    auc_mean = np.mean(results["AUC"], axis=0)
    fpr_mean = np.mean(results["FPR"], axis=0)
    fnr_mean = np.mean(results["FNR"], axis=0)
    dr_mean = np.mean(results["DR"], axis=0)
    feat_rank_mean = np.mean(results["feat_rank"], axis=0)
    auc_std = np.std(results["AUC"], axis=0)
    fpr_std = np.std(results["FPR"], axis=0)
    fnr_std = np.std(results["FNR"], axis=0)
    dr_std = np.std(results["DR"], axis=0)
    feat_rank_std = np.std(results["feat_rank"], axis=0)

    if print_r:
        for cat in range(len(labels)):
            print("---------------------"+labels[cat]+"-----------------------------")
            # print(confusion_matrix(y_true, prediction_binary, labels=[1,0]))
            print('AUC =', round(auc_mean[cat] * 100, 2), " +/- ", round(auc_std[cat] * 100, 2))
            print('Detection ratio =', round(dr_mean[cat] * 100, 2), " +/- ", round(dr_std[cat] * 100, 2))
            print('False Positive Ratio =', round(fpr_mean[cat], 2), " +/- ", round(fpr_std[cat] , 2))
            print('False Negative Ratio = ', round(fnr_mean[cat], 2), " +/- ", round(fnr_std[cat] , 2))

    # plots
    if plot:
        parent_path = os.path.split(file)[1]
        bar_plot(auc_mean * 100, 'AUC', categories=labels, store=store, store_name=parent_path+"auc")
        bar_plot(dr_mean * 100, 'Detection ratio', categories=labels, store=store, store_name=parent_path+"dr")
        bar_plot(fpr_mean * 100, 'False Positive Ratio', categories=labels, store=store,  store_name=parent_path+"fpr")
        bar_plot(fnr_mean * 100, 'False Negative Ratio', categories=labels, store=store, store_name=parent_path+"fnr")

    return results

# Evaluation of the model

# metrics


def false_positive_rate_one_cat(Y_test, predictions):
    false_pos = np.sum(np.logical_and(Y_test == 0, predictions == 1), axis=0)
    if np.sum(np.sum(Y_test == 0, axis=0)) == 0:
        print('Division by zero. There are only Resistant samples. FPR cannot be evaluated')
        return None
    else:
        false_pos_rate = false_pos / np.sum(Y_test == 0, axis=0)  ##False alarm
        return false_pos_rate


def false_negative_rate_one_cat(Y_test, predictions):
    false_neg = np.sum(np.logical_and(Y_test == 1, predictions == 0), axis=0)
    false_neg_rate = false_neg / np.sum(Y_test == 1, axis=0)  ## Missing ratio
    return false_neg_rate


def false_positive_rate(Y_test, predictions):
    false_pos = np.sum(np.logical_and(Y_test == 0, predictions == 1), axis=0)
    if np.sum(np.sum(Y_test == 0, axis=0)) == 0:
        print('Division by zero. There are only Resistant samples. FPR cannot be evaluated')
    else:
        false_pos_rate = false_pos / np.sum(Y_test == 0, axis=0)  ##False alarm
    return false_pos_rate


def false_negative_rate(Y_test, predictions):
    false_neg = np.sum(np.logical_and(Y_test == 1, predictions == 0), axis=0)
    false_neg_rate = false_neg / np.sum(Y_test == 1, axis=0) * 100  ## Missing ratio
    return false_neg_rate


def evaluate(Y_test, preds_cv, categories):
    from sklearn.metrics import roc_auc_score, accuracy_score

    preds = preds_cv > 0.5

    total_samples = preds_cv.shape[0]

    detected = np.sum(preds, axis=0)

    detection_perc = detected / total_samples * 100

    false_pos = np.sum(np.logical_and(Y_test == 0, preds == 1), axis=0)
    false_pos_rate = false_pos / np.sum(Y_test == 0, axis=0) * 100  ##False alarm

    false_neg = np.sum(np.logical_and(Y_test == 1, preds == 0), axis=0)
    false_neg_rate = false_neg / np.sum(Y_test == 1, axis=0) * 100  ## Missing ratio

    Detection_ratio = 100 - false_neg_rate

    accuracy_cat = []
    roc_auc_cat = []
    for cat in range(preds_cv.shape[1]):
        y_true = Y_test[:, cat]
        prediction_binary = preds[:, cat]
        prediction_prob = preds_cv[:, cat]

        acc = accuracy_score(y_true, prediction_binary)
        accuracy_cat.append(acc)

        roc_auc = roc_auc_score(y_true, prediction_prob, average='micro')
        roc_auc_cat.append(roc_auc)
        print('--------------------------------------------------')
        print(categories[cat])
        # print(confusion_matrix(y_true, prediction_binary, labels=[1,0]))
        print('Accuracy =', round(acc * 100, 2))
        print('AUC =', round(roc_auc * 100, 2))
        print('Ratio de Detección =', round(Detection_ratio[cat], 2))
        print('Ratio de Falsa Alarma =', round(false_pos_rate[cat], 2))
        print('Ratio de pérdidas = ', round(false_neg_rate[cat], 2))
    # print(classification_report(Y_test,preds, target_names= categories, zero_division =0))

    # plots
    bar_plot(np.array(roc_auc_cat) * 100, 'AUC')
    bar_plot(np.array(accuracy_cat) * 100, 'Precisión')
    bar_plot(Detection_ratio, 'Ratio de Detección')
    bar_plot(false_pos_rate, 'Falsa Alarma')
    bar_plot(false_neg_rate, 'Ratio de pérdidas')


# Graphs
def bar_plot(ratio, name, categories, store=False, store_name="default"):
    ax = plt.figure(figsize=(12, 6))
    plt.bar(range(len(categories)), ratio, tick_label=categories)
    plt.xticks(fontsize=10, rotation=30)
    plt.yticks(np.arange(0, 100, step=5), fontsize=9)
    plt.ylabel(name)
    plt.title('Antibiotic comparative wrt ' + name, fontsize=16)

    labels = [str(round(ratio[i], 2)) + '%' for i in range(len(ratio))]

    for index, data in enumerate(ratio):
        plt.text(x=index - 0.4, y=data + 1, s=str(round(data, 2)) + '%', fontdict=dict(fontsize=10))
    if store:
        path = "./Results/Plots/"+store_name+".png"
        plt.savefig(path)
    plt.show()


def bar_plot_color(ratio, name, categories, threshold, better_low=False):
    ''' Auxiliar function to plot ratios in a bar plot coloring each bar depending on the value.
    The threshold variable must be a list that contains two values, lower threshold and upper threshold to change color, in that order.
    Better_low is a boleean, if True, the lower values are considered better and coloured in green
    False is the default (Higher values in green) '''

    color_pallete = ['lightcoral', 'royalblue', 'mediumseagreen']
    if better_low:
        color_pallete.reverse()
    col = []
    for val in ratio:
        if val < threshold[0]:
            col.append(color_pallete[0])
        elif val > threshold[1]:
            col.append(color_pallete[2])
        else:
            col.append(color_pallete[1])

    ax = plt.figure(figsize=(15, 6))
    plt.bar(categories, ratio, color=col)
    plt.xticks(fontsize=10, rotation=30)
    plt.yticks(np.arange(0, 100, step=5), fontsize=9)
    plt.ylabel(name)
    plt.xlabel('Antibióticos')
    plt.title('Comparación por antibiótico de ' + name, fontsize=16)

    labels = [str(round(ratio[i], 2)) + '%' for i in range(len(ratio))]

    for index, data in enumerate(ratio):
        plt.text(x=index - 0.4, y=data + 1, s=str(round(data, 2)) + '%', fontdict=dict(fontsize=10))
    plt.show()


def create_box_plot_parameter_per_antibiotic(aucs, name_parameter):
    aucs_box_plot = ordered_dict_per_antibiotic(aucs)
    # list(aucs_box_plot.keys())

    fig = plt.figure(figsize=(10, 4))

    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])

    # Creating plot
    bp = ax.boxplot(list(aucs_box_plot.values()), patch_artist=True)
    ax.set_xticklabels(list(aucs_box_plot.keys()), rotation=40)
    ax.set_ylabel(name_parameter)
    ax.set_xlabel('Antibióticos')
    plt.setp(bp['boxes'], facecolor='lightseagreen')
    plt.setp(bp['medians'], linewidth=4)
    # show plot
    plt.title('BoxPlot de ' + name_parameter)
    plt.grid()
    plt.show()


# Tables
def results_table(fields, list_values):
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Table(header=dict(values=fields),
                                   cells=dict(values=list_values))
                          ])
    fig.show()


## Deal with dictionaries

def ordered_dict_per_antibiotic(dict):
    per_antibiotic = {}
    for k in dict[0].keys():
        per_antibiotic[k] = list(d[k] for d in dict)
    return per_antibiotic


def compute_mean_on_ordered_dict(dict):
    means = []
    for key, value in dict.items():
        means.append(np.nanmean(value) * 100)
    return means


def compute_std_on_ordered_dict(dict):
    std = []
    for key, value in dict.items():
        std.append(np.nanstd(value) * 100)
    return std


def mean_per_fold(list_of_dicts):
    means = [np.mean(list(d.values())) for d in list_of_dicts]
    return means


###### K-FOLD
### Visualization
def plot_cv_indices(cv, X, y, ax, n_splits, categories, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    add_space = 0
    cmap_cv = 'Set3'
    cmap_data = 'tab20c'
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    for i in range(y.shape[1]):
        add_space = add_space + 1
        ax.scatter(range(len(X)), [ii + add_space + .5] * len(X),
                   c=y[:, i], marker='_', lw=lw, cmap=cmap_data)

    # ax.scatter(range(len(X)), [ii + 2.5] * len(X),
    # c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + categories
    ax.set(yticks=np.arange(n_splits + len(categories)) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration", xlim=[0, X.shape[0]])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax


def plot_curve_auc_roc_in_CV(trues, preds, antibiotic_list):
    color_palette = ['teal', 'darkolivegreen', 'cadetblue', 'indianred',
                     'purple']  # ['lemonchiffon','gold', 'olive', 'goldenroad', 'darkkhaki']
    # plt.figure()
    subplot_index = -1
    col = 0
    fig, ax = plt.subplots(9, 2, figsize=(20, 65))
    for antibiotic in antibiotic_list:
        subplot_index = subplot_index + 1

        if subplot_index < 9:
            row = subplot_index
            col = 0
        else:
            col = 1
            row = subplot_index - 9

        axis = ax[row, col]
        axis.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                  label='Chance', alpha=.8)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for fold in range(len(trues)):
            fpr, tpr, threshold = roc_curve(trues[fold][antibiotic], preds[fold][antibiotic])
            roc_auc = auc(fpr, tpr)
            label = 'ROC Fold ' + str(fold) + ' = ' + str(round(roc_auc, 2))
            axis.plot(fpr, tpr, color_palette[fold], label=label, alpha=0.3)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        axis.plot(mean_fpr, mean_tpr, color='b',
                  label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                  lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        axis.fill_between(mean_fpr, tprs_lower, tprs_upper, color='darkgrey', alpha=.2,
                          label=r'$\pm$ 1 std. dev.')

        axis.set_title('ROC ' + antibiotic)

        axis.legend(loc='lower right')
        axis.plot([0, 1], [0, 1], 'r--')
        axis.set_xlim([0, 1])
        axis.set_ylim([0, 1])
        axis.set_ylabel('True Positive Rate')
        axis.set_xlabel('False Positive Rate')
        axis.grid(alpha=0.4)

    ax[8, 1].axis('off')
    # plt.tight_layout()
    plt.show()

    ############
    # Para Comparative

def retrieve_general_results(path):
    # Get the name and path of each file stored in the general directory /Models
    # Only recognizes those folders that end with _model!!!!!!
    list_subfolders_with_paths = [(f.path, f.name) for f in os.scandir(path) if f.is_dir()]
    # Select those files that correspond to models
    list_of_models = [(path, x) for path, x in list_subfolders_with_paths if x.endswith('_model')]
    models_names = [name for (path, name) in list_of_models]

    # Save the scores in a dictionary and the names in a list
    results = {}
    models_names = []
    for path, model in list_of_models:
        results[model] = load_results_from_files(path)
        # models_names.append(model)
    return results

def load_results_from_files(path):
    pass


def extract_parameter(results, parameter_name):
    # Organize those results from separated by folds to separated by antibiotic per each model
    ## First, create a nested dictionaty, with the model name as fist key, and antibiotic name as second key.
    new_results = {}
    for my_model, general_dict in results.items():
        for parameter, scores_per_antibiotic_dict in general_dict.items():
            if parameter in [parameter_name]:
                new_dict = {}
                for antibiotic, score in ordered_dict_per_antibiotic(scores_per_antibiotic_dict).items():
                    new_dict[antibiotic] = np.mean(score)
                new_results[my_model] = new_dict
    new_results

    ## Reorder in a new dict in which each antibiotic(key) has associated a list of scores (values) that corresponds to the models stores in _models_, with the same order
    models = []
    antib_dicts = []
    for my_model, general_dict in new_results.items():
        # print(my_model)
        models.append(my_model)
        antib_dicts.append(general_dict)
    final_dict = ordered_dict_per_antibiotic(antib_dicts)
    return final_dict, models


def bar_plot_compare_models_per_antibiotic_one_parameter(final_dict, models, parameter_name, threshold,
                                                         better_low=False):
    ##
    for key, scores in final_dict.items():

        plt.figure(figsize=(12, 4))
        perc_scores = [score * 100 for score in scores]

        color_pallete = ['lightcoral', 'royalblue', 'mediumseagreen']
        if better_low:
            color_pallete.reverse()

        col = []
        for val in perc_scores:
            if val < threshold[0]:
                col.append(color_pallete[0])
            elif val > threshold[1]:
                col.append(color_pallete[2])
            else:
                col.append(color_pallete[1])

        plt.bar(models, perc_scores, color=col)
        plt.ylim([0, 100])
        plt.xticks(fontsize=10, rotation=75)
        plt.yticks(np.arange(50, 100, step=5), fontsize=9)
        plt.ylabel(parameter_name)
        plt.title(key)

        # labels = [str(round(perc_score[i],2))+'%' for i in range(len(perc_scores))]

        for index, data in enumerate(perc_scores):
            plt.text(x=index - 0.2, y=data - 5, s=str(round(data, 2)) + '%', fontdict=dict(fontsize=10))
        plt.show()


def bar_plot_compare_models_per_antibiotic_all_parameters(path, threshold):
    # path = 'C:/Users/aida_/Desktop/TFM/Algoritmo/Models'
    results = retrieve_general_results(path)
    final_dict_auc, models = extract_parameter(results, 'AUC')
    final_dict_fnr, models = extract_parameter(results, 'FNR')
    final_dict_fpr, models = extract_parameter(results, 'FPR')
    final_dict_detection, models = extract_parameter(results, 'Detection_rate')

    width = 0.4
    color_pallete_auc = ['lightcoral', 'royalblue', 'mediumseagreen']
    color_pallete_det = ['firebrick', 'darkblue', 'seagreen']

    for key, scores in final_dict_auc.items():
        fig, ax = plt.subplots(figsize=(15, 5))
        perc_auc_scores = [score * 100 for score in scores]
        perc_det_scores = [score * 100 for score in final_dict_detection[key]]
        perc_fnr_scores = [score * 100 for score in final_dict_fnr[key]]
        perc_fpr_scores = [score * 100 for score in final_dict_fpr[key]]

        col_auc = []
        for val in perc_auc_scores:
            if val < threshold[0]:
                col_auc.append(color_pallete_auc[0])
            elif val > threshold[1]:
                col_auc.append(color_pallete_auc[2])
            else:
                col_auc.append(color_pallete_auc[1])

        col_det = []
        for val in perc_det_scores:
            if val < threshold[0]:
                col_det.append(color_pallete_det[0])
            elif val > threshold[1]:
                col_det.append(color_pallete_det[2])
            else:
                col_det.append(color_pallete_det[1])

        b1 = ax.bar(models, perc_auc_scores, width, color=col_auc, label='AUC', alpha=0.5)
        b2 = ax.bar(np.arange(len(models)) + width, perc_det_scores, width, color=col_det, alpha=0.5,
                    label='Detection ratio')
        plt.plot(np.arange(len(models)) + width / 2, perc_fnr_scores, marker='*', markersize=10, color='g', label='FNR')
        plt.plot(np.arange(len(models)) + width / 2, perc_fpr_scores, marker='o', markersize=10, color='purple',
                 label='FPR')

        plt.ylim([0, 104])
        plt.xlim([-0.5, len(models) + 1])
        plt.xticks(fontsize=10, rotation=75)
        plt.yticks(np.arange(0, 104, step=5), fontsize=9)
        plt.ylabel('Percentage')
        plt.title(key)
        plt.grid()

        # labels = [str(round(perc_score[i],2))+'%' for i in range(len(perc_scores))]

        for index, data in enumerate(perc_auc_scores):
            plt.text(x=index - 0.2, y=data - 5, s=str(round(data, 2)) + '%', fontdict=dict(fontsize=10))

        for index, data in enumerate(perc_det_scores):
            plt.text(x=index + width - 0.15, y=data - 5, s=str(round(data, 2)) + '%', fontdict=dict(fontsize=10))

        plt.legend()
        plt.show()


def compare_preprocess(ratio_model_1, ratio_model_2, name, categories, model_name):
    fig, ax = plt.subplots(figsize=(15, 6))
    width = 0.4
    b1 = ax.bar(categories, ratio_model_1, width, color='royalblue', label=model_name[0])

    b2 = ax.bar(np.arange(len(categories)) + width + 0.02, ratio_model_2, width, color='mediumseagreen',
                label=model_name[1])

    plt.xticks(fontsize=10, rotation=30)
    plt.yticks(np.arange(0, 100, step=5), fontsize=9)
    plt.ylabel(name)
    plt.xlabel('Antibióticos')
    plt.title('Comparación por antibiótico de ' + name, fontsize=16)

    for index, data in enumerate(ratio_model_1):
        plt.text(x=index - 0.4, y=data + 1, s=str(round(data, 2)) + '%', fontdict=dict(fontsize=10))

    for index, data in enumerate(ratio_model_2):
        plt.text(x=index - 0.4, y=data + 1, s=str(round(data, 2)) + '%', fontdict=dict(fontsize=10))

    plt.legend()
    plt.show()