# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 18:48:15 2020

@author: aida_

FUNCTIONS TFM
"""
# pip installs required
# pip install 
# pip install dash # para plotly (tablas)

## PREPROCESSING

# Basics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import _pickle # cPickle for python 2.x
import zipfile
import pickle

# Custom functions
import sys
sys.path.append('C:/Users/aida_/Desktop/TFM/Algoritmo')
import tfm_functions as tf

# Aditional tools
import os
from joblib import dump, load 
from datetime import datetime
import json
from impyute.imputation.cs import fast_knn
from scipy import interpolate as interp
from scipy.signal import find_peaks
from BaselineRemoval import BaselineRemoval
from scipy.signal import savgol_filter
from msalign import Aligner

# Aditional tools from sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 

# Model
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import MultiTaskElasticNetCV

# Scoring metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc
# Functions

### To import the data
def import_data(data_path):
# Read Train data
    #data_path = 'C:/Users/aida_/Desktop/TFM/Algoritmo/DATOS_DEFI/test_4/'
    zf = zipfile.ZipFile(data_path + 'zipped_TrainData.zip', 'r')
    df_train = _pickle.loads(zf.open('TrainData.pkl').read())
    zf.close()

    # Read Test data
    zf = zipfile.ZipFile(data_path +'zipped_TestData.zip', 'r')
    df_test = _pickle.loads(zf.open('TestData.pkl').read())
    zf.close()
    return df_train, df_test


def clean_df(df):
        # Cleaning part: Changes in the antibiotics used here!
    columnstodrop = ['AMPICILINA'] ## 'CEFOTAXIMA', 'CEFEPIME', 'AZTREONAM']
    #df_clean = df_clean.rename(columns={"CEFTAZIDIMA": "BLEE"})

    return df.drop(columns=columnstodrop)

def obtain_labels(df, impute):
    df_clean = clean_df(df)
    # PREPARE LABELS
    categories = df_clean.columns[6:-1]
    Y = df_clean[categories].values
    
    if impute:
        imputed_knn=fast_knn(Y,30)
        imputed_labels=imputed_knn>0.95
        Y = imputed_labels.astype(int)

    return Y, categories


def obtain_labels_no_cleaning(df, impute):
    #df_clean = clean_df(df)
    # PREPARE LABELS
    categories = df.columns[6:-1]
    Y = df[categories].values
    
    if impute:
        imputed_knn=fast_knn(Y,30)
        imputed_labels=imputed_knn>0.95
        Y = imputed_labels.astype(int)

    return Y, categories


def clean_and_prepare_data(df):

    # Cleaning part
    df_clean = clean_df(df)

    # Extract data from dataframes
    spectra = df_clean['intensity'].values
    coord_mz = df_clean['mz_coords'].values

    # PREPROCESING

    #STEP 0: Align the coordinates
    ## Create the new axis
    all_values= []
    max_val = 20000
    min_val = 0
    for i in range(0,spectra.shape[0]):
        arr = np.round(coord_mz[i], decimals=2)
        aux = np.concatenate((all_values, arr))
        all_values = np.unique(aux)
        if np.amax(arr)<max_val: # Find the smallest maximun value
            max_val=np.amax(arr)
        if np.amin(arr)>min_val:# Find the bigger minimun value
            min_val=np.amin(arr)
        new_coord_mz_axis = np.arange(min_val,max_val, 0.2)
    ## Fill the zxis interpolating each spectrum
    interpolated_spectra=np.zeros((spectra.shape[0],new_coord_mz_axis.shape[0]))
    for i in range(0,spectra.shape[0]):
        my_interp = interp.interp1d(coord_mz[i],spectra[i])
        interpolated_spectra[i,:] = my_interp(new_coord_mz_axis)

    X = interpolated_spectra




    # PREPARE LABELS
    categories = df_clean.columns[6:-1]
    Y = df_clean[categories].values

    return X,Y, categories


### PREPROCESSING FUNCTIONS

# STEP O :
def crop_spectra(df_clean):

    # Extract data from dataframes
    spectra = df_clean['intensity'].values
    coord_mz = df_clean['mz_coords'].values

    # PREPROCESING

    #STEP 0: Align the coordinates
    ## Create the new axis
    all_values= []
    max_val =20000
    min_val=0
    for i in range(0,spectra.shape[0]):
        arr = np.round(coord_mz[i], decimals=2)
        aux = np.concatenate((all_values, arr))
        all_values = np.unique(aux)
        if np.amax(arr)<max_val: # Find the smallest maximun value
            max_val=np.amax(arr)
        if np.amin(arr)>min_val:# Find the bigger minimun value
            min_val=np.amin(arr)


    new_coord_mz_axis = np.arange(round(min_val),round(max_val),1)
    interpolated_spectra=np.zeros((spectra.shape[0],new_coord_mz_axis.shape[0]))
    for i in range(0,spectra.shape[0]):
        my_interp = interp.interp1d(coord_mz[i],spectra[i],'zero',)
        interpolated_spectra[i,:] = my_interp(new_coord_mz_axis)

    X = interpolated_spectra

    return X, new_coord_mz_axis

   

# STEP 1: NORMALIZE

def normalize(spectra, norm_type):
    if norm_type == 'no_norm':
        spectra=spectra
    elif norm_type == 'TIC':
        for i in range(spectra.shape[0]):
            TIC = np.sum(spectra[i])
            normalized_spectrum = spectra[i]/TIC
            spectra[i]=normalized_spectrum
    elif norm_type == 'median':
        for i in range(spectra.shape[0]):
            spectra[i]=normalize_median(spectra[i])
    return spectra

def normalize_median(spectrum):
    median = np.median(spectrum)
    normalized_spectrum = spectrum/median
    return normalized_spectrum

# STEP 2:

def standard_sc(X, isSC):
    if isSC:
        # Normalize the data (this normalization is over features)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return X

# STEP 3:

    
def preprocess(df, step, limits,interpolation_type,norm_type, isSC):
    df_clean = clean_df(df)
    X_eq, new_coord_mz_axis = equal_mz(df_clean, step, limits, interpolation_type)
    X_n = normalize(X_eq,norm_type)
    X = standard_sc(X_n,isSC)
    return X

def preprocess_mz_out(df, step, limits,interpolation_type,norm_type, isSC):
    df_clean = clean_df(df)
    X_eq, new_coord_mz_axis = equal_mz(df_clean, step, limits, interpolation_type)
    X_n = normalize(X_eq,norm_type)
    X = standard_sc(X_n,isSC)
    return X, new_coord_mz_axis

def preprocess_peaks_alineation(df, step, limits,interpolation_type,norm_type, isSC):
    df_clean = clean_df(df)
    X_eq, new_coord_mz_axis = equal_mz(df_clean, step, limits, interpolation_type)
    X_n = normalize(X_eq,norm_type)
    X_bs_rem = baseline_removal_smoothing(X_n)
    X_align = peak_alineation(X_bs_rem, new_coord_mz_axis)
    X = standard_sc(X_align,isSC)

    return X, new_coord_mz_axis


def baseline_removal_smoothing(intensity_matrix):
    polynomial_degree = 15
    intensity_to_align_array = np.zeros(intensity_matrix.shape)
    for line in range(intensity_matrix.shape[0]):
        to_filter = intensity_matrix[line,:]

        # Smoothing 
        yhat = savgol_filter(to_filter, 51, 3) # window size 51, polynomial order 3
        #plt.plot(yhat)
        
        input_array= yhat
        # Baseline removal
        baseObj=BaselineRemoval(input_array)
        Modpoly_output=baseObj.IModPoly(polynomial_degree)
        #Modpoly_output=baseObj.ZhangFit()

        
        intensity_to_align_array[line,:]=Modpoly_output

    return intensity_to_align_array


def peak_alineation(intensity_to_align_array, new_coord_mz_axis):
    # compute mean spectrum
    mean_spectrum = np.mean(intensity_to_align_array, axis=0)

    # Find peaks in mean spectrum
    peaks, _ = find_peaks(mean_spectrum, height = 0.00001,  distance=70)

    # compute variance along spectra
    var_spectrum = np.var(intensity_to_align_array, axis=0)


    # Select those peaks with low variance along spectra
    no_var_peaks = []
    for peak_index in peaks:
        if var_spectrum[peak_index]<0.0007:
            no_var_peaks.append(peak_index)

    selected_pos = [ no_var_peaks[i] for i in np.argsort(var_spectrum[no_var_peaks])[:4]]
    selected_peaks = new_coord_mz_axis[selected_pos]
    print('Selected peaks',selected_peaks)

    # Align the spectra
    ## Create X vector
    x = new_coord_mz_axis

    array = intensity_to_align_array.T
    peaks = np.sort(selected_peaks)
    #weights = [60, 100, 60, 100]

    # instantiate aligner object
    aligner = Aligner(
        x, 
        array, 
        peaks, 
        #weights=weights,
        return_shifts=True,
        align_by_index=True,
        only_shift= False,
        method="cubic", ## Cubic es más rápido que pchip
        iterations=5
    )
    aligner.run()
    aligned_array, shifts_out = aligner.align() 

    return aligned_array


def equal_mz(df, step, limits,interpolation_type):
    from scipy import interpolate as interp
    spectra = df['intensity'].values
    coord_mz = df['mz_coords'].values
    # Check if all the spectra can be interpolated to the nes limits
    all_values= []
    max_val =20000
    min_val=0
    for i in range(0,spectra.shape[0]):
        arr = np.round(coord_mz[i], decimals=2)
        aux = np.concatenate((all_values, arr))
        all_values = np.unique(aux)
        if np.amax(arr)<max_val: # Find the smallest maximun value
            max_val=np.amax(arr)
        if np.amin(arr)>min_val:# Find the bigger minimun value
            min_val=np.amin(arr)

    if (min_val>limits[0]) | (max_val<limits[1]):
        limits[0]=np.max(min_val,limits[0])
        limits[1]=np.max(max_val,limits[1])
        print('Interpolation in the limits selected cannot be accomplished')
        print('New interpolation limits are', limits)
        
    # Create the new axis and interpolate
    new_coord_mz_axis = np.arange(round(limits[0]),round(limits[1]),step)
    interpolated_spectra=np.zeros((spectra.shape[0],new_coord_mz_axis.shape[0]))
    for i in range(0,spectra.shape[0]):
        my_interp = interp.interp1d(coord_mz[i],spectra[i],'zero')
        interpolated_spectra[i,:] = my_interp(new_coord_mz_axis)

    X = interpolated_spectra
    return X, new_coord_mz_axis


def downsample(X_train, Y_train ):
    
    balanced_list = np.ones((Y_train.shape[1],1))
    balanced_list = np.sum(Y_train, axis=0)/Y_train.shape[0]<0.5
    #print(balanced_list)
    
    new_sets =[]
    for ind, not_balanced in enumerate(balanced_list):
        #print(ind)
        if not_balanced:
            Y_aux_resistant = Y_train[Y_train[:,ind]==1, ind]
            X_aux_resistant = X_train[Y_train[:,ind]==1]
            Y_aux_non_resistant = Y_train[Y_train[:,ind]==0, ind]
            X_aux_non_resistant = X_train[Y_train[:,ind]==0]

            n_samples =X_aux_resistant.shape[0]
            X_resample = np.vstack((X_aux_non_resistant[:n_samples, :], X_aux_resistant))
            Y_resample = np.hstack((Y_aux_non_resistant[:n_samples], Y_aux_resistant))
        else:
            X_resample = X_train 
            Y_resample = Y_train[:,ind]
            
        new_sets.append([X_resample, Y_resample])
        
    return new_sets


#### MODEL

## Training process

def save_results_in_files(main_folder, aucs, false_negative_rates, false_positive_rates, detection_ratio, predictions, true_values, feature_importance):
   
    ''' To save the results in JSON and pickle files
    These are snippets to sew how to load them after

    # Snippet to read pickle files
    with open(main_folder + '/predictions.pkl','rb') as f:
        predispredis = pickle.load(f)
        print(predispredis)

    with open(main_folder + '/true_values.pkl','rb') as f:
        truestrues = pickle.load( f)
        print(truestrues)

    # Code snippet to open json files
    with open(main_folder+'\AUCS') as json_file:
        data = json.load(json_file)

    '''

    # Save the results
    with open(main_folder + '/AUCS' , 'w') as fout:
        json.dump(aucs, fout)

    with open(main_folder + '/FNR' , 'w') as fout:
        json.dump(false_negative_rates, fout)

    with open(main_folder + '/FPR' , 'w') as fout:
        json.dump(false_positive_rates, fout)

    with open(main_folder + '/DetectionRate' , 'w') as fout:
        json.dump(detection_ratio, fout)

    with open(main_folder + '/predictions.pkl','wb') as f:
        pickle.dump(predictions, f)

    with open(main_folder + '/true_values.pkl','wb') as f:
        pickle.dump(true_values, f)

    with open(main_folder + '/Feature_importances.pkl','wb') as f:
        pickle.dump(feature_importance, f)


def load_results_from_files(main_folder):
    ''' To load the results form the files in a dict that contains all the variables
    Code snippet to reassing variables from dict results:
    aucs = results['AUC']
    false_negative_rates = results['FNR']
    false_positive_rates = results['FPR']
    detection_ratio = results['Detection_rate']
    predictions = results['Predictions']
    true_values = results['True_values']
    antibiotic_list = categories.tolist()


    '''

    
    results ={}
    # Code snippet to open json files
    with open(main_folder+'\AUCS') as json_file:
        aucs = json.load(json_file)
    results['AUC']= aucs
    
    with open(main_folder+'\FNR') as json_file:
         false_negative_rate = json.load(json_file)
    results['FNR']=false_negative_rate
        
    with open(main_folder+'\FPR') as json_file:
        false_positive_rate = json.load(json_file)
    results['FPR'] = false_positive_rate
    
    with open(main_folder+'\DetectionRate') as json_file:
        detection_rate = json.load(json_file)
    results['Detection_rate'] = detection_rate
        
        
        
    # Snippet to read pickle files
    with open(main_folder + '/predictions.pkl','rb') as f:
        predis = pickle.load(f)
    results['Predictions']=predis
    
    with open(main_folder + '/true_values.pkl','rb') as f:
        trues = pickle.load( f)
    results['True_values']=trues
    
    with open(main_folder + '/Feature_importances.pkl','rb') as f:
        feat_importance = pickle.load( f)
    results['Feature_importances']=feat_importance
        
    return  results 



# Evaluation of the model

# metrics

def false_positive_rate_one_cat(Y_test, predictions):
    false_pos = np.sum(np.logical_and(Y_test==0, predictions==1),axis =0)
    if np.sum(np.sum(Y_test==0, axis=0))==0:
        print('Division by zero. There are only Resistant samples. FPR cannot be evaluated')
    else:
        false_pos_rate = false_pos/np.sum(Y_test==0, axis=0)  ##False alarm
    return false_pos_rate

def false_negative_rate_one_cat(Y_test, predictions):
    false_neg = np.sum(np.logical_and(Y_test==1, predictions==0),axis =0) 
    false_neg_rate = false_neg/np.sum(Y_test==1, axis=0) ## Missing ratio
    return false_neg_rate


def false_positive_rate(Y_test, predictions):
    false_pos = np.sum(np.logical_and(Y_test==0, predictions==1),axis =0)
    if np.sum(np.sum(Y_test==0, axis=0))==0:
        print('Division by zero. There are only Resistant samples. FPR cannot be evaluated')
    else:
        false_pos_rate = false_pos/np.sum(Y_test==0, axis=0)  ##False alarm
    return false_pos_rate

def false_negative_rate(Y_test, predictions):
    false_neg = np.sum(np.logical_and(Y_test==1, predictions==0),axis =0) 
    false_neg_rate = false_neg/np.sum(Y_test==1, axis=0)*100 ## Missing ratio
    return false_neg_rate


def evaluate(Y_test, preds_cv, categories):

    from sklearn.metrics import roc_auc_score, auc, accuracy_score, classification_report, confusion_matrix

    preds =preds_cv>0.5

    total_samples = preds_cv.shape[0]

    detected = np.sum(preds, axis=0)

    detection_perc = detected/total_samples*100;

    false_pos = np.sum(np.logical_and(Y_test==0, preds==1),axis =0)
    false_pos_rate = false_pos/np.sum(Y_test==0, axis=0)*100  ##False alarm

    false_neg = np.sum(np.logical_and(Y_test==1, preds==0),axis =0) 
    false_neg_rate = false_neg/np.sum(Y_test==1, axis=0)*100 ## Missing ratio

    Detection_ratio = 100-false_neg_rate

    accuracy_cat = []
    roc_auc_cat = []    
    for cat in range(preds_cv.shape[1]):
        y_true = Y_test[:,cat]
        prediction_binary = preds[:,cat]
        prediction_prob = preds_cv[:,cat]

        acc = accuracy_score(y_true, prediction_binary)
        accuracy_cat.append(acc)

        roc_auc = roc_auc_score(y_true,prediction_prob, average='micro')
        roc_auc_cat.append(roc_auc)
        print('--------------------------------------------------')
        print(categories[cat])
        #print(confusion_matrix(y_true, prediction_binary, labels=[1,0]))
        print('Accuracy =',round(acc*100,2))
        print('AUC =',round(roc_auc*100,2))
        print('Ratio de Detección =',round(Detection_ratio[cat],2))
        print('Ratio de Falsa Alarma =', round(false_pos_rate[cat],2))
        print('Ratio de pérdidas = ', round(false_neg_rate[cat],2))
    #print(classification_report(Y_test,preds, target_names= categories, zero_division =0))  
        
        #plots
    bar_plot(np.array(roc_auc_cat)*100, 'AUC')
    bar_plot(np.array(accuracy_cat)*100, 'Precisión')
    bar_plot(Detection_ratio, 'Ratio de Detección')
    bar_plot(false_pos_rate, 'Falsa Alarma')
    bar_plot(false_neg_rate, 'Ratio de pérdidas')


## Graphs
def bar_plot(ratio, name, categories):
        
    ax =plt.figure(figsize=(12,6))
    plt.bar(categories,ratio)
    plt.xticks(fontsize=10, rotation=30)
    plt.yticks(np.arange(0, 100, step=5), fontsize=9)
    plt.ylabel(name)
    plt.title('Comparación por antibiótico de '+ name, fontsize=16)

    labels = [str(round(ratio[i],2))+'%' for i in range(len(ratio))]

    for index,data in enumerate(ratio):
        plt.text(x=index-0.4 , y =data+1 , s=str(round(data,2))+'%' , fontdict=dict(fontsize=10))
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
    
    ax =plt.figure(figsize=(15,6))
    plt.bar(categories,ratio, color=col)
    plt.xticks(fontsize=10, rotation=30)
    plt.yticks(np.arange(0, 100, step=5), fontsize=9)
    plt.ylabel(name)
    plt.xlabel('Antibióticos')
    plt.title('Comparación por antibiótico de '+ name, fontsize=16)

    labels = [str(round(ratio[i],2))+'%' for i in range(len(ratio))]

    for index,data in enumerate(ratio):
        plt.text(x=index-0.4 , y =data+1 , s=str(round(data,2))+'%' , fontdict=dict(fontsize=10))
    plt.show()

def create_box_plot_parameter_per_antibiotic(aucs, name_parameter):
    aucs_box_plot =tf.ordered_dict_per_antibiotic(aucs)
    #list(aucs_box_plot.keys())

    fig = plt.figure(figsize =(10, 4)) 

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
    plt.title('BoxPlot de '+name_parameter)
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
    for key,value in dict.items():
        means.append(np.nanmean(value)*100)
    return means

def compute_std_on_ordered_dict(dict):
    std = []
    for key,value in dict.items():
        std.append(np.nanstd(value)*100)
    return std



def mean_per_fold(list_of_dicts):
    means = [np.mean(list(d.values())) for d in list_of_dicts]
    return means



###### K-FOLD
### Visualization
def plot_cv_indices(cv, X, y, ax, n_splits,categories, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    add_space=0
    cmap_cv='Set3'
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
        add_space=add_space+1
        ax.scatter(range(len(X)), [ii + add_space +.5] * len(X),
                   c=y[:,i], marker='_', lw=lw, cmap=cmap_data)
        

    #ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               #c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + categories
    ax.set(yticks=np.arange(n_splits+len(categories)) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration", xlim=[0, X.shape[0]])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax


def plot_curve_auc_roc_in_CV(trues, preds, antibiotic_list):
    color_palette = ['teal','darkolivegreen', 'cadetblue', 'indianred', 'purple']#['lemonchiffon','gold', 'olive', 'goldenroad', 'darkkhaki']
    #plt.figure()
    subplot_index = -1
    col=0
    fig,ax = plt.subplots(9,2, figsize=(20,65))
    for antibiotic in antibiotic_list:
        subplot_index = subplot_index +1
        
        if subplot_index<9:
            row= subplot_index
            col=0
        else:
            col=1
            row= subplot_index-9
            
        axis = ax[row, col]
        axis.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for fold in range(len(trues)):
            fpr, tpr, threshold = roc_curve(trues[fold][antibiotic], preds[fold][antibiotic])
            roc_auc = auc(fpr, tpr)
            label = 'ROC Fold '+ str(fold) +' = ' + str(round(roc_auc,2))
            axis.plot(fpr, tpr, color_palette[fold], label = label, alpha=0.3)
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

        axis.set_title('ROC '+ antibiotic)

        axis.legend(loc = 'lower right')
        axis.plot([0, 1], [0, 1],'r--')
        axis.set_xlim([0, 1])
        axis.set_ylim([0, 1])
        axis.set_ylabel('True Positive Rate')
        axis.set_xlabel('False Positive Rate')
        axis.grid(alpha=0.4)


    ax[8,1].axis('off')
    #plt.tight_layout()
    plt.show()


    ############
    ############  Para Comparative

def retrieve_general_results(path):
    # Get the name and path of each file stored in the general directory /Models
    #Only recognizes those folders that end with _model!!!!!!
    list_subfolders_with_paths = [(f.path, f.name) for f in os.scandir(path) if f.is_dir()]
    # Select those files that correspond to models
    list_of_models = [(path,x) for path,x in list_subfolders_with_paths if x.endswith('_model')]
    models_names = [name for (path, name) in list_of_models]

    # Save the scores in a dictionary and the names in a list
    results = {}
    models_names=[]
    for path, model in list_of_models:
        results[model]= tf.load_results_from_files(path)
        #models_names.append(model)
    return results
    
def extract_parameter(results, parameter_name):
    # Organize those results from separated by folds to separated by antibiotic per each model
    ## First, create a nested dictionaty, with the model name as fist key, and antibiotic name as second key.
    new_results={}
    for my_model, general_dict in results.items():
        for parameter, scores_per_antibiotic_dict in general_dict.items():
            if parameter in [parameter_name]:
                new_dict={}
                for antibiotic, score in tf.ordered_dict_per_antibiotic(scores_per_antibiotic_dict).items():
                    new_dict[antibiotic] = np.mean(score)
                new_results[my_model] = new_dict
    new_results 

    ## Reorder in a new dict in which each antibiotic(key) has associated a list of scores (values) that corresponds to the models stores in _models_, with the same order
    models=[]
    antib_dicts=[]
    for my_model,general_dict in new_results.items():
        #print(my_model)
        models.append(my_model)
        antib_dicts.append(general_dict)
    final_dict =tf.ordered_dict_per_antibiotic(antib_dicts)
    return final_dict, models

def bar_plot_compare_models_per_antibiotic_one_parameter(final_dict, models, parameter_name, threshold, better_low=False):
	##
    for key, scores in final_dict.items():

        plt.figure(figsize=(12,4))
        perc_scores = [score*100 for score in scores]

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

        plt.bar(models,perc_scores, color =col)
        plt.ylim([0,100])
        plt.xticks(fontsize=10, rotation=75)
        plt.yticks(np.arange(50, 100, step=5), fontsize=9)
        plt.ylabel(parameter_name)
        plt.title(key)

        #labels = [str(round(perc_score[i],2))+'%' for i in range(len(perc_scores))]

        for index,data in enumerate(perc_scores):
            plt.text(x=index-0.2 , y =data-5 , s=str(round(data,2))+'%' , fontdict=dict(fontsize=10))
        plt.show()

def bar_plot_compare_models_per_antibiotic_all_parameters(path, threshold):
    #path = 'C:/Users/aida_/Desktop/TFM/Algoritmo/Models'
    results = tf.retrieve_general_results(path)
    final_dict_auc, models = tf.extract_parameter(results, 'AUC')
    final_dict_fnr, models = tf.extract_parameter(results, 'FNR')
    final_dict_fpr, models = tf.extract_parameter(results, 'FPR')
    final_dict_detection, models = tf.extract_parameter(results, 'Detection_rate')

    width =0.4
    color_pallete_auc = ['lightcoral', 'royalblue', 'mediumseagreen']
    color_pallete_det = ['firebrick', 'darkblue', 'seagreen']

    for key, scores in final_dict_auc.items():
        fig,ax =plt.subplots(figsize=(15,5))
        perc_auc_scores = [score*100 for score in scores]
        perc_det_scores = [score*100 for score in final_dict_detection[key]]
        perc_fnr_scores = [score*100 for score in final_dict_fnr[key]]
        perc_fpr_scores = [score*100 for score in final_dict_fpr[key]]


        col_auc = []
        for val in perc_auc_scores:
            if val < threshold[0]:
                col_auc.append(color_pallete_auc[0])
            elif val > threshold[1]:
                col_auc.append(color_pallete_auc[2])
            else:
                col_auc.append(color_pallete_auc[1])    

        col_det =[]
        for val in perc_det_scores:
            if val < threshold[0]:
                col_det.append(color_pallete_det[0])
            elif val > threshold[1]:
                col_det.append(color_pallete_det[2])
            else:
                col_det.append(color_pallete_det[1])            


        b1 = ax.bar(models,perc_auc_scores,width, color =col_auc, label ='AUC', alpha=0.5)
        b2 = ax.bar(np.arange(len(models))+width,perc_det_scores,width, color=col_det, alpha =0.5, label='Detection ratio')
        plt.plot(np.arange(len(models))+width/2,perc_fnr_scores, marker='*', markersize= 10,color ='g',  label='FNR')
        plt.plot(np.arange(len(models))+width/2,perc_fpr_scores, marker='o', markersize=10, color='purple', label='FPR')


        plt.ylim([0,104])
        plt.xlim([-0.5,len(models)+1])
        plt.xticks(fontsize=10, rotation=75)
        plt.yticks(np.arange(0, 104, step=5), fontsize=9)
        plt.ylabel('Percentage')
        plt.title(key)
        plt.grid()

        #labels = [str(round(perc_score[i],2))+'%' for i in range(len(perc_scores))]

        for index,data in enumerate(perc_auc_scores):
            plt.text(x=index-0.2 , y =data-5 , s=str(round(data,2))+'%' , fontdict=dict(fontsize=10))

        for index,data in enumerate(perc_det_scores):
            plt.text(x=index+width-0.15 , y =data-5 , s=str(round(data,2))+'%' , fontdict=dict(fontsize=10))

        plt.legend()
        plt.show()


def compare_preprocess(ratio_model_1, ratio_model_2, name, categories, model_name):
    fig, ax =plt.subplots(figsize=(15,6))
    width = 0.4
    b1 =ax.bar(categories,ratio_model_1, width,  color='royalblue', label = model_name[0])

    b2 = ax.bar(np.arange(len(categories))+width+0.02 ,ratio_model_2, width, color='mediumseagreen', label = model_name[1])

    plt.xticks(fontsize=10, rotation=30)
    plt.yticks(np.arange(0, 100, step=5), fontsize=9)
    plt.ylabel(name)
    plt.xlabel('Antibióticos')
    plt.title('Comparación por antibiótico de '+ name, fontsize=16)


    for index,data in enumerate(ratio_model_1):
        plt.text(x=index-0.4 , y =data+1 , s=str(round(data,2))+'%' , fontdict=dict(fontsize=10))

    for index,data in enumerate(ratio_model_2):
        plt.text(x=index-0.4 , y =data+1 , s=str(round(data,2))+'%' , fontdict=dict(fontsize=10))

    plt.legend()
    plt.show()