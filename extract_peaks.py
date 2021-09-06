import numpy as np
import pandas as pd
import os
from pyteomics import mzml
from matplotlib import pyplot as plt


######################## READ AND PROCESS DATA ############################
# PATH TO MZML FILES
hgm_rep_mzml_path = './data/Reproducibilidad/mzml'
# Path to peak excel
excel_path = './data/peaks.xlsx'
# Path to excel with HGM ids samples
ids_excel = "./data/Reproducibilidad/Klebsiellas_Estudio_Reproducibilidad_rev.xlsx"


# READ DATA FROM FOLDS
listOfFiles = list()
for (dirpath, dirnames, filenames_rep) in os.walk(hgm_rep_mzml_path):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames_rep]

id_samples_rep = []
maldis = []
# CONVERT TO A PANDAS DATAFRAME
for filepath in listOfFiles:
    file = filepath.split("/")[-1]
    if file == ".DS_Store" or file.split('_')[2] == '1988':
        continue
    print(file)
    t = mzml.read(filepath)
    a = next(t)
    maldis.append(a["intensity array"][2000:12000])
    id_samples_rep.append(file.split('_')[2])

gm_data = pd.DataFrame(data=np.empty((len(maldis), 2)), columns=["Nº Espectro", "maldi"])
gm_data["maldi"] = maldis
gm_data["Nº Espectro"] = id_samples_rep
gm_x = gm_data.set_index("Nº Espectro")

# AS EVERY BACTERIA HAS MORE THAN ONE MALDI MS WE SELECT THE MORE SIMILAR TO THE MEDIAN
hgm_median = pd.DataFrame(data=np.vstack(maldis))
hgm_median['id'] = id_samples_rep
hgm_median = hgm_median.set_index('id')
median_sample = hgm_median.groupby('id').median()
gm_median_1s = pd.DataFrame(data=np.empty((len(median_sample), 2)), columns=["Nº Espectro", "maldi"])
gm_median_1s['Nº Espectro'] = median_sample.index
gm_random_1s = gm_median_1s.set_index('Nº Espectro')
data_median = []
for s in median_sample.index:
    print(s)
    if hgm_median.loc[s].shape[0] == 10000:
        data_median.append(hgm_median.loc[s].values)
    else:
        data_median.append(hgm_median.loc[s].iloc[(median_sample.loc[s]-hgm_median.loc[s]).mean(axis=1).abs().argmin(), :].values)
gm_median_1s['maldi'] = data_median
maldis_med = np.vstack(gm_median_1s['maldi'])

################# ROIS OF INTEREST ###################
peak1a = [2200, 2350]
peak1b = [2350, 2500]
peak2a = [7150, 7300]
peak2b = [7350, 7500]
peak3 = [9800, 10000]
peaks = [peak1a, peak1b, peak2a, peak2b, peak3]
peakinfo = np.zeros((maldis_med.shape[0],len(peaks)*2))

j=0
for peak in peaks:
    roi = maldis_med[:, peak[0]-2000:peak[1]-2000]
    peak_intensity = np.max(roi, axis=1)
    peak_pos = np.argmax(roi, axis=1)+peak[0]
    peakinfo[:,j]=peak_pos
    peakinfo[:, j+1]=peak_intensity
    j+=2

    ## Check if the peaks are correct
    plt.figure()
    plt.plot(range(peak[0], peak[1]),maldis_med[0, peak[0]-2000:peak[1]-2000], label="Maldi sample")
    plt.plot(peak_pos[0], peak_intensity[0], 'r*')

peakpd = pd.DataFrame(data=np.empty((len(median_sample), 11)), columns=["Nº Espectro", "pos_peak_1", "int_peak_1", "pos_peak_2", "int_peak_2", "pos_peak_3", "int_peak_3", "pos_peak_4", "int_peak_4", "pos_peak_5", "int_peak_5"])
peakpd['Nº Espectro'] = gm_median_1s['Nº Espectro']
peakpd.iloc[:, 1:] = np.vstack(peakinfo)
peakpd = peakpd.set_index("Nº Espectro")

hgm_excel = pd.read_excel(ids_excel, engine='openpyxl', dtype={'Número de muestra': str})
pd.merge(how='outer', left=hgm_excel, right=peakpd, left_on='Nº Espectro',
        right_on='Nº Espectro')


# hgm_peaks_excel = peaks_excel[peaks_excel['Centro']=='HGM']

# pd.merge(how='inner', left=hgm_data['Nº Micro'], 
#         right=hgm_peaks_excel, 
#         left_on='Nº Micro',
#         right_on='Número de muestra')


######################################## HRC


# PATH TO MZML FILES
ryc_path = "./data/Klebsiellas_RyC/"
# Columns to read from excel data
cols = ['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL', 'AMOXI/CLAV .1', 'PIP/TAZO.1', 'CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1', 'AZTREONAM.1', 'IMIPENEM.1', 'MEROPENEM.1', 'ERTAPENEM.1']
# BOOLEAN TO NORMALIZE BY TIC
tic_norm=True

######################## READ AND PROCESS DATA ############################
# LOAD RYC MALDI-TOF
listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk(ryc_path):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]
# ============= READ FEN/GEN/AB INFO ============
full_data = pd.read_excel("./data/DB_conjunta.xlsx", engine='openpyxl')

# READ DATA FROM FOLDS
data_int = []
id = []
letter = ["A", "B", "BTS", "C", "D", "E", "F", "G", "H"]
# CONVERT TO A PANDAS DATAFRAME
for file in listOfFiles:
    print(file)
    t = mzml.read(file)
    a = next(t)
    data_int.append(a["intensity array"][2000:12000])
    filename = file.split("/")[4]
    erase_end = filename.split(".")[0]
    if erase_end.split("_")[0] in letter:
        id.append(erase_end.split("_")[0] + erase_end.split("_")[1])
    else:
        id.append(erase_end.split("_")[0] + "-" + erase_end.split("_")[1])

ryc_data = pd.DataFrame(data=np.empty((len(data_int), 2)), columns=["Número de muestra", "maldi"])
ryc_data["maldi"] = data_int
ryc_data["Número de muestra"] = id

# AS EVERY BACTERIA HAS MORE THAN ONE MALDI MS WE SELECT THE MORE SIMILAR TO THE MEDIAN
ryc_median = pd.DataFrame(data=np.vstack(data_int))
ryc_median['id'] = id
ryc_median = ryc_median.set_index('id')
median_sample = ryc_median.groupby('id').median()

ryc_data_1s = pd.DataFrame(data=np.empty((len(median_sample), 2)), columns=["Número de muestra", "maldi"])
ryc_data_1s['Número de muestra'] = median_sample.index
ryc_data_1s = ryc_data_1s.set_index('Número de muestra')
data_median = []
for s in median_sample.index:
    print(s)
    if ryc_median.loc[s].shape[0] == 10000:
        data_median.append(ryc_median.loc[s].values)
    else:
        data_median.append(ryc_median.loc[s].iloc[(median_sample.loc[s]-ryc_median.loc[s]).mean(axis=1).abs().argmin(), :].values)
ryc_data_1s['maldi'] = data_median

maldis_med = np.vstack(ryc_data_1s['maldi'].values)

################# ROIS OF INTEREST ###################
peak1a = [2200, 2350]
peak1b = [2350, 2500]
peak2a = [7150, 7300]
peak2b = [7350, 7500]
peak3 = [9800, 10000]
peaks = [peak1a, peak1b, peak2a, peak2b, peak3]
peakinfo = np.zeros((maldis_med.shape[0],len(peaks)*2))

j=0
for peak in peaks:
    roi = maldis_med[:, peak[0]-2000:peak[1]-2000]
    peak_intensity = np.max(roi, axis=1)
    peak_pos = np.argmax(roi, axis=1)+peak[0]
    peakinfo[:,j]=peak_pos
    peakinfo[:, j+1]=peak_intensity
    j+=2

    ## Check if the peaks are correct
    plt.figure()
    plt.plot(range(peak[0], peak[1]),maldis_med[0, peak[0]-2000:peak[1]-2000], label="Maldi sample")
    plt.plot(peak_pos[0], peak_intensity[0], 'r*')


peakpd = pd.DataFrame(data=np.empty((len(maldis_med), 11)), columns=["Nº Espectro", "pos_peak_1", "int_peak_1", "pos_peak_2", "int_peak_2", "pos_peak_3", "int_peak_3", "pos_peak_4", "int_peak_4", "pos_peak_5", "int_peak_5"])
peakpd['Nº Espectro'] = ryc_data_1s.index
peakpd.iloc[:, 1:] = np.vstack(peakinfo)
peakpd = peakpd.set_index("Nº Espectro")

peakpd.to_excel('./data/rycpeaks.xlsx')


