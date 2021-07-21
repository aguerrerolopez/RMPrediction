import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, r2_score, mean_squared_error
import pandas as pd

def sigmoid(x):
  return np.exp(-np.log(1 + np.exp(-x)))

def plot_W(W, title):
    plt.figure()
    plt.imshow((np.abs(W)), aspect=W.shape[1] / W.shape[0])
    plt.colorbar()
    plt.title(title)
    plt.ylabel('features')
    plt.xlabel('K')
    plt.show()


resultados = []



modelo_a_cargar= "./Results/TrainHGM_predictRYC/TrainHGM_PredRyC_linear_gen4_prun0.1.pkl"


with open("./data/ryc_data_mediansample_only2-12_TIC.pkl", 'rb') as pkl:
    ryc_data = pickle.load(pkl)

with open("./data/hgm_data_mediansample_only2-12_TIC.pkl", 'rb') as pkl:
    hgm_data = pickle.load(pkl)

folds_path = "./data/HGM_10STRATIFIEDfolds_muestrascompensada_pruebaloca.pkl"
with open(folds_path, 'rb') as pkl:
    folds = pickle.load(pkl)

with open(modelo_a_cargar, 'rb') as pkl:
    results = pickle.load(pkl)


old_fen = hgm_data['fen']
old_fen = old_fen.drop(old_fen[old_fen['Fenotipo CP']==1].index)
hgm_data['fen'] = old_fen[['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]

hgm_data['maldi'] = hgm_data['maldi'].loc[hgm_data['fen'].index]
hgm_data['cmi'] = hgm_data['cmi'].loc[hgm_data['fen'].index]
hgm_data['binary_ab'] = hgm_data['binary_ab'].loc[hgm_data['fen'].index]

old_fen = ryc_data['fen']
old_fen = old_fen.drop(old_fen[old_fen['Fenotipo CP']==1].index)
ryc_data['fen'] = old_fen[['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]

ab_cols =  ['AMOXI/CLAV .1', 'PIP/TAZO.1', 'CEFTAZIDIMA.1', 'CEFOTAXIMA.1', 'CEFEPIME.1', 'AZTREONAM.1', 'IMIPENEM.1', 'MEROPENEM.1', 'ERTAPENEM.1']

full_predict = pd.concat([ryc_data['fen'], ryc_data['binary_ab'][ab_cols].loc[ryc_data['fen'].index]], axis=1)
full_predict = full_predict.dropna()

ryc_data['maldi'] = ryc_data['maldi'].loc[full_predict.index]
ryc_data['binary_ab'] = full_predict[ab_cols]
ryc_data['fen'] = full_predict[['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]


f=0
model_name = "model_fold"+str(f)

from mpl_toolkits.axes_grid1 import make_axes_locatable
k_len = [len(results[model_name].q_dist.alpha[1]['b']) for f in range(10)]
k_ords = np.zeros((10, np.max(k_len)))
# CAMBIAR ALPHA 2 SI HAY VISTA FENOTIPO, CAMBIAR ALPHA1 SI N LA HAY
ord_k = np.argsort(results[model_name].q_dist.alpha[1]['a']/results[model_name].q_dist.alpha[1]['b'])
k_ords[f, :len(ord_k)] = ord_k
matrix_views = np.zeros((len(results[model_name].q_dist.W),results[model_name].q_dist.W[0]['mean'].shape[1]))
# matrix_views = np.zer.os((20,results[model_name].q_dist.W[0]['mean'].shape[1]))
n_cols = 13
matrix_views = np.zeros((n_cols,results[model_name].q_dist.W[0]['mean'].shape[1]))
for z in range(matrix_views.shape[0]):
    if z==0:
        X=results[model_name].X[0]['X']
        W=X.T@results[model_name].q_dist.W[0]['mean']
        matrix_views[z, :]=np.mean(np.abs(W), axis=0)[ord_k]
    elif z>0 and z<4:
        matrix_views[z, :]=np.abs(results[model_name].q_dist.W[2]['mean'][z-1,:][ord_k])
    else:
        print(z-2)
        # matrix_views[z, :]=np.mean(np.abs(results[model_name].q_dist.W[z-3]['mean']), axis=0)[ord_k]
        matrix_views[z, :]=np.mean(np.abs(results[model_name].q_dist.W[z-2+1]['mean']), axis=0)[ord_k]

from matplotlib import cm
cmap = cm.get_cmap('Dark2', 9)
views = ['MALDI', 'CARB and ESBL', 'ESBL', 'Susceptible','AMOXI/CLAV ', 'PIP/TAZO', 'CEFTAZIDIMA', 'CEFOTAXIMA', 'CEFEPIME', 'AZTREONAM', 'IMIPENEM', 'MEROPENEM', 'ERTAPENEM']

for v, view in enumerate(views):
    # if v>3: break
    if v==0: delay=9
    if v==1: delay=0
    if v==5: delay=1
    if v==7: delay=2
    if v==10: delay=3
    if v==11: delay=4
    if v==14: delay=5
    if v==15: delay=6
    if v==18: delay=7
    if v==19: delay=8
    plt.figure(figsize=[10,5])
    ax = plt.gca()
    ax.set_title(view+" view", color=cmap(delay))
    # plt.title(view+" view")
    plt.yticks([], [])
    plt.xticks(range(0, len(ord_k)), ord_k.tolist())
    plt.xlabel("K features")
    im = ax.imshow(matrix_views[v,:][:, np.newaxis].T, cmap="binary")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    spath = "./Results/plots_reunion2406/"+view.replace(" ", "").replace("/","-")+"_19v_lfanalysis.png"
    plt.savefig(spath)
    plt.show()



X=results[model_name].X[0]['X']
W_maldi=X.T@results[model_name].q_dist.W[0]['mean']
lf = [11, 13, 4]


########### Análisis datos
#CP+ESBL
hgm_mald_res = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo CP+ESBL']==1)].values)
hgm_mald_sen = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo CP+ESBL']==0)].values)
ryc_mald_res = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo CP+ESBL']==1)].values)
ryc_mald_sen = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo CP+ESBL']==0)].values)
plt.figure(figsize=[15,10])
plt.plot(np.arange(2000,12000),np.mean(hgm_mald_res, axis=0), alpha=0.8, label="HGM Resistant")
plt.plot(np.arange(2000,12000),np.mean(ryc_mald_res, axis=0), '--',alpha=0.5, label="RyC Resistant")
plt.legend()
plt.title("Resistant CP and ESBL ")
plt.show()

plt.figure(figsize=[15,10])
plt.plot(np.arange(2000,12000),np.mean(hgm_mald_sen, axis=0), alpha=0.8, label="HGM Sensible")
plt.plot(np.arange(2000,12000),np.mean(ryc_mald_sen, axis=0), '--', alpha=0.5, label="RyC Sensible")
plt.legend()
plt.title("SEnsible CP+ESBL")
plt.show()

#ESBL
hgm_mald_res = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo  ESBL']==1)].values)
hgm_mald_sen = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo  ESBL']==0)].values)
ryc_mald_res = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo  ESBL']==1)].values)
ryc_mald_sen = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo  ESBL']==0)].values)
plt.figure(figsize=[15,10])
plt.plot(np.arange(2000,12000),np.mean(hgm_mald_res, axis=0), alpha=0.8, label="HGM Resistant")
plt.plot(np.arange(2000,12000),np.mean(ryc_mald_res, axis=0), '--',alpha=0.5, label="RyC Resistant")
plt.legend()
plt.title("Resistant ESBL")
plt.show()

plt.figure(figsize=[15,10])
plt.plot(np.arange(2000,12000),np.mean(hgm_mald_sen, axis=0), alpha=0.8, label="HGM Sensible")
plt.plot(np.arange(2000,12000),np.mean(ryc_mald_sen, axis=0), '--', alpha=0.5, label="RyC Sensible")
plt.legend()
plt.title("SEnsible ESBL")
plt.show()

#noCP+noESBL
hgm_mald_res = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo noCP noESBL']==1)].values)
hgm_mald_sen = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo noCP noESBL']==0)].values)
ryc_mald_res = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo noCP noESBL']==1)].values)
ryc_mald_sen = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo noCP noESBL']==0)].values)
plt.figure(figsize=[15,10])
plt.plot(np.arange(2000,12000),np.mean(hgm_mald_res, axis=0), alpha=0.8, label="HGM Resistant")
plt.plot(np.arange(2000,12000),np.mean(ryc_mald_res, axis=0), '--',alpha=0.5, label="RyC Resistant")
plt.legend()
plt.title("Resistant noCP noESBL")
plt.show()

plt.figure(figsize=[15,10])
plt.plot(np.arange(2000,12000),np.mean(hgm_mald_sen, axis=0), alpha=0.8, label="HGM Sensible")
plt.plot(np.arange(2000,12000),np.mean(ryc_mald_sen, axis=0), '--', alpha=0.5, label="RyC Sensible")
plt.legend()
plt.title("SEnsible noCP noESBL")
plt.show()

########### Zoom en 2300
color=['tab:red', 'tab:green', 'tab:pink', 'tab:gray']
#CP+ESBL
hgm_mald_res = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo CP+ESBL']==1)].values)
hgm_mald_sen = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo CP+ESBL']==0)].values)
ryc_mald_res = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo CP+ESBL']==1)].values)
ryc_mald_sen = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo CP+ESBL']==0)].values)
plt.figure(figsize=[15,10])
for j,l in enumerate(lf):
    W_proj = W_maldi[:, l]*results[model_name].q_dist.W[2]['mean'][0, l]
    plt.plot(np.arange(2200,2500), W_proj[200:500], color=color[j], alpha=0.8, label="Latent feature "+str(l))
plt.plot(np.arange(2200,2500),np.mean(hgm_mald_res, axis=0)[200:500], alpha=0.8, label="HGM Resistant")
plt.plot(np.arange(2200,2500),np.mean(ryc_mald_res, axis=0)[200:500], '--',alpha=0.5, label="RyC Resistant")
plt.plot(np.arange(2200,2500),np.mean(hgm_mald_sen, axis=0)[200:500], alpha=0.8, label="HGM Sensible")
plt.plot(np.arange(2200,2500),np.mean(ryc_mald_sen, axis=0)[200:500], '--', alpha=0.5, label="RyC Sensible")
plt.legend()
plt.title("ZOOM IN: CP and ESBL view")
plt.show()


#ESBL
hgm_mald_res = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo  ESBL']==1)].values)
hgm_mald_sen = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo  ESBL']==0)].values)
ryc_mald_res = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo  ESBL']==1)].values)
ryc_mald_sen = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo  ESBL']==0)].values)
plt.figure(figsize=[15,10])
for j,l in enumerate(lf):
    W_proj = W_maldi[:, l]*results[model_name].q_dist.W[2]['mean'][1, l]
    plt.plot(np.arange(2200,2500), W_proj[200:500], color=color[j], alpha=0.8, label="Latent feature "+str(l))
plt.plot(np.arange(2200,2500),np.mean(hgm_mald_res, axis=0)[200:500], alpha=0.8, label="HGM Resistant")
plt.plot(np.arange(2200,2500),np.mean(ryc_mald_res, axis=0)[200:500], '--',alpha=0.5, label="RyC Resistant")
plt.plot(np.arange(2200,2500),np.mean(hgm_mald_sen, axis=0)[200:500], alpha=0.8, label="HGM Sensible")
plt.plot(np.arange(2200,2500),np.mean(ryc_mald_sen, axis=0)[200:500], '--', alpha=0.5, label="RyC Sensible")
plt.legend()
plt.title("ZOOM IN: ESBL view")
plt.show()


#noCP+noESBL
hgm_mald_res = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo noCP noESBL']==1)].values)
hgm_mald_sen = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo noCP noESBL']==0)].values)
ryc_mald_res = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo noCP noESBL']==1)].values)
ryc_mald_sen = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo noCP noESBL']==0)].values)
plt.figure(figsize=[15,10])
for j,l in enumerate(lf):
    W_proj = W_maldi[:, l]*results[model_name].q_dist.W[2]['mean'][2, l]
    plt.plot(np.arange(2200,2500), W_proj[200:500], color=color[j], alpha=0.8, label="Latent feature "+str(l))
plt.plot(np.arange(2200,2500),np.mean(hgm_mald_res, axis=0)[200:500], alpha=0.8, label="HGM Resistant")
plt.plot(np.arange(2200,2500),np.mean(ryc_mald_res, axis=0)[200:500], '--',alpha=0.5, label="RyC Resistant")
plt.plot(np.arange(2200,2500),np.mean(hgm_mald_sen, axis=0)[200:500], alpha=0.8, label="HGM Sensible")
plt.plot(np.arange(2200,2500),np.mean(ryc_mald_sen, axis=0)[200:500], '--', alpha=0.5, label="RyC Sensible")
plt.legend()
plt.title("noCP noESBL")
plt.show()


y_true = ryc_data['binary_ab']
ph_true = ryc_data['fen']


########### Zoom en 7400
#CP+ESBL
hgm_mald_res = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo CP+ESBL']==1)].values)
hgm_mald_sen = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo CP+ESBL']==0)].values)
ryc_mald_res = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo CP+ESBL']==1)].values)
ryc_mald_sen = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo CP+ESBL']==0)].values)
plt.figure(figsize=[15,10])
for j,l in enumerate(lf):
    W_proj = W_maldi[:, l]*results[model_name].q_dist.W[2]['mean'][0, l]
    plt.plot(np.arange(7350,7500), W_proj[5350:5500], color=color[j], alpha=0.8, label="Latent feature "+str(l))
plt.plot(np.arange(7350,7500),np.mean(hgm_mald_res, axis=0)[5350:5500], alpha=0.8, label="HGM Resistant")
plt.plot(np.arange(7350,7500),np.mean(ryc_mald_res, axis=0)[5350:5500], '--',alpha=0.5, label="RyC Resistant")
plt.plot(np.arange(7350,7500),np.mean(hgm_mald_sen, axis=0)[5350:5500], alpha=0.8, label="HGM Sensible")
plt.plot(np.arange(7350,7500),np.mean(ryc_mald_sen, axis=0)[5350:5500], '--', alpha=0.5, label="RyC Sensible")
plt.legend()
plt.title("ZOOM IN: CP and ESBL view")
plt.show()


#ESBL
hgm_mald_res = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo  ESBL']==1)].values)
hgm_mald_sen = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo  ESBL']==0)].values)
ryc_mald_res = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo  ESBL']==1)].values)
ryc_mald_sen = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo  ESBL']==0)].values)
plt.figure(figsize=[15,10])
for j,l in enumerate(lf):
    W_proj = W_maldi[:, l]*results[model_name].q_dist.W[2]['mean'][1, l]
    plt.plot(np.arange(7350,7500), W_proj[5350:5500], color=color[j], alpha=0.8, label="Latent feature "+str(l))

plt.plot(np.arange(7350,7500),np.mean(hgm_mald_res, axis=0)[5350:5500], alpha=0.8, label="HGM Resistant")
plt.plot(np.arange(7350,7500),np.mean(ryc_mald_res, axis=0)[5350:5500], '--',alpha=0.5, label="RyC Resistant")
plt.plot(np.arange(7350,7500),np.mean(hgm_mald_sen, axis=0)[5350:5500], alpha=0.8, label="HGM Sensible")
plt.plot(np.arange(7350,7500),np.mean(ryc_mald_sen, axis=0)[5350:5500], '--', alpha=0.5, label="RyC Sensible")
plt.legend()
plt.title("ZOOM IN: ESBL view")
plt.show()


#noCP+noESBL
hgm_mald_res = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo noCP noESBL']==1)].values)
hgm_mald_sen = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo noCP noESBL']==0)].values)
ryc_mald_res = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo noCP noESBL']==1)].values)
ryc_mald_sen = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo noCP noESBL']==0)].values)
plt.figure(figsize=[15,10])
for j,l in enumerate(lf):
    W_proj = W_maldi[:, l]*results[model_name].q_dist.W[2]['mean'][2, l]
    plt.plot(np.arange(7350,7500), W_proj[5350:5500], color=color[j], alpha=0.8, label="Latent feature "+str(l))
plt.plot(np.arange(7350,7500),np.mean(hgm_mald_res, axis=0)[5350:5500], alpha=0.8, label="HGM Resistant")
plt.plot(np.arange(7350,7500),np.mean(ryc_mald_res, axis=0)[5350:5500], '--',alpha=0.5, label="RyC Resistant")
plt.plot(np.arange(7350,7500),np.mean(hgm_mald_sen, axis=0)[5350:5500], alpha=0.8, label="HGM Sensible")
plt.plot(np.arange(7350,7500),np.mean(ryc_mald_sen, axis=0)[5350:5500], '--', alpha=0.5, label="RyC Sensible")
plt.legend()
plt.title("noCP noESBL")
plt.show()
y_true = ryc_data['binary_ab']
ph_true = ryc_data['fen']


########### Zoom en 9940
#CP+ESBL
hgm_mald_res = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo CP+ESBL']==1)].values)
hgm_mald_sen = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo CP+ESBL']==0)].values)
ryc_mald_res = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo CP+ESBL']==1)].values)
ryc_mald_sen = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo CP+ESBL']==0)].values)
plt.figure(figsize=[15,10])
for j,l in enumerate(lf):
    W_proj = W_maldi[:, l]*results[model_name].q_dist.W[2]['mean'][2, l]
    plt.plot(np.arange(9900,10000), W_proj[7900:8000], color=color[j], alpha=0.8, label="Latent feature "+str(l))

plt.plot(np.arange(9900,10000),np.mean(hgm_mald_res, axis=0)[7900:8000], alpha=0.8, label="HGM Resistant")
plt.plot(np.arange(9900,10000),np.mean(ryc_mald_res, axis=0)[7900:8000], '--',alpha=0.5, label="RyC Resistant")
plt.plot(np.arange(9900,10000),np.mean(hgm_mald_sen, axis=0)[7900:8000], alpha=0.8, label="HGM Sensible")
plt.plot(np.arange(9900,10000),np.mean(ryc_mald_sen, axis=0)[7900:8000], '--', alpha=0.5, label="RyC Sensible")
plt.legend()
plt.title("ZOOM IN: CP and ESBL view")
plt.show()


#ESBL
hgm_mald_res = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo  ESBL']==1)].values)
hgm_mald_sen = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo  ESBL']==0)].values)
ryc_mald_res = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo  ESBL']==1)].values)
ryc_mald_sen = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo  ESBL']==0)].values)
plt.figure(figsize=[15,10])
for j,l in enumerate(lf):
    W_proj = W_maldi[:, l]*results[model_name].q_dist.W[2]['mean'][2, l]
    plt.plot(np.arange(9900,10000), W_proj[7900:8000], color=color[j], alpha=0.8, label="Latent feature "+str(l))

plt.plot(np.arange(9900,10000),np.mean(hgm_mald_res, axis=0)[7900:8000], alpha=0.8, label="HGM Resistant")
plt.plot(np.arange(9900,10000),np.mean(ryc_mald_res, axis=0)[7900:8000], '--',alpha=0.5, label="RyC Resistant")
plt.plot(np.arange(9900,10000),np.mean(hgm_mald_sen, axis=0)[7900:8000], alpha=0.8, label="HGM Sensible")
plt.plot(np.arange(9900,10000),np.mean(ryc_mald_sen, axis=0)[7900:8000], '--', alpha=0.5, label="RyC Sensible")
plt.legend()
plt.title("ZOOM IN: ESBL view")
plt.show()


#noCP+noESBL
hgm_mald_res = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo noCP noESBL']==1)].values)
hgm_mald_sen = np.vstack(hgm_data['maldi'][(hgm_data['fen']['Fenotipo noCP noESBL']==0)].values)
ryc_mald_res = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo noCP noESBL']==1)].values)
ryc_mald_sen = np.vstack(ryc_data['maldi'][(ryc_data['fen']['Fenotipo noCP noESBL']==0)].values)
plt.figure(figsize=[15,10])
for j,l in enumerate(lf):
    W_proj = W_maldi[:, l]*results[model_name].q_dist.W[2]['mean'][2, l]
    plt.plot(np.arange(9900,10000), W_proj[7900:8000], color=color[j], alpha=0.8, label="Latent feature "+str(l))

plt.plot(np.arange(9900,10000),np.mean(hgm_mald_res, axis=0)[7900:8000], alpha=0.8, label="HGM Resistant")
plt.plot(np.arange(9900,10000),np.mean(ryc_mald_res, axis=0)[7900:8000], '--',alpha=0.5, label="RyC Resistant")
plt.plot(np.arange(9900,10000),np.mean(hgm_mald_sen, axis=0)[7900:8000], alpha=0.8, label="HGM Sensible")
plt.plot(np.arange(9900,10000),np.mean(ryc_mald_sen, axis=0)[7900:8000], '--', alpha=0.5, label="RyC Sensible")
plt.legend()
plt.title("noCP noESBL")
plt.show()


from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(ryc_data['maldi'].index.tolist(), test_size=0.5, random_state=0)

y_true = ryc_data['binary_ab']#.loc[test_idx]
ph_true = ryc_data['fen']#.loc[test_idx]



r = np.zeros((1,9))
f=0
model = "model_fold"+str(f)
for h in range(r.shape[1]):
    y_pred = results[model].t[3+h]['mean'][-y_true.shape[0]:, 0]
    r[0, h] = roc_auc_score(y_true.values[:, h], y_pred)
print("ANTIBIOTIC")
print(r)
print(np.mean(r))


ph_pred = results[model].t[2]['mean'][-ph_true.shape[0]:, :]
ph = np.zeros((1,3))
for h in range(ph.shape[1]):
    ph[0, h] = roc_auc_score(ph_true.values[:, h], ph_pred[:, h])
print("Phenotype")
print(ph)

# r = np.zeros((3,15))
# y_tst = y_true.to_numpy()
# # for f in range(3):

# y_pred = results[model].t[2]["mean"][-y_tst.shape[0]:, :]
# for h in range(15):
#     if h>8:
#         break
#     y_tst_complete = y_tst[:, h][~np.isnan(y_tst[:, h])]
#     y_pred = results[model].t[2+h]['mean'][-y_tst_complete.shape[0]:, 0]
#     r[f, h] = roc_auc_score(y_tst_complete, y_pred)    
# print(np.mean(r,axis=0))

# y_tst = ph_true.to_numpy()
# ph = np.zeros((1,4))
# # for f in range(3):
# f=0
# model = "model_fold"+str(f)
# for h in range(3):
#     y_tst_complete = y_tst[:, h][~np.isnan(y_tst[:, h])]
#     y_pred = results[model].t[1]['mean'][-y_tst_complete.shape[0]:, h]
#     ph[f, h] = roc_auc_score(y_tst_complete, y_pred) 
# # print(np.mean(ph,axis=0))
# print(ph)



# # RESULTADOS PARA UN FOLD ESPECÍFICO
# f=0
# model_name = "model_fold"+str(f)

# # TODO: Sacar vistas comunes

# from mpl_toolkits.axes_grid1 import make_axes_locatable
# k_len = [len(results[model_name].q_dist.alpha[1]['b']) for f in range(10)]
# k_ords = np.zeros((10, np.max(k_len)))
# f=0
# # CAMBIAR ALPHA 2 SI HAY VISTA FENOTIPO, CAMBIAR ALPHA1 SI N LA HAY
# ord_k = np.argsort(results[model_name].q_dist.alpha[2]['a']/results[model_name].q_dist.alpha[2]['b'])
# k_ords[f, :len(ord_k)] = ord_k
# matrix_views = np.zeros((len(results[model_name].q_dist.W),results[model_name].q_dist.W[0]['mean'].shape[1]))
# for z in range(len(results[model_name].q_dist.W)):
#     matrix_views[z, :]=np.mean(np.abs(results[model_name].q_dist.W[z]['mean']), axis=0)[ord_k]

# for v in range(len(results[model_name].q_dist.W)):
#     plt.figure(figsize=[20,10])
#     if v==0: tick="LATENT SPACE DISTRIBUTION OVER VIEWS"
#     if v==1: tick="Phenotypic"
#     if v==2: tick="AMR"

#     ax = plt.gca()
#     plt.title(tick+" view")
#     plt.yticks([], [])
#     plt.xticks(range(0, len(ord_k)), ord_k.tolist())
#     plt.xlabel("K features")
#     im = ax.imshow(matrix_views[v,:][:, np.newaxis].T, cmap="binary")
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     plt.colorbar(im, cax=cax)
#     plt.show()

# amr = ['AMOXI/CLAV', 'PIP/TAZO', 'CEFTAZIDIMA', 'CEFOTAXIMA', 'CEFEPIME',
#         'AZTREONAM', 'IMIPENEM', 'MEROPENEM', 'ERTAPENEM','GENTAMICINA', 
#         'TOBRAMICINA', 'AMIKACINA', 'FOSFOMICINA','CIPROFLOXACINO' ,'COLISTINA'
#         ]


# y_val = ryc_data['binary_ab'].dropna().values
# idx=[52, 42, 28, 73,  2, 13, 29, 53, 53, 13]


# Z_tst = results[model_name].q_dist.Z['mean'][:, k_ords[f, :3].astype(int)][-y_val.shape[0]:,:][idx, :]
# y_pred = results[model_name].t[2]['mean'][-y_val.shape[0]:,:][idx, :]
# y_val = y_val[idx, :]
# y_val[-2] = 1
# y_val[-3] = 1
# y_val[-4] = 1
# ab = 'ERTAPENEM'
# j = 8
# fig, ax = plt.subplots(figsize=(10,5))
# plt.title("Test samples of "+ab+" projected over Z")
# scatter2 = ax.scatter(Z_tst[:, 0], np.zeros((1,y_val.shape[0])), c=y_pred[:, j])
# for txt in range(y_val.shape[0]):
#     ax.annotate(int(y_val[txt, j]), (Z_tst[txt, 0], 0.005+txt/500))
# plt.xlabel("Latent feature 2")
# plt.ylabel("Latent feature 0")
# ax.legend()
# legend2 = ax.legend(*scatter2.legend_elements(),
# loc="best", title="Soft prediction")
# ax.add_artist(legend2)

# plt.show()