import pickle
import random

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


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

ryc = True
hgm = False
both = False
no_ertapenem = False
hgm=1
#

#"./Results/mediana_10fold_rbf/HGM_full_linear_15views_noCP_1_prun0.1.pkl"
modelo_a_cargar = "./Results/mediana_10fold_rbf/HGM_full_linear_15views_PRUEBALOCA_prun0.1.pkl"
fifteen_views =1
eight_views=0
# modelo_a_cargar = "./Results/RyC_noprior_"+familia+"_prun0.01.pkl"

folds_path = "./data/HGM_10STRATIFIEDfolds_muestrascompensada_pruebaloca.pkl"
data_path = "./data/hgm_data_mediansample_only2-12_TIC.pkl"

with open(data_path, 'rb') as pkl:
   gm_data = pickle.load(pkl)
with open(folds_path, 'rb') as pkl:
    folds = pickle.load(pkl)


##### PREDECIR CARB Y BLEE
old_fen = gm_data['fen']
old_fen = old_fen.drop(old_fen[old_fen['Fenotipo CP']==1].index)
ph = old_fen[['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']]

x = gm_data["maldi"].loc[ph.index]
y = gm_data['binary_ab'].loc[ph.index]

# # ###### PREDECIR RESTO
# x = gm_data["maldi"]
# y = gm_data['binary_ab']


with open(modelo_a_cargar, 'rb') as pkl:
    results = pickle.load(pkl)

# auc_by_ab = np.zeros((5,10,15))
# auc_by_phen = np.zeros((5,10,4))
auc_by_ab = np.zeros((10,15))
auc_by_phen = np.zeros((10,3))
# for m in range(5):
    # modelname = "model_random"+str(m)
    # results = results_random[modelname]

familias = { "penicilinas": ['AMOXI/CLAV ', 'PIP/TAZO'],
                    "cephalos": ['CEFTAZIDIMA', 'CEFOTAXIMA', 'CEFEPIME'],
                    "monobactams": ['AZTREONAM'],
                    "carbapenems": ['IMIPENEM', 'MEROPENEM', 'ERTAPENEM'],
                    "fluoro":['CIPROFLOXACINO'],
                    "aminos": ['GENTAMICINA', 'TOBRAMICINA', 'AMIKACINA'],
                    "fosfo": ['FOSFOMICINA'],
                    "otros":['COLISTINA']
                    }

for f in range(len(folds["train"])):
    y_tr, y_val = y.loc[folds["train"][f]], y.loc[folds["val"][f]]
    ## PRED carb blee
    ph_val = ph.loc[folds["val"][f]].values
    y_val = y_val.values
    # ### PRED RESTO
    # cols = ['GENTAMICINA', 'TOBRAMICINA', 'AMIKACINA', 'FOSFOMICINA', 'CIPROFLOXACINO', 'COLISTINA']
    # y_val = y_val[cols].values


    model_name = "model_fold" + str(f)
    if fifteen_views:
        for h in range(15):
            ### PRED RESTO
            # if h>5:
            #     break
            # ### PRED CARB BLEE
            if h>8:
                break
            y_true = y_val[:, h]
            ### PRED CARB BLEE
            y_pred = results[model_name].t[2+h]['mean'][-y_val.shape[0]:, 0]
            ### PRED RESTO
            # y_pred = results[model_name].t[1+h]['mean'][-y_val.shape[0]:, 0]
            # auc_by_ab[f, h] = roc_auc_score(y_true, y_pred)
    if eight_views:
        for i,fam in enumerate(familias):
            if fam=="penicilinas": delay=0
            if fam=="cephalos": delay=2
            if fam=="monobactams": delay=5
            if fam=="carbapenems": delay=6
            if fam=="fluoro": delay=9
            if fam=="aminos": delay=10
            if fam=="fosfo": delay=13
            if fam=="otros": delay=14

            for j, ab in enumerate(familias[fam]):
                y_true = y_val[:, j+delay]
                y_pred = results[model_name].t[2+i]['mean'][-y_val.shape[0]:, j]
                auc_by_ab[f, j+delay] = roc_auc_score(y_true, y_pred)
    ### PRED CARB BLEE
    y_true = ph_val
    y_pred = results[model_name].t[1]['mean'][-y_val.shape[0]:, :]
    print(y_true)
    print(y_pred)
    auc_by_phen[f,0] = roc_auc_score(y_true[:, 0], y_pred[:, 0])
    auc_by_phen[f,1] = roc_auc_score(y_true[:, 1], y_pred[:, 1])
    auc_by_phen[f,2] = roc_auc_score(y_true[:, 2], y_pred[:, 2])
   

print("RESULTS PHEN")
print("MEAN")
print(np.mean(auc_by_phen, axis=0))
print("STD")
print(np.std(auc_by_phen, axis=0))

print("RESULTS ANTIBIOTIC")
print("MEAN")
print(np.mean(auc_by_ab, axis=0))
print(np.mean(auc_by_ab))
print("STD")
print(np.std(auc_by_ab, axis=0))


# RESULTADOS PARA UN FOLD ESPECÍFICO
representative_fold = np.argmin(np.abs(np.sum((auc_by_phen-np.mean(auc_by_phen, axis=0)), axis=1)))
f=representative_fold
model_name = "model_fold"+str(f)

# TODO: Sacar vistas comunes

from mpl_toolkits.axes_grid1 import make_axes_locatable
k_len = [len(results[model_name].q_dist.alpha[1]['b']) for f in range(10)]
k_ords = np.zeros((10, np.max(k_len)))
# CAMBIAR ALPHA 2 SI HAY VISTA FENOTIPO, CAMBIAR ALPHA1 SI N LA HAY
ord_k = np.argsort(results[model_name].q_dist.alpha[1]['a']/results[model_name].q_dist.alpha[1]['b'])
k_ords[f, :len(ord_k)] = ord_k
if hgm: matrix_views = np.zeros((len(results[model_name].q_dist.W),results[model_name].q_dist.W[0]['mean'].shape[1]))
# matrix_views = np.zer.os((20,results[model_name].q_dist.W[0]['mean'].shape[1]))
n_cols = 13
matrix_views = np.zeros((n_cols,results[model_name].q_dist.W[0]['mean'].shape[1]))
for z in range(matrix_views.shape[0]):
    if z==0:
        X=results[model_name].X[0]['X']
        W=X.T@results[model_name].q_dist.W[0]['mean']
        matrix_views[z, :]=np.mean(np.abs(W), axis=0)[ord_k]
    elif z>0 and z<4:
        matrix_views[z, :]=np.abs(results[model_name].q_dist.W[1]['mean'][z-1,:][ord_k])
    else:
        print(z-2)
        # matrix_views[z, :]=np.mean(np.abs(results[model_name].q_dist.W[z-3]['mean']), axis=0)[ord_k]
        matrix_views[z, :]=np.mean(np.abs(results[model_name].q_dist.W[z-2]['mean']), axis=0)[ord_k]

if fifteen_views:
#     views = ['MALDI','CP+ESBL', 'ESBL', 'noCP noESBL','AMOXI/CLAV ', 'PIP/TAZO', 'CEFTAZIDIMA', 'CEFOTAXIMA', 'CEFEPIME', 'AZTREONAM', 'IMIPENEM', 'MEROPENEM', 'ERTAPENEM',
# 'CIPROFLOXACINO', 'GENTAMICINA', 'TOBRAMICINA', 'AMIKACINA', 'FOSFOMICINA', 'COLISTINA']
    views = ['MALDI', 'CARB and ESBL', 'ESBL', 'Susceptible','AMOXI/CLAV ', 'PIP/TAZO', 'CEFTAZIDIMA', 'CEFOTAXIMA', 'CEFEPIME', 'AZTREONAM', 'IMIPENEM', 'MEROPENEM', 'ERTAPENEM']
if eight_views:
    views = ['MALDI','PHENOTYPE', 'PENICILLINS', 'CEPHALOUS', 'MONOBACTAMS', 'CARBAPENEMS', 'FLUOROS', 'AMINOS', 'FOSFOMYCIN','POLYMIXINS']

from matplotlib import cm
cmap = cm.get_cmap('Dark2', 9)

for v, view in enumerate(views):
    # if v>3: break
    if fifteen_views:
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

    if eight_views:
        delay=v

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



# X=results[model_name].X[0]['X']
# W=X.T@results[model_name].q_dist.W[0]['mean']

# W_2norm = np.zeros((len(views), 10000))
# for v, view in enumerate(views):
#     if v>0:
#         mask = ord_k[np.where(matrix_views[v, :]>0.1)]
#         W_2norm[v, :] = np.linalg.norm(W[:, mask], axis=1)
#     else:
#         W_2norm[v, :] = np.linalg.norm(W, axis=1)

# groups = [['AMOXI/CLAV ', 'PIP/TAZO', 'CEFTAZIDIMA', 'CEFOTAXIMA', 'CEFEPIME', 'AZTREONAM','CIPROFLOXACINO'],
#           ['ERTAPENEM','TOBRAMICINA',],
#           ['IMIPENEM', 'MEROPENEM'],
#           ['GENTAMICINA',  'AMIKACINA', 'FOSFOMICINA', 'COLISTINA']]

# plt.figure(figsize=(15,10))
# plt.title("Weight matrix for Group 4")
# for v, view in enumerate(views):
#     if view in groups[3]:
#         r_samples = np.vstack(x.values[(y[views[v]]==1).values])
#         s_samples = np.vstack(x.values[(y[views[v]]==0).values])
#         plt.plot(range(2000, 12000), np.mean(r_samples, axis=0), label=view+" RESISTANT mean")
#         plt.plot(range(2000, 12000), np.mean(s_samples, axis=0), label=view+" SENSIBLE mean")
        
#     # if view in groups[1]:
#     #     fig2.plot(W_2norm[v, :], alpha=0.5, label=view)
#     # if view in groups[2]:
#     #     fig3.plot(W_2norm[v, :], alpha=0.5, label=view)
#     # if view in groups[3]:
#     #     fig4.plot(W_2norm[v, :], alpha=0.5, label=view)
# plt.plot(range(2000, 12000), W_2norm[-1, :], alpha=0.5, label="W matrix for Group 4")
# plt.legend()
# spath = "./Results/plots_reunion2406/weightmatrices_group4.png"
# # plt.savefig(spath)
# plt.show()


# names = ['CP+ESBL', 'ESBL', 'noCP noESBL']

# X=results[model_name].X[0]['X']
# W_maldi=X.T@results[model_name].q_dist.W[0]['mean']
# W_proj = W_maldi@results[model_name].q_dist.W[1]['mean'].T

# for v, view in enumerate(ph.columns):
#     # W_proj = W_maldi@results[model_name].q_dist.W[1+v]['mean'].T
#     r_samples = np.vstack(x.values[(ph[view]==1).values])
#     s_samples = np.vstack(x.values[(ph[view]==0).values])
#     plt.figure(figsize=[7.5,5])
#     plt.title(names[v])
#     if v<2:
#         label1="Resistant to "+view
#         label2="Sensible to "+view
#         plt.plot(range(2000,12000), np.mean(r_samples, axis=0), label=label1)
#         plt.plot(range(2000,12000), np.mean(s_samples, axis=0),'--', label=label2)
#     else:
#         label1="Sensible to CP and ESBL"
#         label2="Resistant to CP OR ESBL"
#         plt.plot(range(2000,12000), np.mean(s_samples, axis=0), label=label2)
#         plt.plot(range(2000,12000), np.mean(r_samples, axis=0),'--', label=label1)
        
#     # if v==0:
#     #     W_proj[:, v] = W_proj[:, v]*10
#     plt.plot(range(2000,12000), W_proj[:, v],color='black', alpha=0.2, label="Weight vector")
#     # plt.plot(range(2000,12000), W_proj,color='black', alpha=0.2, label="Weight vector")
#     plt.legend()
#     spath = "./Results/plots_reunion2406/weightmatrices_"+names[v].replace(" ", "").replace("/","-")+".png"
#     # plt.savefig(spath)
#     plt.show()

# for v, view in enumerate(ph.columns):
#     # W_proj = W_maldi@results[model_name].q_dist.W[1+v]['mean'].T
#     r_samples = np.vstack(x.values[(ph[view]==1).values])
#     s_samples = np.vstack(x.values[(ph[view]==0).values])
#     plt.figure(figsize=[7.5,5])
#     plt.title("ZOOM into: "+names[v])
#     if v<2:
#         label1="Resistant to "+view
#         label2="Sensible to "+view
#         plt.plot(range(2000,3000), np.mean(r_samples, axis=0)[0:1000], label=label1)
#         plt.plot(range(2000,3000), np.mean(s_samples, axis=0)[0:1000],'--', label=label2)
    
#     else:
#         label1="Sensible to CP and ESBL"
#         label2="Resistant to CP OR ESBL"
#         plt.plot(range(2000,3000), np.mean(s_samples, axis=0)[0:1000], label=label2)
#         plt.plot(range(2000,3000), np.mean(r_samples, axis=0)[0:1000],'', label=label1)
#     # if v==0:
#     #     W_proj[:, v] = W_proj[:, v]*10
#     plt.plot(range(2000,3000), W_proj[:, v][0:1000],color='black', alpha=0.2, label="Weight vector")
#     # plt.plot(range(8000,10000), W_proj,color='black', alpha=0.2, label="Weight vector")
#     plt.legend()
#     spath = "./Results/plots_reunion2406/weightmatrices_"+names[v].replace(" ", "").replace("/","-")+".png"
#     # plt.savefig(spath)
#     plt.show()
    

################################################# Solo latentes específicas
X=results[model_name].X[0]['X']
W_maldi=X.T@results[model_name].q_dist.W[0]['mean']
# lf = ord_k[:4]
# lf = [3, 16, 1, 13]
lf = [23, 1, 21]
ls=['--', '-.', ':', (0, (1,1)), (0, (3,5,1,5))]

plt.figure(figsize=[15,10])
plt.title("Latent feature "+str(lf)+ "in MALDI weight matrix")
for k,l in enumerate(lf):
    alpha=0.5
    plt.plot(range(2000,12000), W_maldi[:, l], alpha=alpha, linestyle=ls[k], label="Latent feature "+str(l))
plt.legend()
plt.savefig("Results/plots_reunion2406/latentfeatures.png")
plt.show()

titles=['to CARB', 'to ESBL']
for k, col in enumerate(ph.columns):
    plt.figure(figsize=[15,10])
    plt.title(col+" view")
    if k==0:
        label1 ="Resistant "+titles[0]+" or "+titles[1]
        label2 ="Sensible "+titles[0]+" and "+titles[1]
    if k==1:
        label1 ="Resistant "+titles[1]
        label2 ="Sensible "+titles[1]
    if k==2:
        label2 ="Sensible "+titles[0]+" and "+titles[1]
        label1 ="Resistant "+titles[0]+" or "+titles[1]
    r_samples = np.vstack(x.values[(ph[col]==1).values])
    s_samples = np.vstack(x.values[(ph[col]==0).values])
    plt.plot(range(2000,12000), np.mean(r_samples, axis=0), color='tab:blue', label=label1)
    plt.fill_between(range(2000,12000), np.mean(r_samples, axis=0)-np.std(r_samples, axis=0), np.mean(r_samples, axis=0)+np.std(r_samples, axis=0), color='tab:blue', alpha=0.2)
    plt.plot(range(2000,12000), np.mean(s_samples, axis=0), color='tab:red', label=label2)
    plt.fill_between(range(2000,12000), np.mean(s_samples, axis=0)-np.std(s_samples, axis=0), np.mean(s_samples, axis=0)+np.std(s_samples, axis=0), color='tab:red', alpha=0.2)
    plt.xticks(ticks=np.arange(2000,12000, 1000), labels=['2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000'])
    plt.legend() 
    plt.show()

X=results[model_name].X[0]['X']
W_maldi=X.T@results[model_name].q_dist.W[0]['mean']
lf = [[23, 1, 21], [23, 1, 21], [23, 1, 21]]
# lf = [[3, 16, 1, 13], [3, 16, 1, 13], [3, 16, 1, 13]]
# lf = [ord_k[:4], ord_k[:4],ord_k[:4]]
color=['tab:red', 'tab:green', 'tab:pink', 'tab:gray']

res_samples = [np.vstack(x[ph[col]==1].sample(1, random_state=0).values)[0] for col in ph.columns]
sen_samples = [np.vstack(x[ph[col]==0].sample(1, random_state=0).values)[0] for col in ph.columns]

titulo = ['CARB and ESBL', 'ESBL', 'Susceptible']
for k, col in enumerate(ph.columns):
    plt.figure(figsize=[15,10])
    plt.title(titulo[k]+" view")
    if k==2:
        label1 = "A susceptible sample"
        label2 = "A random resistant sample"
    else:
        label1 = "A resistant sample"
        label2 = "A random sensible sample"
    plt.plot(range(2000,12000), res_samples[k], '--', alpha=0.2, label=label1)
    plt.plot(range(2000,12000), sen_samples[k], '--', alpha=0.2, label=label2)
    for j,l in enumerate(lf[k]):
        W_proj = W_maldi[:, l]*results[model_name].q_dist.W[1]['mean'][k,l]
        plt.plot(range(2000,12000), W_proj, color=color[j], alpha=0.8, label="Latent feature "+str(l))
    plt.xticks(ticks=np.arange(2000,12000, 1000), labels=['2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000'])
    plt.legend()
    path = "Results/plots_reunion2406/privadas_"+col+".png"
    # plt.savefig(path)
    plt.show()

for k, col in enumerate(ph.columns):
    plt.figure(figsize=[15,10])
    plt.title("ZOOM IN: "+titulo[k]+" view")
    if k==2:
        label1 = "A susceptible sample"
        label2 = "A random resistant sample"
    else:
        label1 = "A resistant sample"
        label2 = "A random sensible sample"
    plt.plot(range(2000,2500), res_samples[k][0:500], '--', alpha=0.2, label=label1)
    plt.plot(range(2000,2500), sen_samples[k][0:500], alpha=0.2, label=label2)
    for j,l in enumerate(lf[k]):
        W_proj = W_maldi[:, l]*results[model_name].q_dist.W[1]['mean'][k,l]
        plt.plot(range(2000,2500), W_proj[0:500], color=color[j], alpha=0.8, label="Latent feature "+str(l))
    # plt.xticks(ticks=np.arange(2000,12000, 1000), labels=['2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000'])
    plt.legend()
    path = "Results/plots_reunion2406/zoom1_"+col+".png"
    plt.savefig(path)
    plt.show()

for k, col in enumerate(ph.columns):
    if k==2:
        label1 = "A susceptible sample"
        label2 = "A random resistant sample"
    else:
        label1 = "A resistant sample"
        label2 = "A random sensible sample"
    plt.figure(figsize=[15,10])
    plt.title("ZOOM IN: "+titulo[k]+" view")
    plt.plot(range(7000,7500), res_samples[k][5000:5500], '--', alpha=0.2, label=label1)
    plt.plot(range(7000,7500), sen_samples[k][5000:5500], alpha=0.2, label=label2)
    for j,l in enumerate(lf[k]):
        W_proj = W_maldi[:, l]*results[model_name].q_dist.W[1]['mean'][k,l]
        plt.plot(range(7000,7500), W_proj[5000:5500], color=color[j], alpha=0.8, label="Latent feature "+str(l))
    # plt.xticks(ticks=np.arange(2000,12000, 1000), labels=['2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000'])
    plt.legend()
    path = "Results/plots_reunion2406/zoom2_"+col+".png"
    plt.savefig(path)
    plt.show()

for k, col in enumerate(ph.columns):
    if k==2:
        label1 = "A susceptible sample"
        label2 = "A random resistant sample"
    else:
        label1 = "A resistant sample"
        label2 = "A random sensible sample"
    plt.figure(figsize=[15,10])
    plt.title("ZOOM IN: "+titulo[k]+" view")
    plt.plot(range(9500,10000), res_samples[k][7500:8000], '--', alpha=0.2, label=label1)
    plt.plot(range(9500,10000), sen_samples[k][7500:8000], alpha=0.2, label=label2)
    for j,l in enumerate(lf[k]):
        W_proj = W_maldi[:, l]*results[model_name].q_dist.W[1]['mean'][k,l]
        plt.plot(range(9500,10000), W_proj[7500:8000], color=color[j], alpha=0.8, label="Latent feature "+str(l))
    # plt.xticks(ticks=np.arange(2000,12000, 1000), labels=['2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000'])
    plt.legend()
    path = "Results/plots_reunion2406/zoom3_"+col+".png"
    plt.savefig(path)
    plt.show()

for k, col in enumerate(ph.columns):
    if k==2:
        label1 = "A susceptible sample"
        label2 = "A random resistant sample"
    else:
        label1 = "A resistant sample"
        label2 = "A random sensible sample"
    plt.figure(figsize=[15,10])
    plt.title("ZOOM IN: "+titulo[k]+" view")
    plt.plot(range(9900,10000), res_samples[k][7900:8000], '--', alpha=0.2, label=label1)
    plt.plot(range(9900,10000), sen_samples[k][7900:8000], alpha=0.2, label=label2)
    for j,l in enumerate(lf[k]):
        W_proj = W_maldi[:, l]*results[model_name].q_dist.W[1]['mean'][k,l]
        plt.plot(range(9900,10000), W_proj[7900:8000], color=color[j], alpha=0.8, label="Latent feature "+str(l))
    # plt.xticks(ticks=np.arange(2000,12000, 1000), labels=['2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000'])
    plt.legend()
    path = "Results/plots_reunion2406/zoom3_"+col+".png"
    plt.show()


################# Project to 3D

# ph_val = ph.loc[folds["val"][f]].values

# ph_tr = results[model_name].t[1]['mean'][:-ph_val.shape[0],:]

# for k, col in enumerate(ph.columns):
#     ord_k = np.argsort(-np.abs(results[model_name].q_dist.W[1]['mean'])[k,:])
#     Z_tr = results[model_name].q_dist.Z['mean'][:, ord_k[:3].astype(int)][:-ph_val.shape[0],:]
#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot(111, projection='3d')
#     plt.title(col+" projected over Z")
#     scatter1 = ax.scatter(Z_tr[:, 0], Z_tr[:, 1], Z_tr[:, 2], c=ph_tr[:, k])
#     legend1 = ax.legend(*scatter1.legend_elements(), loc='best', title="Classes")
#     ax.add_artist(legend1)

#     plt.legend()
#     ax.set_xlabel('Latent feature '+str(int(ord_k[0])))
#     ax.set_ylabel('Latent feature '+str(int(ord_k[1])))
#     ax.set_zlabel('Latent feature '+str(int(ord_k[2])))
#     ax.legend()

#     plt.show()

# Z_tst = results[model_name].q_dist.Z['mean'][:, ord_k[:3].astype(int)][-ph_val.shape[0]:,:]


# for k, col in enumerate(ph.columns):
#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot(111, projection='3d')
#     plt.title(col+" projected over Z")
#     scatter1 = ax.scatter(Z_tst[:, 0], Z_tst[:, 1], Z_tst[:, 2], c=ph_val[:, k])
#     legend1 = ax.legend(*scatter1.legend_elements(), loc='best', title="Classes")
#     ax.add_artist(legend1)

#     plt.legend()
#     ax.set_xlabel('Latent feature '+str(int(ord_k[0])))
#     ax.set_ylabel('Latent feature '+str(int(ord_k[1])))
#     ax.set_zlabel('Latent feature '+str(int(ord_k[2])))
#     ax.legend()

#     plt.show()


# X=results[model_name].X[0]['X']
# W_maldi=X.T@results[model_name].q_dist.W[0]['mean']
# lf = [[6, 16, 9], [6, 16, 9], [6, 16, 9]]
# names = ['Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL']
# color=['black', 'red', 'green']
# titles=['to CP', 'to ESBL']
# for k, col in enumerate(names):
#     plt.figure(figsize=[7.5,5])
#     if k==0:
#         plt.title(col+"+ESBL view")
#     else:
#         plt.title(col+" view")
#     r_samples = np.vstack(x.values[(ph[col]==1).values])
#     s_samples = np.vstack(x.values[(ph[col]==0).values])
#     if k==0:
#         label1 ="Resistant "+titles[0]+" or "+titles[1]
#         label2 ="Sensible "+titles[0]+" and "+titles[1]
#     if k==1:
#         label1 ="Resistant "+titles[1]
#         label2 ="Sensible "+titles[1]
#     if k==2:
#         label1 ="Sensible "+titles[0]+" and "+titles[1]
#         label2 ="Resistant "+titles[0]+" or "+titles[1]
#     plt.plot(range(2000,12000), np.mean(r_samples, axis=0), label=label1)
#     plt.plot(range(2000,12000), np.mean(s_samples, axis=0),'--', label=label2)
#     for j,l in enumerate(lf[k]):
#         W_proj = W_maldi[:, l]*results[model_name].q_dist.W[1]['mean'][k,l]
#         plt.plot(range(2000,12000), W_proj, color=color[j], alpha=0.3, label="Latent feature "+str(l))
#     plt.legend()
#     path = "Results/plots_reunion2406/comunes_"+col+".png"
#     # plt.savefig(path)
#     plt.show()

# plt.figure()
# plt.plot(W_maldi[:, 13])
# plt.plot(W_maldi[:, 22])
# plt.plot(W_maldi[:, 13]+W_maldi[:, 22])
# plt.show()



# phen = ['Fenotipo CP','Fenotipo CP+ESBL', 'Fenotipo  ESBL', 'Fenotipo noCP noESBL',]
# from matplotlib.colors import ListedColormap
# from matplotlib import cm

# greys = cm.get_cmap('Greys')
# newcolors = greys(np.linspace(0.5, 1, 2))
# greycmap = ListedColormap(newcolors)

# greys = cm.get_cmap('Blues')
# newcolors = greys(np.linspace(0.5, 1, 2))
# bluecmap = ListedColormap(newcolors)

# greys = cm.get_cmap('Greens')
# newcolors = greys(np.linspace(0.5, 1, 2))
# redcmap = ListedColormap(newcolors)

# ph_tr, ph_val = ph.loc[folds["train"][f]].values, ph.loc[folds["val"][f]].values
# ph_pred= results[model_name].t[1]['mean'][-ph_val.shape[0]:,:]
# for p, phenotype in enumerate(phen):
#     if p%2 == 0:
#         continue
#     k_len = [len(results[model_name].q_dist.alpha[1]['b']) for f in range(10)]
#     k_ords = np.zeros((1, np.max(k_len)))
#     ord_k = np.argsort(results[model_name].q_dist.alpha[1]['a']/results[model_name].q_dist.alpha[1]['b'])
#     k_ords[0, :len(ord_k)] = ord_k
    
#     Z_tr = results[model_name].q_dist.Z['mean'][:, k_ords[0, :3].astype(int)][:ph_tr.shape[0],:]
#     Z_tst = results[model_name].q_dist.Z['mean'][:, k_ords[0, :3].astype(int)][-ph_val.shape[0]:,:]

#     fig = plt.figure(figsize=(5,5))
#     ax = fig.add_subplot(111, projection='3d')
#     plt.title(phenotype+" projected over Z")
#     scatter1 = ax.scatter(Z_tr[:, 0], Z_tr[:, 1], Z_tr[:, 2], c=ph_tr[:, p], alpha=0.5, cmap=greycmap)
#     scatter2 = ax.scatter(Z_tst[:, 0], Z_tst[:, 1], Z_tst[:, 2], c=ph_val[:, p], cmap='coolwarm')
#     ax.set_xlabel('Latent feature '+str(int(k_ords[0, 0])))
#     ax.set_ylabel('Latent feature '+str(int(k_ords[0, 1])))
#     ax.set_zlabel('Latent feature '+str(int(k_ords[0, 2])))
#     ax.legend()
#     legend1 = ax.legend(*scatter1.legend_elements(),
#     loc="upper right", title="Train samples")
#     legend2 = ax.legend(*scatter2.legend_elements(),
#     loc="upper left", title="Test_samples")
#     ax.add_artist(legend1)
#     ax.add_artist(legend2)

#     plt.show()


# familias = {"Penicillins": ['AMOXI/CLAV ', 'PIP/TAZO'],
#             "Cephalous": ['CEFTAZIDIME', 'CEFOTAXIME', 'CEFEPIME'],
#             "Monobactams": ['AZTREONAM'],
#             "Carbapenems": ['IMIPENEM', 'MEROPENEM', 'ERTAPENEM'],
#             "Aminos": ['GENTAMYCIN', 'TOBRAMYCIN', 'AMIKACIN'],
#             "Fluoro":['CIPROFLOXACINO'],
#             "Fosfomycin": ['FOSFOMYCIN'],
#             "Polymixins":['COLISTIN']}
# y_tr, y_val = y.loc[folds["train"][f]].values, y.loc[folds["val"][f]].values
# cmaps = [greycmap, bluecmap, redcmap]
# legend_loc = ['upper right', 'upper left', 'upper center']
# markers = ['o', 's', 'x']
# sizes = [500, 200, 100]
# for fa, fam in enumerate(familias):
#     if fa>0:
#         break
#     fam = "Cephalous"
#     y_pred= results[model_name].t[2+fa]['mean'][-y_val.shape[0]:,:]

#     ord_k = np.argsort(results[model_name].q_dist.alpha[2+fa]['a']/results[model_name].q_dist.alpha[2+fa]['b'])

#     Z_tr = results[model_name].q_dist.Z['mean'][:, ord_k[:3].astype(int)][:y_tr.shape[0],:]
#     Z_tst = results[model_name].q_dist.Z['mean'][:, ord_k[:3].astype(int)][-y_val.shape[0]:,:]

#     fig = plt.figure(figsize=(15,15))
#     ax = fig.add_subplot(111, projection='3d')
#     plt.title(fam+" projected over Z")
    
#     for a, anti in enumerate(familias[fam]):
#         scatter1 = ax.scatter(Z_tr[:, 0], Z_tr[:, 1], Z_tr[:, 2], s=sizes[a], marker=markers[a] ,c=y_tr[:, a], cmap=cmaps[a])
#         legend1 = ax.legend(*scatter1.legend_elements(), loc=legend_loc[a], title=anti)
#         ax.add_artist(legend1)

#     plt.legend()
#     ax.set_xlabel('Latent feature '+str(int(ord_k[0])))
#     ax.set_ylabel('Latent feature '+str(int(ord_k[1])))
#     ax.set_zlabel('Latent feature '+str(int(ord_k[2])))
#     ax.legend()

#     plt.show()

    