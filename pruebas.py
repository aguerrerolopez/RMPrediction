import pickle
import pandas
import missingno as msno

data_path = "./data/ryc_data_mediansample_only2-12_TIC.pkl"
# folds_path = "./data/HGM_10STRATIFIEDfolds_muestrascompensadas_"+familia+".pkl"

with open(data_path, 'rb') as pkl:
            hgm_data = pickle.load(pkl)

