#%%
#################################
#       Import Statements       #
#################################

# setting paths correctly 
import os
import sys
os.chdir('/home/ubuntu/sarah/files/mesmer-m-tp-publication/')
sys.path.append('/home/ubuntu/sarah/files/mesmer-m-tp-publication/')

# computation
import xarray as xr
import numpy as np
import joblib
from joblib              import Parallel, delayed

# utils
from tqdm   import tqdm 
from pathlib import Path
import warnings

# plotting
import matplotlib.pyplot as plt
import cartopy.crs       as ccrs
import matplotlib as mpl
import seaborn as sns

# setting paths such that execution works from console & interactive shell 
import config.settings as cset
import config.constants as ccon

from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from utils.prepare_input import get_input_data

from train.glm_general import converting_precipitation
from train.glm_var import transform_residuals, month_specific_kde


#%%

if __name__ == '__main__':
    for model_id in cset.model_ids:
        run_id_training  = cset.model_training_mapping[model_id]

        calib_path_var   = cset.output_path.joinpath(f'glm_emus_v02/calib/loc_var/{model_id}')

        try: 
            print('Opening precipitaiton residuals')
            pr_var = joblib.load(calib_path_var.joinpath(f'PrVar_{model_id}_{run_id_training}.pkl')) 
        except:
            print('Calculating precipitaiton residuals')
            Path(calib_path_var).mkdir(parents = True, exist_ok = True)
            tas_training, pr_training_, GMT_training, GMT_trend_training, GMT_var_training = get_input_data(model_id, run_id_training = run_id_training)
            # converting to average mm/day in the month 
            # cutting off extremely small values & rounding to save memory
            pr_training      = converting_precipitation(pr_training_, n_digits = 10)

            calib_path_trend = cset.output_path.joinpath(f'glm_emus_v02/calib/loc_trend/{model_id}')
            YPREDS           = joblib.load(calib_path_trend.joinpath(f'YPREDS_{model_id}_{run_id_training}.pkl')) 
            
            pr_var           = np.log(pr_training/YPREDS)

            joblib.dump(pr_var, calib_path_var.joinpath(f'PrVar_{model_id}_{run_id_training}.pkl')) 

        n_years                  = int(np.shape(pr_var)[0]/12)
        # StandardScale residuals, apply PCA and then 
        # fit independent KDEs for each month 

        res_pr_StdPca_pipeline = transform_residuals(pr_var) 
        pr_var_pca             = res_pr_StdPca_pipeline.transform(pr_var)
        kde_list               = month_specific_kde(pr_var_pca)

        joblib.dump(res_pr_StdPca_pipeline, calib_path_var.joinpath(f'ResPrStdPcaPipeline_{model_id}_{run_id_training}.pkl'))
        joblib.dump(kde_list, calib_path_var.joinpath(f'KDEs_{model_id}_{run_id_training}.pkl'))

