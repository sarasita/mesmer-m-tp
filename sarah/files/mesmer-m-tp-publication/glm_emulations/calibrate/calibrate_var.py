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
        print(model_id)
        run_id_training  = cset.model_training_mapping[model_id]
        
        # finding all available files for training
        # i.e. iterating over folder that contains all the CMIP6 data for a given model_id 
        # and selecting all simulations available for the given run_id_training
        # only use data for which the ESM offers temperature AND precipitation data 
        tmp_files      = [f for f in os.listdir(cset.output_path.joinpath(f'processed_CMIP6/{model_id}/tas/')) if (run_id_training in f)]
        training_files = sorted([f for f in tmp_files if (os.path.isfile(cset.output_path.joinpath(f'processed_CMIP6/{model_id}/pr/pr_{f[4:]}')))])
        del tmp_files
        
        # if any training data exist, calibrate the glm parameters
        if training_files:
            ##########################################################
            #  Loading all available training data and concatenating #
            #  into a aingle array used for training                 #
            ##########################################################
            # on our internal server, ESM data is stored using numpy arrays; 
            # ESM-arrays have two dimensions: (time, coords)
            #     - time dimension: 
            #            spans period 1850-2015 for historical simulation (=165yearsx12months = 1980 timesteps)
            #            spans period 2015-2100 for RCP simulations (=86yearsx12months = 1032 timesteps)
            #     - coord dimension: 
            #            contains only data for land gridpoints; 2652 gridpoints, 
            #            follows the (lat,lon) combinations as given in the array ./data/new_coords.py    
            tas_training = np.concatenate([np.load(cset.output_path.joinpath(f'processed_CMIP6/{model_id}/tas/{f}')) for f in training_files])
            # precipitation data from ESMs does usually not contain a cut-off value 
            # therefore, the data sometimes contains very small negative values 
            # and also some very small positive residuals; to make the fitting robus
            # to these values, we convert precipitation from kg/m^2/s to mm/day and then
            # apply a cut-off (in this case we roudn to 10 digits)
            pr_training_ = np.concatenate([np.load(cset.output_path.joinpath(f'processed_CMIP6/{model_id}/pr/pr_{f[4:]}')) for f in training_files])
            pr_training  = converting_precipitation(pr_training_, n_digits =10)
            del pr_training_
            # the number of years should be sonsistent with 165+86*number_of_available_rcps
            n_years      = int(np.shape(tas_training)[0]/12)

            ################
            #  Calibration #
            ################
            
            print('Calculating precipitaiton residuals')
            calib_path_var   = cset.output_path.joinpath(f'glm_emus_v02/calib/loc_var/{model_id}')
            Path(calib_path_var).mkdir(parents = True, exist_ok = True)
            # loading trend calibration
            calib_path_trend = cset.output_path.joinpath(f'glm_emus_v02/calib/loc_trend/{model_id}')
            YPREDS           = joblib.load(calib_path_trend.joinpath(f'YPREDS_{model_id}_{run_id_training}.pkl')) 
            # computing residuals
            pr_var           = np.log(pr_training/YPREDS)
            joblib.dump(pr_var, calib_path_var.joinpath(f'PrVar_{model_id}_{run_id_training}.pkl')) 

            print('Calibrating parameters')
            # StandardScale residuals, apply PCA and then 
            # fit independent KDEs for each month 
            res_pr_StdPca_pipeline = transform_residuals(pr_var) 
            pr_var_pca             = res_pr_StdPca_pipeline.transform(pr_var)
            kde_list               = month_specific_kde(pr_var_pca)

            joblib.dump(res_pr_StdPca_pipeline, calib_path_var.joinpath(f'ResPrStdPcaPipeline_{model_id}_{run_id_training}.pkl'))
            joblib.dump(kde_list, calib_path_var.joinpath(f'KDEs_{model_id}_{run_id_training}.pkl'))

