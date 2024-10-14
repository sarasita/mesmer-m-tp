#%%
"""
Wrapper script for calibrating GLM paramters (including PCA & Normalization paramters) 
across models; 
"""

#################################
#       Import Statements       #
#################################

# setting paths such that execution works from console & interactive shell 
import os
import sys
os.chdir('/home/ubuntu/sarah/files/mesmer-m-tp-publication/')
sys.path.append('/home/ubuntu/sarah/files/mesmer-m-tp-publication/')

# computation
import xarray as xr
import numpy as np
import joblib
from joblib              import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# utils
from tqdm   import tqdm 
from pathlib import Path
import warnings

# plotting
import matplotlib.pyplot as plt
import cartopy.crs       as ccrs
import matplotlib as mpl
import seaborn as sns

# constants and settings
import config.settings as cset
import config.constants as ccon

# helper functions
from train.glm_general import converting_precipitation
from train.glm_trend import get_closest_locations, get_gridpoint_month_pca, gridpoint_month_glm

#%%

if __name__ == '__main__':
    # takes approximately 8 mins for each model
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
            # setting output path
            calib_path       = cset.output_path.joinpath(f'glm_emus_v02/calib/loc_trend/{model_id}')
            # check if basic calibration parameters (i.e. List of locations close to each grid-point, 
            # StandardScaler object for scaling temperatures, and PCAs already exist); if so, open the
            # pre-calibrated parameters, if not generate the pre-calibrated parameters 
            try: 
                print('Opening pre-calibrated parameters')
                selected_loc = joblib.load(calib_path.joinpath(f'SelectedLoc_{model_id}_{run_id_training}.pkl'))
                std_tas      = joblib.load(calib_path.joinpath(f'StdTas_{model_id}_{run_id_training}.pkl'))
                PCAs         = joblib.load(calib_path.joinpath(f'PCAs_{model_id}_{run_id_training}.pkl'))
                tas_training_std = std_tas.transform(tas_training.reshape(-1, 12*ccon.n_sindex)).reshape(-1, ccon.n_sindex)
            except:  
                print('Calibrating basic parameters')
                Path(calib_path).mkdir(parents = True, exist_ok = True)
                # computing the set of selected temperature locations used as predictors
                # for precipitation at a certain location 
                # - set of selected locations 
                selected_loc     = get_closest_locations()
                joblib.dump(selected_loc, calib_path.joinpath(f'SelectedLoc_{model_id}_{run_id_training}.pkl')) 
                # - standardizing and PCA transforming predictors
                # - standardizing predictors 
                std_tas          = StandardScaler() 
                tas_training_std = std_tas.fit_transform(tas_training.reshape(-1, 12*ccon.n_sindex)).reshape(-1, ccon.n_sindex)
                joblib.dump(std_tas, calib_path.joinpath(f'StdTas_{model_id}_{run_id_training}.pkl')) 
                # - PCAs (parallelizing does noot lead to a speedup)
                PCAs     = [get_gridpoint_month_pca(tas_training_std[m_::12, selected_loc[i_]]) for m_, i_ in tqdm(ccon.mi_ind, total = len(ccon.mi_ind))] 
                joblib.dump(PCAs, calib_path.joinpath(f'PCAs_{model_id}_{run_id_training}.pkl'))

            print('Conducting actual calibration')
            Res = Parallel(n_jobs = -1)(delayed(gridpoint_month_glm)(tas_training_std[m_::12, selected_loc[i_]],
                                                                    pr_training[m_::12, i_],
                                                                    PCAs[m_*2652+i_]) for m_, i_ in tqdm(ccon.mi_ind, total = len(ccon.mi_ind)))
            # unpack return data
            GLMs = [Res[j][1] for j in range(len(ccon.mi_ind))]
            STDs = [Res[j][0] for j in range(len(ccon.mi_ind))]
            
            joblib.dump(GLMs, calib_path.joinpath(f'GLMs_{model_id}_{run_id_training}.pkl')) 
            joblib.dump(STDs, calib_path.joinpath(f'GLM-STDs_{model_id}_{run_id_training}.pkl')) 

            # computing and fitting residuals 
            YPREDS  = np.array([GLMs[m_*2652 + i_].fittedvalues for m_, i_ in tqdm(ccon.mi_ind, total = len(ccon.mi_ind))]).T.reshape(-1, 2652)
            joblib.dump(YPREDS, calib_path.joinpath(f'YPREDS_{model_id}_{run_id_training}.pkl')) 

# %%
