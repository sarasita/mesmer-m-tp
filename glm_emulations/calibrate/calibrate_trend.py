#%%
"""
Wrapper script for calibrating all GLM paramters (including PCA & Normalization paramters) 
for MESMER-M-TP
"""
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

from utils.prepare_input import get_input_data

from train.glm_general import converting_precipitation
from train.glm_trend import get_closest_locations, get_gridpoint_month_pca, gridpoint_month_glm


#%%
if __name__ == '__main__':
    for model_id in cset.model_ids:
        run_id_training  = cset.model_training_mapping[model_id]

        tas_training, pr_training_, GMT_training, GMT_trend_training, GMT_var_training = get_input_data(model_id, run_id_training = run_id_training)
        n_years          = np.shape(GMT_training)[0]

        # converting to average mm/day in the month 
        # cutting off extremely small values & rounding to save memory
        pr_training      = converting_precipitation(pr_training_, n_digits = 10)

        # Opening pre-calibrated parameters 
        calib_path       = cset.output_path.joinpath(f'glm_emus_v02/calib/loc_trend/{model_id}')
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

        # fit GLM response; ToDo: adjust selection process for alpha's
        
        # - fitting actual GLM
        #     - selecting alpha: chose the lowest alpha that does not suffer 
        #       from overfitting; selection process can be adjusted later on 
        all_alphas       = [4*[0] + 15*[0], 
                            4*[0] + 7*[0] + 8*[1],
                            2*[0] + 1*[1] + 1*[0] +  7*[0] + 8*[1],
                            4*[0] + 15*[1], 
                            4*[0] + 7*[1] + 8*[10],
                            4*[0] + 15*[10],
                            4*[0] + 7*[10] + 8*[100],
                            4*[0] + 15*[100], 
                            1*[0] + 18*[100],
                            1*[0] + 18*[1000],
                            1*[0] + 18*[100000]]

        n_fitting_rounds = len(all_alphas)

        GLMs            = [0]*(2652*12)
        STDs            = [0]*(2652*12)

        for i_round in range(n_fitting_rounds): 
            print(i_round)
            idx_fit  = [(m_, i_) for m_,i_ in ccon.mi_ind if GLMs[m_*2652+i_] == 0] 
            
            print(len(idx_fit))
            # GLMs_fit = [gridpoint_month_glm(m, i_, alphas = all_alphas[i_round]) for i_ in tqdm(idx_fit, total = len(idx_fit))]
            Res = Parallel(n_jobs = -1)(delayed(gridpoint_month_glm)(tas_training_std[m_::12, selected_loc[i_]],
                                                                    pr_training[m_::12, i_],
                                                                    PCAs[m_*2652+i_],
                                                                    alphas = all_alphas[i_round]) for m_, i_ in tqdm(idx_fit, total = len(idx_fit)))
            GLMs_fit = [Res[j][1] for j in range(len(idx_fit))]
            STDs_fit = [Res[j][0] for j in range(len(idx_fit))]

            for j in range(len(idx_fit)):
                m_, i_   = idx_fit[j]
                GLMs[m_*2652 + i_] = GLMs_fit[j]
                STDs[m_*2652 + i_] = STDs_fit[j]

        joblib.dump(GLMs, calib_path.joinpath(f'GLMs_{model_id}_{run_id_training}.pkl')) 
        joblib.dump(STDs, calib_path.joinpath(f'GLM-STDs_{model_id}_{run_id_training}.pkl')) 

        # computing and fitting residuals 
        YPREDS  = np.array([GLMs[m_*2652 + i_].fittedvalues for m_, i_ in tqdm(ccon.mi_ind, total = len(ccon.mi_ind))]).T.reshape(-1, 2652)
        joblib.dump(YPREDS, calib_path.joinpath(f'YPREDS_{model_id}_{run_id_training}.pkl')) 
