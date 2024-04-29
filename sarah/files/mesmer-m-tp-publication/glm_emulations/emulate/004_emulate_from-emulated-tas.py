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

# from utils.load_data import load_and_prepare_land_dataset, load_land_gmt_dataset
from utils.prepare_input import get_input_data

# scipr specific 
# training process
from sklearn.preprocessing   import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.api import GLM 

# global trend  
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.preprocessing   import PolynomialFeatures
from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import cross_val_score

# local trend 
from sklearn.metrics.pairwise   import haversine_distances

from statsmodels.api import GLM 
import statsmodels.api as sm

from statsmodels.nonparametric.smoothers_lowess import lowess

#%%

ssp_id      = 'ssp585'

if __name__ == '__main__':
    for model_id in cset.model_ids:
        print(model_id)
        run_id_training   = cset.model_training_mapping[model_id]

        n_emus      = 100
        n_years     = 251
        
        # ToDo! 
        # use std_tas, PCA and GLM as coefficient matrices and not as sklearn objects
        # has to be adapted in later versions of the code as to store 
        # fitted statsmodels/sklearn objects directly as useable numpy arrays
        # will also resolve versioning difficulties
        calib_path_trend  = cset.output_path.joinpath(f'glm_emus_v02/calib/loc_trend/{model_id}')
        selected_loc      = joblib.load(calib_path_trend.joinpath(f'SelectedLoc_{model_id}_{run_id_training}.pkl')) 
        std_tas           = joblib.load(calib_path_trend.joinpath(f'StdTas_{model_id}_{run_id_training}.pkl')) 
        PCAs              = joblib.load(calib_path_trend.joinpath(f'PCAs_{model_id}_{run_id_training}.pkl')) 
        PCA_MEANs         = np.array([pca.mean_ for pca in PCAs])
        PCA_COMPs         = np.array([pca.components_ for pca in PCAs])
        del PCAs
        GLMs              = joblib.load(calib_path_trend.joinpath(f'GLMs_{model_id}_{run_id_training}.pkl')) 
        GLM_COEFFs        = np.array([glm.params for glm in GLMs])
        del GLMs
        STDs              = joblib.load(calib_path_trend.joinpath(f'GLM-STDs_{model_id}_{run_id_training}.pkl'))         
        STD_MEANs         = np.array([std.mean_ for std in STDs])
        STD_SCALEs        = np.array([std.scale_ for std in STDs])  
        del STDs  
    
        tas_emus          = np.array(joblib.load(cset.output_path.joinpath(f'simple_tas_emus/full_emu/{model_id}/tas_full-emus_{model_id}_{ssp_id}_{run_id_training}_v1.pkl')))
        # precip_emuvar_path= cset.output_path.joinpath(f'glm_emus_v02/emus/pr/loc_var/{model_id}/')
        # new_samples       = joblib.load(precip_emuvar_path.joinpath(f'pr_var_{model_id}_{run_id_training}_v0.pkl'))[:n_emus, :, :]
        
        def parallel_predictions(i_run):
            '''
            Function for estimating the fraction fo the precipitaiton signal that 
            is deterministcally derivable from temperature data. Takes the filename f 
            of the temperature data to be used as predictors as input and then applies 
            the calibrated parameters 
            '''
            
            tas     = tas_emus[i_run, :, :]
            n_years = int(np.shape(tas)[0]/12)
            tas_std = std_tas.transform(tas[:, :].reshape(-1, 12*ccon.n_sindex)).reshape(-1, ccon.n_sindex)

            def gridpoint_month_pred_coeffs(X_tas, glm_coeffs, pca_mean, pca_comp, std_mean, std_scale):
                X_trafo = (X_tas - pca_mean).dot(pca_comp.T)[:, :8] 
                
                n_ts             = n_years
                frac_lowess      = 50 / n_ts
                
                t_lowess = lowess(X_trafo[:, 0],
                                np.arange(n_ts), 
                                return_sorted=False, 
                                frac=frac_lowess, 
                                it=0)

                X_trafo = np.c_[t_lowess,
                                t_lowess**2,
                                X_trafo[:, 0]-t_lowess, 
                                X_trafo[:, 1:8],
                                t_lowess*(X_trafo[:, 0]-t_lowess),
                                t_lowess*X_trafo[:, 1],
                                t_lowess*X_trafo[:, 2],
                                t_lowess*X_trafo[:, 3],
                                t_lowess*X_trafo[:, 4],
                                t_lowess*X_trafo[:, 5],
                                t_lowess*X_trafo[:, 6],
                                t_lowess*X_trafo[:, 7]
                                ]
                
                X_pred = np.c_[np.ones(n_years), 
                               (X_trafo - std_mean)/std_scale
                               ]
                
                return(np.exp(X_pred@glm_coeffs))
                        
            #takes ~1min, parallelizing makes it slower 
            pr_trend_emus = np.array([gridpoint_month_pred_coeffs(tas_std[m::12, selected_loc[i]],
                                                                        GLM_COEFFs[m*2652 + i, :],
                                                                        PCA_MEANs[m*2652 + i, :],
                                                                        PCA_COMPs[m*2652 + i, :, :],
                                                                        STD_MEANs[m*2652 + i, :],
                                                                        STD_SCALEs[m*2652 + i, :]
                                                                        ) for m, i in ccon.mi_ind]).T.reshape(-1, ccon.n_sindex)
        
            # store trend predictions
            precip_emu_path   = cset.output_path.joinpath(f'glm_emus_v02/emus/pr/loc_trend/{model_id}/full/')
            Path(precip_emu_path).mkdir(parents = True, exist_ok = True)
            joblib.dump(pr_trend_emus/86400, precip_emu_path.joinpath(f'pr_full-trend-emu_{model_id}_{ssp_id}_{i_run}.pkl'))
            return(0)
        
        print('Reconstructing deterministic fraction of precipitation signal')
        _ = Parallel(n_jobs = 12)(delayed(parallel_predictions)(i_run_) for i_run_ in tqdm(range(n_emus), total = n_emus))
        

    