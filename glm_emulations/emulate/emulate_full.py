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

#%%

ssp_id      = 'ssp585'
n_emu_years = 251

if __name__ == '__main__':
    for model_id in cset.model_ids:

        print(model_id)
        run_id_training   = cset.model_training_mapping[model_id]

        n_emus      = 100
        n_emu_years = 251
                
        GMT_emus                       = joblib.load(cset.output_path.joinpath(f'gmt_emus_LO/full/{model_id}/GMT-emus_{model_id}_{ssp_id}_{run_id_training}.pkl'))
        GMT_emus_trend                 = np.zeros(251)
        GMT_emus_trend[:ccon.n_hist_years]  = GMT_ds['rel_gmt_trend'].sel(run_id = run_id_training, ssp_id = 'historical').dropna(dim = 'year').values
        GMT_emus_trend[ccon.n_hist_years:]  = GMT_ds['rel_gmt_trend'].sel(run_id = run_id_training, ssp_id = ssp_id).dropna(dim = 'year').values
        GMT_emus_var                   = np.array([GMT_emus[i_emu, :]-GMT_emus_trend for i_emu in range(n_emus)]) 
        
        tas_emus       = np.array(joblib.load(cset.output_path.joinpath(f'/simple_tas_emus/full_emu/{model_id}/tas_full-emus_{model_id}_{ssp_id}_{run_id_training}_v1.pkl')))

        # load data for predicting trends
        calib_path   = cset.output_path.joinpath(f'glm_emus_v02/calib/loc_trend/{model_id}')
        selected_loc = joblib.load(calib_path.joinpath(f'SelectedLoc_{model_id}_{run_id_training}.pkl')) 
        std_tas      = joblib.load(calib_path.joinpath(f'StdTas_{model_id}_{run_id_training}.pkl')) 
        PCAs         = joblib.load(calib_path.joinpath(f'PCAs_{model_id}_{run_id_training}.pkl')) 
        GLMs         = joblib.load(calib_path.joinpath(f'GLMs_{model_id}_{run_id_training}.pkl')) 
        STDs         = joblib.load(calib_path.joinpath(f'GLM-STDs_{model_id}_{run_id_training}.pkl')) 
        
        from statsmodels.nonparametric.smoothers_lowess import lowess
        
        def parallel_predictions(i_emu):

            tas_emu_std = std_tas.transform(tas_emus[i_emu, :, :].reshape(-1, 12*ccon.n_sindex)).reshape(-1, ccon.n_sindex)
            
            def gridpoint_month_pred(m, i):
                X_tas  = tas_emu_std[m::12, selected_loc[i]]
                pca    = PCAs[m*2652 + i]
                std    = STDs[m*2652 + i] 
                X_trafo = pca.transform(X_tas)[:, :15]
                
                n_ts             = n_emu_years
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

                X_pred = np.c_[np.ones(n_emu_years), std.transform(X_trafo)]
                glm    = GLMs[m*2652 + i]
                return(glm.predict(X_pred))
            
            pr_trend_emus_tmp = np.array([gridpoint_month_pred(m_, i_) for m_, i_ in ccon.mi_ind]).T.reshape(-1, ccon.n_sindex)
            return(pr_trend_emus_tmp)
        
        pr_trend_emus = np.array(Parallel(n_jobs = 10)(delayed(parallel_predictions)(i_emu) for i_emu in tqdm(range(n_emus), total = n_emus) ) )

        # store trend predictions
        precip_emu_path   = cset.output_path.joinpath(f'glm_emus_v02/emus/pr/loc_trend/{model_id}/')
        Path(precip_emu_path).mkdir(parents = True, exist_ok = True)
        joblib.dump(pr_trend_emus/86400, precip_emu_path.joinpath(f'pr_full-trend-emu_{model_id}_{ssp_id}_{run_id_training}_v0.pkl'))
        # make full predictions
        precip_emu_path   = cset.output_path.joinpath(f'glm_emus_v02/emus/pr/loc_var/{model_id}/')
        new_samples       = joblib.load(precip_emu_path.joinpath(f'pr_var_{model_id}_{run_id_training}_v0.pkl'))[:n_emus, :, :]

        # store full predictions 
        pr_emus = pr_trend_emus*np.exp(new_samples)

        precip_emu_path   = cset.output_path.joinpath(f'glm_emus_v02/emus/pr/full/{model_id}/')
        Path(precip_emu_path).mkdir(parents = True, exist_ok = True)
        joblib.dump(pr_emus/86400, precip_emu_path.joinpath(f'pr_full-emus_{model_id}_{ssp_id}_{run_id_training}_v0.pkl'))

        def get_health(m, i):
            pr_max = np.max(pr_emus[:, m::12, i])
            pr_q99 = np.quantile(pr_emus[:, m::12, i], q = .995)

            if pr_max > 5*pr_q99: 
                return(True)
            else:
                return(False)

        fail_binary = Parallel(n_jobs = -1)(delayed(get_health)(m_, i_) for m_, i_ in tqdm(ccon.mi_ind, total = len(ccon.mi_ind)))
        idx_fail    = np.array(ccon.mi_ind)[fail_binary]

        # correct extremely high values: 
        pr_emus_v02 = np.copy(pr_emus)
        for m_, i_ in idx_fail: 
            pr_max = np.max(pr_emus[:, m_::12, i_])
            pr_q99 = np.quantile(pr_emus[:, m_::12, i_], q = .995)
            
            scaling_factor = pr_q99/pr_max
            
            pr_tmp                  = pr_emus_v02[:, m_::12, i_]
            pr_tmp[pr_tmp > pr_q99] *= scaling_factor
            
            pr_emus_v02[:, m_::12, i_] = pr_tmp

        joblib.dump(pr_emus_v02/86400, precip_emu_path.joinpath(f'pr_full-emus_{model_id}_{ssp_id}_{run_id_training}_v1.pkl'))
