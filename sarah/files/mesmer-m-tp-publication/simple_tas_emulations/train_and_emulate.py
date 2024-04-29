
#%%
"""
Simple routine for generating temperature emulations.
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

#%%

# parallelizing
from joblib import Parallel, delayed

# response module
from sklearn.linear_model import LinearRegression

# variability module 
# - mapping to normal distribution 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
# - fitting AR process
from scipy.optimize import curve_fit
def lin_func(x, a, b):
    return(a * x + b)
def execute_bounded_linfit(x,y):
    return(curve_fit(lin_func, x, y, bounds=([-1,-np.inf], [1, np.inf]))[0])


#%%

if __name__ == '__main__':
    # for model_id in tqdm(cset.model_ids[1:], total = cset.n_models):
    for model_id in ['CanESM5', 'MPI-ESM1-2-LR']:
            run_id_training = cset.model_training_mapping[model_id]
            # load training data
            tas_training, _, GMT_training, GMT_trend_training, GMT_var_training = get_input_data(model_id, run_id_training = run_id_training)
            
            # TREND
            # approximate as linear model 
            n_comp      = ccon.n_sindex
            LinReg      = LinearRegression().fit(np.array([GMT_trend_training, GMT_var_training]).T, tas_training.reshape(-1, 12*n_comp))
            trend_fits  = LinReg.predict(np.array([GMT_trend_training, GMT_var_training]).T).reshape(-1, n_comp)
            tas_res     = tas_training - trend_fits
            
            # RESIDUALS 
            std          = StandardScaler().fit(tas_res)
            tas_res_std  = std.transform(tas_res) 
            pca          = PCA().fit(tas_res_std)
            tas_res_pca  = pca.transform(tas_res_std)
            n_components = pca.n_components_
            
            pt          = PowerTransformer(standardize = True).fit(tas_res_pca) 
            tas_res_pt  = pt.transform(tas_res_pca)
            # estimate non-stationarity & rescale 
            n_years     = int(np.shape(tas_res_pt)[0]/12)

            tas_res_rescaled  = tas_res_pt
            combi_res         = tas_res_rescaled
            
            combi_params      = np.array(Parallel(n_jobs=-1)(delayed(execute_bounded_linfit)(combi_res[11+m:-1:12, i], combi_res[12+m::12, i]) for m in tqdm(range(12), desc = 'Fitting AR time coefficients', position = 0, leave = True, file=sys.stdout) for i in range(n_components))).reshape(12, n_components, 2)

            combi_ar_time_fit = np.zeros((n_years-1, 12, n_components))

            for m in range(12):
                combi_ar_time_fit[:,m,:] = combi_params[m, :, 0]*combi_res[11+m:-1:12,:] + combi_params[m,:,1]

            combi_spatial_res =  np.zeros((12*(n_years- 1), n_components))

            for m in range(12):
                combi_spatial_res[m::12,:] = combi_res[12+m::12,:] - combi_ar_time_fit[:,m,:]
            
            from numpy.random import multivariate_normal

            covs              = [] 
            for m in tqdm(range(12), total = 12):
                cov                       = np.cov(combi_spatial_res[m::12,:].T)
                covs.append(cov)
                
            # generate emulations

            ssp_id   = 'ssp585'
            n_emus   = 100 
            n_years  = 251
            n_buffer = 10

            # - variability / residuals 
            emu_innovs_pt  = np.zeros((n_emus, (n_years+n_buffer)*12, n_components))  
            for m in tqdm(range(12), total = 12):
                emu_innovs_pt[:, m::12, :] = multivariate_normal(mean = np.zeros((n_components)), cov = covs[m], size = (n_emus, n_years + n_buffer))

            emu_var_pt = np.copy(emu_innovs_pt)
            for t in range(12, (n_years + n_buffer)*12):
                emu_var_pt[:, t, :] = combi_params[t%12, :, 0]*emu_var_pt[:, t-1, :] + combi_params[t%12, :, 1] + emu_innovs_pt[:, t, :]

            emu_var  = np.array([std.inverse_transform(pca.inverse_transform(pt.inverse_transform(emu_var_pt[i_emu, :, :][10*12:, :]))) for i_emu in tqdm(range(n_emus), total = n_emus)])
                
            # - linear response 
            GMT_ds                              = xr.open_dataset(Path.joinpath(cset.output_path, f'processed_CMIP6/{model_id}/{model_id}_GMT-LO_processed.nc'))

            GMT_emus                            = joblib.load(cset.output_path.joinpath(f'gmt_emus_LO/full/{model_id}/GMT-emus_{model_id}_{ssp_id}_{run_id_training}.pkl'))
            GMT_emus_trend                      = np.zeros(251)
            GMT_emus_trend[:ccon.n_hist_years]  = GMT_ds['rel_gmt_trend'].sel(run_id = run_id_training, ssp_id = 'historical').dropna(dim = 'year').values
            GMT_emus_trend[ccon.n_hist_years:]  = GMT_ds['rel_gmt_trend'].sel(run_id = run_id_training, ssp_id = ssp_id).dropna(dim = 'year').values
            GMT_emus_var                        = np.array([GMT_emus[i_emu, :]-GMT_emus_trend for i_emu in range(n_emus)])      
            
            tas_emus_full = np.array([LinReg.predict(np.array([GMT_emus_trend, GMT_emus_var[i_emu, :]]).T).reshape(-1, n_comp) + emu_var[i_emu, :, :] for i_emu in tqdm(range(n_emus), total = n_emus)])

            emu_path      = f'{cset.output_path}/simple_tas_emus/full_emu/{model_id}/'

            Path(emu_path).mkdir(parents = True, exist_ok = True)
            joblib.dump(tas_emus_full, f'{emu_path}tas_full-emus_{model_id}_{ssp_id}_{run_id_training}_v1.pkl')
            
            
# #%%

# tas_ssp_emus    = np.load(f'/mnt/PROVIDE/mesmer-m-tp-dev/processed_CMIP6/{model_id}/tas-ssp-all_{model_id}.npy')[:, -1, :, :]
# tas_hist_emus   = np.load(f'/mnt/PROVIDE/mesmer-m-tp-dev/processed_CMIP6/{model_id}/tas-hist-all_{model_id}.npy')
# tas_cmip        = np.concatenate((tas_hist_emus, tas_ssp_emus), axis = 1)
# tas_cmip        = tas_cmip[~np.isnan(tas_cmip[:, -1, 0]), :, :]
# del tas_ssp_emus, tas_hist_emus