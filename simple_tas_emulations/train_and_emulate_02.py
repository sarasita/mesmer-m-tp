
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

from utils.prepare_input import get_input_data

#%%
# response module
from sklearn.linear_model import LinearRegression

# variability module 
from sklearn.preprocessing import PowerTransformer

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
            pt         = PowerTransformer(standardize = True).fit(tas_res) 
            tas_res_pt = pt.transform(tas_res)

            # estimate non-stationarity & rescale 
            n_years  = int(np.shape(tas_res_pt)[0]/12)
            i_sort   = np.argsort(GMT_trend_training)
            GMT_sort = np.sort(GMT_trend_training)

            def fit_var(m,i):
                tas_sort     = tas_res_pt[m::12, i][i_sort]
                len_interval = 25
                n_intervals  = int(n_years/len_interval)
                mod          = int(n_years%len_interval)
                
                stds     = []
                gmts     = []
                for k in range(n_intervals): 
                    gmts.append(np.mean(GMT_sort[mod+k*len_interval:mod+(k+1)*len_interval]))
                    stds.append(np.std(tas_sort[mod+k*len_interval:mod+(k+1)*len_interval]))
                
                c_fit = np.polyfit(gmts, stds, deg = 1)
                return(c_fit)

            from joblib import Parallel, delayed

            linear_params_pt  = np.array(Parallel(n_jobs = -1)(delayed(fit_var)(m_, i_) for m_, i_ in tqdm(ccon.mi_ind, total = len(ccon.mi_ind)))).reshape(12, ccon.n_sindex, 2)
            VARS              = np.moveaxis(np.array([c_fit[0]*GMT_trend_training+c_fit[1] for c_fit in linear_params_pt.reshape(-1, 2)]).reshape(12, ccon.n_sindex, -1), [0,1,2], [1,2,0])
            print(VARS.min())
            VARS[VARS < 0.4] = 0.4

            tas_res_rescaled = tas_res_pt/VARS.reshape(-1, 2652)
            
            from scipy.optimize import curve_fit

            def lin_func(x, a, b):
                return(a * x + b)

            def execute_bounded_linfit(x,y):
                return(curve_fit(lin_func, x, y, bounds=([-1,-np.inf], [1, np.inf]))[0])
            
            combi_res         = tas_res_rescaled
            n_components      = ccon.n_sindex

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

            ssp_id   = 'ssp585'
            n_emus   = 100 
            n_years  = 251
            n_buffer = 10

            emu_innovs_pt  = np.zeros((n_emus, (n_years+n_buffer)*12, n_components))  
            for m in tqdm(range(12), total = 12):
                emu_innovs_pt[:, m::12, :] = multivariate_normal(mean = np.zeros((n_components)), cov = covs[m], size = (n_emus, n_years + n_buffer))

            emu_var_pt = np.copy(emu_innovs_pt)
            for t in range(12, (n_years + n_buffer)*12):
                emu_var_pt[:, t, :] = combi_params[t%12, :, 0]*emu_var_pt[:, t-1, :] + combi_params[t%12, :, 1] + emu_innovs_pt[:, t, :]

            vars_adj             = np.zeros((n_years, 12, 2652))
            vars_adj[:165, :, :] = VARS[:165, :, :]
            vars_adj[-86:, :, :] = VARS[-86:, :, :]
            vars_adj = vars_adj.reshape(-1, 2652)

            emu_var  = np.array([pt.inverse_transform(emu_var_pt[i_emu, :, :][10*12:, :]*vars_adj) for i_emu in tqdm(range(n_emus), total = n_emus)])
                   
            ssp_id = 'ssp585'
            GMT_ds            = xr.open_dataset(Path.joinpath(cset.output_path, f'processed_CMIP6/{model_id}/{model_id}_GMT-LO_processed.nc'))

            GMT_emus                       = joblib.load(cset.output_path.joinpath(f'gmt_emus_LO/full/{model_id}/GMT-emus_{model_id}_{ssp_id}_{run_id_training}.pkl'))
            GMT_emus_trend                 = np.zeros(251)
            GMT_emus_trend[:ccon.n_hist_years]  = GMT_ds['rel_gmt_trend'].sel(run_id = run_id_training, ssp_id = 'historical').dropna(dim = 'year').values
            GMT_emus_trend[ccon.n_hist_years:]  = GMT_ds['rel_gmt_trend'].sel(run_id = run_id_training, ssp_id = ssp_id).dropna(dim = 'year').values
            GMT_emus_var                   = np.array([GMT_emus[i_emu, :]-GMT_emus_trend for i_emu in range(n_emus)])      
            
            tas_emus_full = np.array([LinReg.predict(np.array([GMT_emus_trend, GMT_emus_var[i_emu, :]]).T).reshape(-1, n_comp) + emu_var[i_emu, :, :] for i_emu in tqdm(range(n_emus), total = n_emus)])

            emu_path   = f'{cset.output_path}/simple_tas_emus/full_emu/{model_id}/'
    
            Path(emu_path).mkdir(parents = True, exist_ok = True)
            joblib.dump(tas_emus_full, f'{emu_path}tas_full-emus_{model_id}_{ssp_id}_{run_id_training}_v1.pkl')
            # joblib.dump(trend_emu[:, :, :n_index], f'{emu_path}tas_trend-emus_{model_id}_{ssp_id}_{run_id_training}_v1.pkl')
                        
