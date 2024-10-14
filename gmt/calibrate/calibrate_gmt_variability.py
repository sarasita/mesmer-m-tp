#%%
'''
Script opens the gmt output from calibrate_gmt_volcanic and approximates the global mean variability
of an autoregressive process where the number of lags is chosen using the AIC. The AR parameters are 
stored. 
'''

# setting paths correctly 
import os
import sys
os.chdir('/home/ubuntu/sarah/files/mesmer-m-tp-dev/')
sys.path.append('/home/ubuntu/sarah/files/mesmer-m-tp-dev/')

# computation
import xarray as xr
import numpy as np
import joblib
from joblib              import Parallel, delayed

# utils
from tqdm   import tqdm 
from pathlib import Path

# plotting
import matplotlib.pyplot as plt
import cartopy.crs       as ccrs

# setting paths such that execution works from console & interactive shell 
import config.settings as cset
import config.constants as ccon

from utils.load_data import load_and_prepare_land_dataset, load_land_gmt_dataset
from train.gmt_var   import Tglob_AR_parameters


#%%

if __name__ == '__main__':
    # --- add variability approx to GMT_ds dataframe
    for model_id in tqdm(cset.model_ids[:], total = cset.n_models):
        run_id_training         = cset.model_training_mapping[model_id]

        GMT_ds                  = xr.open_dataset(Path.joinpath(cset.output_path, f'processed_CMIP6/{model_id}/{model_id}_GMT-LO_processed.nc'))

        # training happens on one ensemble member for a specfic ssp_id
        c_ssp_ids       = GMT_ds['rel_gmt'].sel(run_id = run_id_training).dropna(dim = 'year', how = 'all').ssp_id.values
        n_c_ssp_ids     = len(c_ssp_ids)-1 
        n_years         = ccon.n_hist_years + n_c_ssp_ids*86
        
        # estimate AR parameters
        for ssp_id in c_ssp_ids[1:]: 
            GMT_var                     = np.zeros((ccon.n_hist_years + ccon.n_ssp_years))
            GMT_var[:ccon.n_hist_years] = GMT_ds['gmt_var'].sel(run_id = run_id_training, ssp_id = 'historical').dropna(dim = 'year').values
            GMT_var[ccon.n_hist_years:] = GMT_ds['gmt_var'].sel(run_id = run_id_training, ssp_id =  ssp_id).dropna(dim = 'year').values
            
            GMT_sub_ds                      = xr.Dataset( data_vars = {'gmt_var': (['year'], GMT_var)}, coords = {'year': GMT_ds.year} )
            AR_lags, AR_params, AR_sigma    = Tglob_AR_parameters(GMT_sub_ds)
            if not AR_lags:
                AR_lags = [0]
            
            AR_results   = {'AR_lags': AR_lags, 'AR_params': AR_params, 'AR_sigma': AR_sigma}

            storage_path = Path.joinpath(cset.output_path, f'calib/GMT-LO/GMT_var/{model_id}/')
            storage_path.mkdir(parents = True, exist_ok = True)
            joblib.dump(AR_results, Path.joinpath(storage_path, f'AR-calib_{model_id}_{ssp_id}_{run_id_training}.pkl'))


# %%
