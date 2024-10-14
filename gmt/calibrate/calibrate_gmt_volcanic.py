#%%

'''
Script opens dataframe containing global mean temperatures & lowess smoothed global mean 
temperatures for every ensemble member of each model & its available scenarios and generates the 
temperature response to volcanic activity. The resposne is approximated as a linear model and the
results are stored as a processed dataframe 
'''

# setting paths correctly 
import os
import sys
os.chdir('/home/ubuntu/sarah/files/mesmer-m-tp-dev/')
sys.path.append('/home/ubuntu/sarah/files/mesmer-m-tp-dev/')

# computation
import xarray as xr
import numpy as np
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
from train.gmt_trend import historic_aod, Tglob_volcanic

import warnings 

#%%

# compute linear relationship between temperature and historic_aod levels for each model 
# and add to dataframe

if __name__ == '__main__':
    # --- add volcanic approx and variability to GMT_ds dataframe
    for model_id in tqdm(cset.model_ids[:], total = cset.n_models):
        try: 
            warnings.filterwarnings("ignore")
            run_id_training     = cset.model_training_mapping[model_id]
            GMT_ds              = xr.open_dataset(Path.joinpath(cset.output_path, f'processed_CMIP6/{model_id}/{model_id}_GMT-LO.nc'))
            # GMT_ds              = xr.open_dataset(Path.joinpath(cset.output_path, f'processed_CMIP6/{model_id}/{model_id}_GMT.nc'))
            # GLMT_ds = xr.open_dataset(Path.joinpath(cset.output_path, f'processed_CMIP6/{model_id}/{model_id}_GMT.nc'))
            
            # adding parameters for volcanic data & training parameters 
            da_aod                  = historic_aod()
            Tglob_volc, Tglob_coefs = Tglob_volcanic((GMT_ds['rel_gmt']-GMT_ds['rel_gmt_lowess']).sel(run_id = run_id_training, ssp_id = 'historical')[:165], da_aod[:165])
            # adding volcanic respoonse as a column to the dataframe 
            Tglob_volcanic_ext      = np.zeros_like(GMT_ds['rel_gmt'].values)
            Tglob_volcanic_ext[:, :, :ccon.n_hist_years] = Tglob_volc
            # multiplying zero by GMT_ds['rel_gmt'] to generate a timeseries of 0 that has nans
            # at the correct positions to form a frame for the volcanic data 
            GMT_ds['rel_gmt_volc']    = GMT_ds['rel_gmt']*0 + Tglob_volcanic_ext
            GMT_ds['volc_params']     = xr.DataArray(data = Tglob_coefs, dims = ['param_type'], coords = dict(param_type = ['coefficient', 'intercept'])).expand_dims(dim = dict(run_id = [run_id_training]), axis = 0)

            GMT_ds['rel_gmt_trend']   = GMT_ds['rel_gmt_lowess'] + GMT_ds['rel_gmt_volc']

            GMT_ds['gmt_var']         = GMT_ds['rel_gmt'] - GMT_ds['rel_gmt_trend']
            
            GMT_ds.to_netcdf(Path.joinpath(cset.output_path, f'processed_CMIP6/{model_id}/{model_id}_GMT-LO_processed.nc'))
            warnings.filterwarnings("default")
        except:
            print('Didn\'t work for model: ', model_id)

        
# %%
