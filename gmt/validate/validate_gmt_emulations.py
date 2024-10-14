#%%

'''
Script for quickly visually verifiyng synthetic GMT trajectories against actual 
ESM data. 
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
import matplotlib as mpl
import seaborn as sns

# setting paths such that execution works from console & interactive shell 
import config.settings as cset
import config.constants as ccon

from utils.load_data import load_and_prepare_land_dataset, load_land_gmt_dataset

# verify  GMT emulations

if __name__ == '__main__':
    for model_id in cset.model_ids_veri:
            run_id_training = cset.model_training_mapping[model_id]
            GMT_ds          = xr.open_dataset(cset.output_path.joinpath(f'processed_CMIP6/{model_id}/{model_id}_GMT-LO_processed.nc'))

            c_ssp_ids       = GMT_ds['rel_gmt'].sel(run_id = run_id_training).dropna(dim = 'year', how = 'all').ssp_id.values
            n_c_ssp_ids     = len(c_ssp_ids)-1 
            n_years         = 165 + n_c_ssp_ids*86
            
            storage_path = cset.output_path.joinpath(f'gmt_emus_LO/full/{model_id}/')
                    
            for ssp_id in c_ssp_ids[1:]: 
                GMT                = np.zeros((ccon.n_hist_years + ccon.n_ssp_years))
                GMT[:ccon.n_hist_years] = GMT_ds['rel_gmt'].sel(run_id = run_id_training, ssp_id = 'historical').dropna(dim = 'year').values
                GMT[ccon.n_hist_years:] = GMT_ds['rel_gmt'].sel(run_id = run_id_training, ssp_id =  ssp_id).dropna(dim = 'year').values
                
                GMT_emus           = joblib.load(storage_path.joinpath(f'GMT-emus_{model_id}_{ssp_id}_{run_id_training}.pkl'))
                
                plt.figure()
                with sns.color_palette("Blues", n_colors=100):
                    plt.plot(GMT_emus.T)
                plt.plot(GMT, color = 'red')
                plt.show()

                esm_run_ids           = GMT_ds['rel_gmt'].sel(ssp_id =  ssp_id).dropna(dim = 'run_id', how = 'all').dropna(dim = 'year').run_id
                n_esm                 = len(esm_run_ids)
                GMT                   = np.zeros((n_esm, ccon.n_hist_years + ccon.n_ssp_years))
                GMT[:, :ccon.n_hist_years] = GMT_ds['rel_gmt'].sel(ssp_id = 'historical', run_id  = esm_run_ids).dropna(dim = 'year').values
                GMT[:, ccon.n_hist_years:] = GMT_ds['rel_gmt'].sel(ssp_id =  ssp_id, run_id = esm_run_ids).dropna(dim = 'year').values
                
                plt.figure()
                with sns.color_palette("Oranges", n_colors=n_esm):
                    plt.plot(GMT.T)
                plt.show()
                
                plt.figure()
                with sns.color_palette("Blues", n_colors=100):
                    plt.plot(GMT_emus.T)
                plt.show()
                
                
# %%
