
#%%
'''
Script generates a dataframe containing global mean temperatures & lowess smoothed global mean 
temperatures for every ensemble member of each model and all of its available scenarios 
'''

# setting paths correctly 
import os
import sys
os.chdir('/home/ubuntu/sarah/files/mesmer-m-tp-publication/')
sys.path.append('/home/ubuntu/sarah/files/mesmer-m-tp-publication/')

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

from utils.load_data import load_and_prepare_land_dataset, load_land_gmt_dataset, load_gmt_dataset
from train.gmt_trend import Tglob_lowess_smoothed

#%%

if __name__ == '__main__': 
    for model_id in tqdm(cset.model_ids[2:], total = cset.n_models):
        # extract all run_ids (only runs with historical period for tas and precip are consiered)
        tas_runs        = [f.split('_')[4] for f in os.listdir(ccon.tas_path) if  (model_id in f and 'historical' in f)]
        pr_runs         = [f.split('_')[4] for f in os.listdir(ccon.precip_path) if  (model_id in f and 'historical' in f)]
        
        combined_runs   = list(set(tas_runs).intersection(pr_runs))
        n_runs          = len(combined_runs)  
        
        all_gmt_ds      = xr.Dataset()
        
        for run_id in tqdm(combined_runs, total = n_runs): 
            run_names   = [f'historical_{run_id}'] + [f'{ssp_id}_{run_id}' for ssp_id in cset.ssp_ids if ccon.tas_path.joinpath(f'tas_mon_{model_id}_{ssp_id}_{run_id}_g025.nc').is_file()]    
            gmt_ds      = load_gmt_dataset(model_id, run_names) 
            
            gmt_list = []
            for run_name in run_names:
                ssp_id   = run_name.split('_')[0]
                run_id   = run_name.split('_')[1]

                a        = Tglob_lowess_smoothed(gmt_ds['rel_gmt'].sel(run_id = run_id,  ssp_id = ssp_id))
                a.name   = 'rel_gmt_lowess'
                
                gmt_list.append(a.expand_dims(dim = {'run_id': [run_id], 'ssp_id': [ssp_id]}).load())
                
            gmt_trend_ds = xr.merge(gmt_list).load()
            
            gmt_ds['rel_gmt_lowess'] =  gmt_trend_ds['rel_gmt_lowess']
            
            all_gmt_ds  = xr.merge([all_gmt_ds, gmt_ds]).load()
        
        out_path = f'{cset.output_path}/processed_CMIP6/{model_id}/'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        all_gmt_ds.to_netcdf(out_path + f'{model_id}_GMT-LO.nc')
 
#%%        

# plt.figure()
# plt.plot(all_gmt_ds['rel_gmt'].sel(ssp_id = 'historical', run_id = 'r1i1p1f1').values)
# plt.plot(all_gmt_ds['rel_gmt_trend'].sel(ssp_id = 'historical', run_id = 'r1i1p1f1').values)
# plt.show()
