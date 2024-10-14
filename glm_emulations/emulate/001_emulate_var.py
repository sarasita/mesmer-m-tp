"""
Generating 100 variability realisations for precipitation 
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

from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# takes approx 4 mins for 100 emulations for each model
if __name__ == '__main__':
    for model_id in cset.model_ids:
        print(model_id)
        run_id_training        = cset.model_training_mapping[model_id]
        print('Opening calibration parameters')
        calib_path_var         = cset.output_path.joinpath(f'glm_emus_v02/calib/loc_var/{model_id}')
        res_pr_StdPca_pipeline = joblib.load(calib_path_var.joinpath(f'ResPrStdPcaPipeline_{model_id}_{run_id_training}.pkl'))
        kde_list               = joblib.load(calib_path_var.joinpath(f'KDEs_{model_id}_{run_id_training}.pkl'))

        print('Sampling variability')
        n_emus           = 100
        n_samples        = 251 
        n_components     = res_pr_StdPca_pipeline['pca_transform'].n_components_
        
        new_samples_pca = np.zeros((n_emus, n_samples*12, n_components))
        for m in tqdm(range(12), total = 12):         
            kde_m                        = kde_list[m]
            new_samples_pca[:, m::12, :] = kde_m.sample((n_emus, n_samples))
            
        new_samples = np.array([res_pr_StdPca_pipeline.inverse_transform(new_samples_pca[i_run, :, :]) for i_run in range(n_emus)])

        precip_emu_path   = cset.output_path.joinpath(f'glm_emus_v02/emus/pr/loc_var/{model_id}/')
        Path(precip_emu_path).mkdir(parents = True, exist_ok = True)
        joblib.dump(new_samples, precip_emu_path.joinpath(f'pr_var_{model_id}_{run_id_training}_v0.pkl'))