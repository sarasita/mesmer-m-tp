from pathlib import Path

data_path           = Path('/mnt/CMIP6_storage/cmip6-ng/')
precip_path         = data_path / 'pr' / 'mon' / 'g025'
tas_path            = data_path / 'tas' / 'mon' / 'g025'

code_path           = Path('/home/ubuntu/sarah/files/mesmer-m-tp-publication/')    
aod_file            = code_path / 'data' / 'isoaod_gl.txt' # path to aerosol optical depth data
coords_file         = code_path / 'data' / 'ew_coords.npy' # path to coordinate array

emu_vars            = ['tas', 'pr']
n_sindex            = 2652
n_months            = 12
ref_period          = [1850, 1900]
n_hist_years        = 165
n_ssp_years         = 86
n_ssps              = 5 
n_closest           = 150 

mi_ind              = [(m, i) for m in range(12) for i in range(n_sindex)]   