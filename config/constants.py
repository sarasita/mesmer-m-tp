from pathlib import Path

data_path           = Path('/mnt/CMIP6_storage/cmip6-ng/')
precip_path         = Path('/mnt/CMIP6_storage/cmip6-ng/pr/mon/g025/')
tas_path            = Path('/mnt/CMIP6_storage/cmip6-ng/tas/mon/g025')

code_path           = Path('/home/ubuntu/sarah/files/mesmer-m-tp-publication/')    

 
emu_vars            = ['tas', 'pr']
n_sindex            = 2652
n_months            = 12
ref_period          = [1850, 1900]
n_hist_years        = 165
n_ssp_years         = 86
n_ssps              = 5 
n_closest           = 150 

mi_ind              = [(m, i) for m in range(12) for i in range(n_sindex)]   