import numpy as np

def infinite_bond_dim_extrapolation():
    return

def finite_size_scaling():
    return

def BKT_corr_fitting(x,y,form):  
    if form=='power':
        coeff=np.polyfit(np.log(x),np.log(y),1) 
    elif form=='exp':
        coeff=np.polyfit(x,np.log(y),1)
        coeff[0]=1./coeff[0]
    else:
        raise ValueError("Only exp-law and power-law are supported.")
    return -coeff[0],coeff[1]   
