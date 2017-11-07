import numpy as np

def BKT_corr(x,y,form):  
    if form=='power':
        coeff=np.polyfit(np.log(x),np.log(y),1) 
    elif form=='exp':
        coeff=np.polyfit(x,np.log(y),1)
        coeff[0]=1./coeff[0]
    else:
        raise ValueError("Only exp-law and power-law are supported.")
    return -coeff[0],coeff[1]   

def central_charge(N,bonds,entros):
    log_bonds=np.log(N/np.pi*np.sin(np.pi/N*bonds))
    coeff=np.polyfit(log_bonds,entros,1)
    return 6.*coeff[0],coeff[1]
