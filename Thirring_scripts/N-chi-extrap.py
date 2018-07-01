import numpy as np
import pandas as pd
from scipy import optimize
import warnings
import matplotlib.pyplot as plt

def write_chi_extrap_table(N, g_idx, ma, chis):
    E0 = []; mg = []; ccs = []
    for chi in chis:
        path = '/home/davidtan/Desktop/MPS_config/XXZ/chi_extrap'
        try:       
            E0.append(np.load(path+'/E-N_{}-ma_{}-chi_{}.npy'.format(N,ma,chi))[g_idx])        
        except:
            E0.append(None)
        try:
            E0_value = np.load(path+'/E-N_{}-ma_{}-chi_{}.npy'.format(N,ma,chi))[g_idx]
            E1_value = np.load(path+'/E1-N_{}-ma_{}-chi_{}.npy'.format(N,ma,chi))[g_idx]
            mg.append(abs(E1_value-E0_value)*N)
        except:    
            mg.append(None)
        try:
            ccs.append(np.load(path+'/ccs-N_{}-ma_{}-chi_{}.npy'.format(N,ma,chi))[g_idx])
        except:    
            ccs.append(None)
     
    dict = {"chi": chis, "E0": E0, "mg": mg, "ccs": ccs}    
    df = pd.DataFrame(dict)
    chis += ['inf-mean','inf-std']
    E0 += inf_bond_fit(df,'E0')
    mg += inf_bond_fit(df,'mg')
    ccs += inf_bond_fit(df,'ccs')
    dict = {"chi": chis, "E0": E0, "mg": mg, "ccs": ccs}    
    df = pd.DataFrame(dict)
    return df

def inf_bond_fit(df,key):
    idx = df.apply(pd.Series.last_valid_index)[key]
    if np.isnan(idx) or idx < 1:
        inf_mean = np.nan
        inf_std = np.nan
    else:             
        idx = int(idx)        
        inv_chi = 1./(df['chi'].values[idx-1:idx+1])
        points = df[key].values[idx-1:idx+1]
        if np.sum(np.isnan(points)):
            inf_mean = np.nan
            inf_std = np.nan
        else:
            m,b = np.polyfit(inv_chi,points,1)
            inf_mean = 0.5*(points[-1]+b)
            inf_std = 0.5*(points[-1]-b)
        if inf_std < 1e-8: inf_std = 1e-7
    return [inf_mean,inf_std]

def read_chi_extrap_table(path,key):
    df = pd.read_csv(path)
    #idx = int(df.apply(pd.Series.last_valid_index)[key])
    key_values = df[key].values
    return key_values

def write_N_extrap_table(Ns, g_idx, ma):
    E0_mean = []; mg_mean = []; ccs_mean = []
    E0_std = []; mg_std = []; ccs_std = []
    for N in Ns:
        path = '/home/davidtan/Desktop/MPS_config/XXZ/chi_extrap/table'
        try:
            inf_mean, inf_std = read_chi_extrap_table(path+'/chi_extrap-N_{}-g_{}-ma_{}.csv'.format(N,gs[g_idx],ma),'E0')[-2:]
            E0_mean.append(inf_mean); E0_std.append(inf_std)
        except:
            E0_mean.append(np.nan); E0_std.append(np.nan)
        try:
            inf_mean, inf_std = read_chi_extrap_table(path+'/chi_extrap-N_{}-g_{}-ma_{}.csv'.format(N,gs[g_idx],ma),'mg')[-2:]
            mg_mean.append(inf_mean); mg_std.append(inf_std)
        except:
            mg_mean.append(np.nan); mg_std.append(np.nan)
        try:
            inf_mean, inf_std = read_chi_extrap_table(path+'/chi_extrap-N_{}-g_{}-ma_{}.csv'.format(N,gs[g_idx],ma),'ccs')[-2:]
            ccs_mean.append(inf_mean); ccs_std.append(inf_std)
        except:
            ccs_mean.append(np.nan); ccs_std.append(np.nan)
            
    dict = {"N": Ns, "E0_mean": E0_mean, "E0_std": E0_std,
                     "mg_mean": mg_mean, "mg_std": mg_std,
                     "ccs_mean": ccs_mean, "ccs_std": ccs_std}
    df = pd.DataFrame(dict)
    Ns += ['inf']
    out = inf_N_fit(df,'E0')
    E0_mean.append(out[0]); E0_std.append(out[1])
    out = inf_N_fit(df,'mg')
    mg_mean.append(out[0]); mg_std.append(out[1])
    out = inf_N_fit(df,'ccs')
    ccs_mean.append(out[0]); ccs_std.append(out[1])
    dict = {"N": Ns, "E0_mean": E0_mean, "E0_std": E0_std,
                     "mg_mean": mg_mean, "mg_std": mg_std,
                     "ccs_mean": ccs_mean, "ccs_std": ccs_std} 
    df = pd.DataFrame(dict)
    return df

def inf_N_fit(df,key):
    Ns = df['N'].values
    mean = df[key+'_mean'].values
    std = df[key+'_std'].values   
    if np.isnan(sum(mean)):
        inf_mean = np.nan
        inf_std = np.nan
    else:
        fitfunc = lambda p, x: p[0] + p[1] * x
        errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
        p_init = [1.0, 1.0]
        out = optimize.leastsq(errfunc, p_init, args=(1./np.array(Ns), mean, std), full_output=1)

        p_final = out[0]
        covar = out[1]
        inf_mean = p_final[0]
        if covar is not None:
            inf_std = np.sqrt(covar[0][0])
        else:
            warnings.simplefilter("always")
            warnings.warn("ValueWarning: Encounter singular matrix in scipy.optimize.leastsq. Try to fit without error estimation.")
            coeff = np.polyfit(1./np.array(Ns),mean,1)
            inf_mean = coeff[1]
            inf_std = 1e-7
    return [inf_mean, inf_std]

def read_N_extrap_table(path,key):
    df = pd.read_csv(path)
    #idx = int(df.apply(pd.Series.last_valid_index)[key])
    key_values = df[key].values
    return key_values

if __name__=='__main__':
    
    N = 1000
    Ns = np.linspace(400,1000,4,dtype=int).tolist()
    gs = np.linspace(-1.0,1.0,101)
    gs = np.array(gs.tolist()[5:20]+gs.tolist()[20::10])
    mas = [0.0,0.1,0.2,0.3,0.4]
    #pd.options.display.float_format = "{:.12f}".format
    
    gamma = (np.pi-gs)/2
    Zs = 2*gamma/np.sin(gamma)/np.pi
    
    for N in Ns:
        for ma in mas:
            for g_idx in xrange(len(gs)):
                chis = np.linspace(300,600,4,dtype=int).tolist()
                path = '/home/davidtan/Desktop/MPS_config/XXZ/chi_extrap/table'
                df = write_chi_extrap_table(N, g_idx, ma, chis)
                df.to_csv(path+'/chi_extrap-N_{}-g_{}-ma_{}.csv'.format(N,gs[g_idx],ma))
                print gs[g_idx],ma,'\n',df   
    
    for ma in mas:
        for g_idx in xrange(len(gs)):
            Ns = np.linspace(400,1000,4,dtype=int).tolist()
            path = '/home/davidtan/Desktop/MPS_config/XXZ/N_extrap'
            df = write_N_extrap_table(Ns, g_idx, ma)
            df.to_csv(path+'/N_extrap-g_{}-ma_{}.csv'.format(gs[g_idx],ma))
            print gs[g_idx],ma,'\n',df
            
            
