import numpy as np
import pandas as pd
from scipy import optimize
import warnings,itertools
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
        errfunc = lambda p, x, y, err: ((y - fitfunc(p, x)) / err) ** 2 
        p_init = [1.0, 1.0]
        out = optimize.leastsq(errfunc, p_init, args=(1./np.array(Ns), mean, std), full_output=1)

        p_final = out[0]
        covar = out[1]
        inf_mean = p_final[0]
        chi2 = np.sum(errfunc(p_final,1./np.array(Ns),mean,std))
        if covar is not None:
            inf_std = np.sqrt(np.linalg.norm(covar))
        else:
            warnings.simplefilter("always")
            warnings.warn("ValueWarning: Encounter singular matrix in scipy.optimize.leastsq. Try to fit without error estimation.")
            coeff = np.polyfit(1./np.array(Ns),mean,1)
            inf_mean = coeff[1]
            inf_std = 1e-7
        if key == 'ccs' and chi2 > 1.:
            inf_std = np.sqrt(chi2/2.) * inf_std
    return [inf_mean, inf_std]

def read_N_extrap_table(path,key):
    df = pd.read_csv(path)
    #idx = int(df.apply(pd.Series.last_valid_index)[key])
    key_values = df[key].values
    return key_values

def write_full_extrap_table(gs,mas):
    E0_mean = []; E0_std = []
    ccs_mean = []; ccs_std = []
    mg_mean = []; mg_std = []
    for g in gs:
        for ma in mas:
            path = '/home/davidtan/Desktop/MPS_config/XXZ/N_extrap/'
            try:
                inf_mean = read_N_extrap_table(path+'/N_extrap-g_{}-ma_{}.csv'.format(gs[g_idx],ma),'E0_mean')[-1]
                inf_std = read_N_extrap_table(path+'/N_extrap-g_{}-ma_{}.csv'.format(gs[g_idx],ma),'E0_std')[-1]
                E0_mean.append(inf_mean); E0_std.append(inf_std)
            except:
                E0_mean.append(np.nan); E0_std.append(np.nan)
            try:
                inf_mean = read_N_extrap_table(path+'/N_extrap-g_{}-ma_{}.csv'.format(gs[g_idx],ma),'ccs_mean')[-1]
                inf_std = read_N_extrap_table(path+'/N_extrap-g_{}-ma_{}.csv'.format(gs[g_idx],ma),'ccs_std')[-1]
                ccs_mean.append(inf_mean); ccs_std.append(inf_std)
            except:
                ccs_mean.append(np.nan); ccs_std.append(np.nan)
            try:
                inf_mean = read_N_extrap_table(path+'/N_extrap-g_{}-ma_{}.csv'.format(gs[g_idx],ma),'mg_mean')[-1]
                inf_std = read_N_extrap_table(path+'/N_extrap-g_{}-ma_{}.csv'.format(gs[g_idx],ma),'mg_std')[-1]
                mg_mean.append(inf_mean); mg_std.append(inf_std)
            except:
                mg_mean.append(np.nan); mg_std.append(np.nan)    
            
    paras = np.array(list(itertools.product(gs,mas)))        
    dict = {"g": paras[:,0], "ma": paras[:,1],
                       "E0_mean": E0_mean, "E0_std": E0_std,
                       "ccs_mean": ccs_mean, "ccs_std": ccs_std,
                       "mg_mean": mg_mean, "mg_std": mg_std}
    df = pd.DataFrame(dict)
    return df

if __name__=='__main__':
    
    N = 1000
    Ns = np.linspace(400,1000,4,dtype=int).tolist()
    gs = np.linspace(-1.0,1.0,101)
    gs = np.array(gs.tolist()[5:20]+gs.tolist()[20::10])
    mas = [0.0,0.1,0.2,0.3,0.4]
    #pd.options.display.float_format = "{:.12f}".format
    ma = mas[0]; g_idx = 4
    
    gamma = (np.pi-gs)/2
    Zs = 2*gamma/np.sin(gamma)/np.pi
    
    # begin of generating chi-extrap tables
    for N in Ns:
        for ma in mas:
            for g_idx in xrange(len(gs)):
                chis = np.linspace(300,600,4,dtype=int).tolist()
                path = '/home/davidtan/Desktop/MPS_config/XXZ/chi_extrap/table'
                df = write_chi_extrap_table(N, g_idx, ma, chis)
                df.to_csv(path+'/chi_extrap-N_{}-g_{}-ma_{}.csv'.format(N,gs[g_idx],ma))
                print N,gs[g_idx],ma,'\n',df  
    # end of generating chi-extrap tables
    
    # begin of generating N-extrap tables
    for ma in mas:
        for g_idx in xrange(len(gs)):
            Ns = np.linspace(400,1000,4,dtype=int).tolist()
            path = '/home/davidtan/Desktop/MPS_config/XXZ/N_extrap'
            df = write_N_extrap_table(Ns, g_idx, ma)
            df.to_csv(path+'/N_extrap-g_{}-ma_{}.csv'.format(gs[g_idx],ma))
            print gs[g_idx],ma,'\n',df
    # end of generating N-extrap tables
    
    # begin of generating N-chi-full-extrap tables
    path = '/home/davidtan/Desktop/MPS_config/XXZ'
    df = write_full_extrap_table(gs, mas)
    df.to_csv(path+'/N-chi_extrap.csv') 
    # end of generating N-chi-full-extrap tables
    
    fontsize = 28; labelsize = 22
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    marker = itertools.cycle(('^', 'D', 'o', 's')) 
    """
    quantity = 'E0'
    path = '/home/davidtan/Desktop/MPS_config/XXZ/chi_extrap/table' 
    values = read_chi_extrap_table(path+'/chi_extrap-N_{}-g_{}-ma_{}.csv'.format(N,gs[g_idx],ma),quantity)[:4]
    
    chis = np.linspace(300,600,4,dtype=int)
    inv_chis = 1./chis
    m,b = np.polyfit(inv_chis[-2:],values[-2:],1)
    E_extrapo=0.5*(values[-1]+b) ; error=0.5*(values[-1]-b)
    x=np.linspace(0.,inv_chis[-2],100)
    
    fig = plt.figure(figsize=(12,9))
    fig.patch.set_alpha(0.0)
    plt.plot(inv_chis,values,marker="o",linestyle='-')
    plt.plot(x,m*x+b,linestyle="-",marker="None")
    plt.plot(0.0,E_extrapo,marker="o")
    plt.errorbar(0.0,E_extrapo,yerr=error,ecolor='r',elinewidth=2,capsize=5,capthick=2,linestyle='None')
    #plt.errorbar(inv_chis,mean[::-1],yerr=std,fmt='-o',elinewidth=2,capsize=8,capthick=2)
    plt.xlabel('$1/D$',fontsize=fontsize)
    plt.ylabel('$E_{0}/N$',fontsize=fontsize)
    #plt.ylabel(r'$|\langle\bar{\psi}\psi\rangle|/N$',fontsize=fontsize)
    ax=plt.gca()
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.52, 0.08, r'$\Delta(g) = {}, \tilde{{m}}_0 a = {}$'.format(gs[g_idx],ma),transform=ax.transAxes,bbox=props,fontsize=fontsize)
    #plt.xlim(-1e-4)
    plt.tick_params(labelsize=labelsize)
    ax.yaxis.offsetText.set_fontsize(labelsize)
    plt.legend((r"$Q(D_{i})$",r"linear extrapolation",r"$Q(D_{\infty})$",r"$\delta Q$"),loc=0,numpoints=1,fontsize=fontsize)
    plt.grid()
    plt.show()
    """
    """
    quantity = 'E0'
    mean = []; std = []
    for i in xrange(1,len(Ns)+2):
        path = '/home/davidtan/Desktop/MPS_config/XXZ/N_extrap' 
        mean_value = read_N_extrap_table(path+'/N_extrap-g_{}-ma_{}.csv'.format(gs[g_idx],ma),quantity+'_mean')[-i]
        std_value = read_N_extrap_table(path+'/N_extrap-g_{}-ma_{}.csv'.format(gs[g_idx],ma),quantity+'_std')[-i]
        mean.append(mean_value); std.append(std_value)
    
    inv_Ns = 1./np.array(np.append(Ns,np.inf))[::-1]
    fig = plt.figure(figsize=(12,9))
    fig.patch.set_alpha(0.0)
    plt.errorbar(inv_Ns,mean,yerr=std,fmt='-o',elinewidth=2,capsize=8,capthick=2)
    plt.xlabel('$1/N$',fontsize=fontsize)
    plt.ylabel('$E_{0}/N$',fontsize=fontsize)
    #plt.ylabel(r'$|\sum_{n}(-1)^{n}\langle c_{n}^{\dagger}c_{n}\rangle|/N$',fontsize=fontsize)
    ax=plt.gca()
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.08, 0.88, r'$\Delta(g) = {}, \tilde{{m}}_0 a = {}$'.format(gs[g_idx],ma),transform=ax.transAxes,bbox=props,fontsize=fontsize)
    plt.xlim(-1e-4)
    plt.tick_params(labelsize=labelsize)
    plt.tight_layout()
    plt.grid()
    plt.show()
    """
    """
    quantity = 'mg'
    fig = plt.figure(figsize=(12,9))
    fig.patch.set_alpha(0.0)
    for ma in mas:
        mean = []; std = []   
        for g in gs:
            if (g,ma) not in [(gs[22],0.4),(gs[20],0.3),(gs[22],0.3),(gs[20],0.2),(gs[21],0.2),(gs[21],0.1),(gs[23],0.1)]:
                path = '/home/davidtan/Desktop/MPS_config/XXZ/N_extrap'
                mean_value = read_N_extrap_table(path+'/N_extrap-g_{}-ma_{}.csv'.format(g,ma),quantity+'_mean')[-1]
                std_value = read_N_extrap_table(path+'/N_extrap-g_{}-ma_{}.csv'.format(g,ma),quantity+'_std')[-1]
                mean.append(mean_value); std.append(std_value)
            else:
                mean.append(None); std.append(None)
        
        meanp = np.array(mean).astype(np.double)
        mask = np.isfinite(meanp)
        stdp = np.array(std).astype(np.double)
        where_are_NaNs = np.isnan(stdp)
        stdp[where_are_NaNs] = 0.0
        plt.errorbar(gs[mask],Zs[mask]*meanp[mask],yerr=Zs[mask]*stdp[mask],fmt='-o',elinewidth=2,capsize=8,capthick=2)    
    plt.xlabel('$\Delta(g)$',fontsize=fontsize)
    #plt.ylabel(r'$\frac{2\gamma}{\pi\sin\gamma}\times E_{0}/N$',fontsize=fontsize)
    plt.ylabel(r'$E_1-E_0$',fontsize=fontsize)
    #plt.ylabel(r'$|\sum_{n}(-1)^{n}\langle c_{n}^{\dagger}c_{n}\rangle|/N$',fontsize=fontsize)
    #plt.ylabel(r'$\chi$',fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    plt.legend((r"$\tilde{{m}}_{0} a = 0.0$",r"$\tilde{{m}}_{0} a = 0.1$",r"$\tilde{{m}}_{0} a = 0.2$",r"$\tilde{{m}}_{0} a = 0.3$",r"$\tilde{{m}}_{0} a = 0.4$"),loc=0,numpoints=1,fontsize=fontsize)
    plt.grid()
    plt.show()
    """
    """
    quantity = 'mg'
    fig = plt.figure(figsize=(12,9))
    fig.patch.set_alpha(0.0)
    for ma in [0.0,0.1,0.2,0.3,0.4]:
        mean = []; std = []      
        for g in gs:
            path = '/home/davidtan/Desktop/MPS_config/XXZ/N_extrap'
            mean_value = read_N_extrap_table(path+'/N_extrap-g_{}-ma_{}.csv'.format(g,ma),quantity+'_mean')[-1]
            std_value = read_N_extrap_table(path+'/N_extrap-g_{}-ma_{}.csv'.format(g,ma),quantity+'_std')[-1]
            mean.append(mean_value); std.append(std_value)

        plt.errorbar(gs,mean*Zs,yerr=std*Zs,fmt=marker.next(),elinewidth=2,capsize=8,capthick=2,markersize=10)    

    plt.xlabel('$\Delta(g)$',fontsize=fontsize)
    #plt.ylabel(r'$E_{0}/N$',fontsize=fontsize)
    plt.ylabel(r'$E_1-E_0$',fontsize=fontsize)
    #plt.ylabel(r'$|\sum_{n}(-1)^{n}\langle c_{n}^{\dagger}c_{n}\rangle|/N$',fontsize=fontsize)
    #plt.ylabel(r'$\chi$',fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    plt.legend((r"$\tilde{{m}}_{0} a = 0.0$",r"$\tilde{{m}}_{0} a = 0.1$",r"$\tilde{{m}}_{0} a = 0.2$",r"$\tilde{{m}}_{0} a = 0.3$",r"$\tilde{{m}}_{0} a = 0.4$"),loc=0,numpoints=1,fontsize=fontsize,markerscale=1.5)
    plt.grid()
    plt.show()
    """    
