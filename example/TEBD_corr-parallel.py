import TNpy
import warnings,time
import numpy as np
from multiprocessing import Pool

def avg_corr(TEBD_corr, m, l, use_config=False, svd_method='numpy'):
    corr = 0.0; Nconf = 0.0
    for m in xrange(m,m+1):
        try:
            TEBD_corr.time_evolution(m,m+l,use_config,svd_method)
            tmp = TEBD_corr.exp_value()
            Nconf += 1
        except:
            tmp = 0.0
            warnings.simplefilter("always")        
            warnings.warn("ValueWarning: Encounter NaN in SVD, skip.")
        corr += tmp            
        print "For length {}, passing site {}, corr = {}".format(l,m,tmp) 
    if Nconf == 0:
        corr = np.nan
    return corr

def wrap_func(args):
    return avg_corr(*args)

if __name__=='__main__':
    
    gs = np.linspace(-1.0,1.0,101)
    #gs = np.array(gs.tolist()[5:20]+gs.tolist()[20::10])
    mas = [0.0, 0.2]
    
    N = 400
    g = gs[10]
    ma = mas[0]
    mu = 0.0
    lamda = 100.0
    S_target = 0.0
    chi = 50
    tolerance = 1e-12

    path = '/home/davidtan/Desktop/MPS_config/XXZ'        
    Gs = TNpy.data.io.read(path+'/N_{}/MPS-N_{}-g_{}-ma_{}-lambda_{}-Starget_{}-chi_{}-tol_{}.npz'.format(N,N,g,ma,lamda,S_target,chi,tolerance))
        
    d = 2
    maxstep = 2
    dt = 1./float(maxstep)
    #dt = (2*np.pi-2*np.arccos(g))/float(maxstep)
    N_conf = 4
    discard_site = N/4
    TEBD_corr = TNpy.measurement.TEBD_corr(Gs, d, chi, dt, maxstep, discard_site=discard_site)
    
    procs = 4   
    pool = Pool(procs)
    
    t0 = time.time()
    ls = np.arange(2,N-2*discard_site,2); corrs = []  
    for l in ls: 
        args = []
        ms = range(discard_site,N-discard_site-l,2)[:N_conf]
        for m in ms:
            args.append((TEBD_corr,m,l))
        
        corr = pool.map(wrap_func, args)
        corrs.append(sum(corr)/len(corr))
    
    corrs = np.array(corrs)
    print "="*30
    print "Execution time = {}".format((time.time()-t0)/60.**2)+" h."
    
