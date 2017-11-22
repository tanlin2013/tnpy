"""
This file contains the physical quantities to be measured.
"""

import warnings
import numpy as np
from scipy.linalg import expm
import operators
import tnstate as tn 

class _update_Env:
    def __init__(self, MPO, Gs, N):
        self.MPO = MPO 
        self.Gs = Gs
        self.N = N      
    
    def _update_EnvL(self, EnvL, site):
        M = self.MPO(site)
        if site == self.N-1:
            EnvL = np.tensordot(np.tensordot(np.tensordot(EnvL,self.Gs[site],axes=(0,1)),
                 M,axes=([0,2],[1,0])),np.conjugate(self.Gs[site]),axes=([0,1],[1,0]))
        else:
            EnvL = np.tensordot(np.tensordot(np.tensordot(EnvL,self.Gs[site],axes=(0,0)),
                 M,axes=([0,2],[1,0])),np.conjugate(self.Gs[site]),axes=([0,2],[0,1]))
        return EnvL

    def _update_EnvR(self, EnvR, site):
        M = self.MPO(site)
        if site == 0: 
            EnvR = np.tensordot(np.tensordot(np.tensordot(EnvR,self.Gs[site],axes=(0,1)),
                 M,axes=([0,2],[1,0])),np.conjugate(self.Gs[site]),axes=([0,1],[1,0]))   
        else:
            EnvR = np.tensordot(np.tensordot(np.tensordot(EnvR,self.Gs[site],axes=(0,2)),
                 M,axes=([0,3],[3,0])),np.conjugate(self.Gs[site]),axes=([0,3],[2,1]))
        return EnvR 
    
    def _update_EnvL2(self, EnvL2, site):
        M = self.MPO(site)
        if site == self.N-1:
            EnvL2 = np.tensordot(np.tensordot(np.tensordot(np.tensordot(EnvL2,
                  self.Gs[site],axes=(0,1)),M,axes=([0,3],[1,0])),
                  M,axes=([0,2],[1,2])),np.conjugate(self.Gs[site]),axes=([0,1],[1,0]))
        else:
            EnvL2 = np.tensordot(np.tensordot(np.tensordot(np.tensordot(EnvL2,
                  self.Gs[site],axes=(0,0)),M,axes=([0,3],[1,0])),
                  M,axes=([0,3],[1,2])),np.conjugate(self.Gs[site]),axes=([0,3],[0,1]))
        return EnvL2

    def _update_EnvR2(self, EnvR2, site):
        M = self.MPO(site)
        if site == 0:    
            EnvR2 = np.tensordot(np.tensordot(np.tensordot(np.tensordot(EnvR2,
                  self.Gs[site],axes=(0,1)),M,axes=([0,3],[1,0])),
                  M,axes=([0,2],[1,2])),np.conjugate(self.Gs[site]),axes=([0,1],[1,0]))
        else:
            EnvR2 = np.tensordot(np.tensordot(np.tensordot(np.tensordot(EnvR2,
                  self.Gs[site],axes=(0,2)),M,axes=([0,4],[3,0])),
                  M,axes=([0,4],[3,2])),np.conjugate(self.Gs[site]),axes=([0,3],[2,1]))
        return EnvR2

def expectation_value(MPO, Gs):
    N = len(Gs); h = _update_Env(MPO,Gs,N); order = tn.get_fmps_order(Gs)
    if order == 'L':
        for site in xrange(N):
            if site == 0:
                M = MPO(site)
                EnvL = tn.transfer_operator(Gs[site],M)
            else:
                EnvL = h._update_EnvL(EnvL,site)
        expval = EnvL.item()
    elif order == 'R':
        for site in xrange(N-1,-1,-1):
            if site == N-1:
                M = MPO(site)
                EnvR = tn.transfer_operator(Gs[site],M)
            else:
                EnvR = h._update_EnvR(EnvR,site)
        expval = EnvR.item()
    return expval
    
def variance(MPO, Gs):
    N = len(Gs); h = _update_Env(MPO,Gs,N); order = tn.get_fmps_order(Gs)
    if order == 'L':
        for site in xrange(N):
            if site == 0:
                M = MPO(site)
                EnvL = tn.transfer_operator(Gs[site],M)
                EnvL2 = np.tensordot(np.tensordot(np.tensordot(Gs[site],
                      M,axes=(0,0)),M,axes=(2,2)),np.conjugate(Gs[site]),axes=(2,0))
            else:
                EnvL = h._update_EnvL(EnvL,site)
                EnvL2 = h._update_EnvL2(EnvL2,site)
        var = EnvL2.item()-EnvL.item()**2    
    elif order == 'R':
        for site in xrange(N-1,-1,-1):
            if site == N-1:
                M = MPO(site)
                EnvR = tn.transfer_operator(Gs[site],M)
                EnvR2 = np.tensordot(np.tensordot(np.tensordot(Gs[site],
                      M,axes=(0,0)),M,axes=(2,2)),np.conjugate(Gs[site]),axes=(2,0))
            else:
                EnvR = h._update_EnvR(EnvR,site)
                EnvR2 = h._update_EnvR2(EnvR2,site)
        var = EnvR2.item()-EnvR.item()**2       
    if var < 0.0:
        warnings.simplefilter("always")
        warnings.warn("PrecisionError: encounter negative variance after the subtraction.")
    return var

def von_Neumann_entropy(S):
    ss = np.square(S)        
    entropy = -np.sum(ss*np.log(ss))
    return entropy

def bipartite_entanglement_entropy(Gs, bond):
    N = len(Gs); d = Gs[0].shape[0]; order = tn.get_fmps_order(Gs)
    
    gs = np.copy(Gs)
    if order == 'R':
        for site in xrange(bond+1):
            gs = tn._normalize_fmps(gs,'L',site)
    elif order == 'L':
        for site in xrange(N-1,bond+1,-1):
            gs = tn._normalize_fmps(gs,'R',site)
    theta = np.ndarray.reshape(gs[bond+1],(gs[bond+1].shape[0],d*gs[bond+1].shape[2]))
    X, S, Y = np.linalg.svd(theta,full_matrices=False)
    
    entropy = von_Neumann_entropy(S); del gs
    return entropy

def Sz_site(Gs, staggering=False):
    Sp, Sm, Sz, I2, O2 = operators.spin()
    N = len(Gs); order = tn.get_fmps_order(Gs); state=[None]*N
    if staggering: stag = -1
    else: stag = 1
    
    def update_Sz(site):
        if site == 0 or site == N-1:
            I = np.identity(Gs[site].shape[1],dtype=float)
            Sz_site = np.tensordot(np.tensordot(np.tensordot(I,Gs[site],axes=(0,1)),
                    stag**site*Sz,axes=(1,0)),np.conjugate(Gs[site]),axes=([0,1],[1,0]))
        else:
            IL = np.identity(Gs[site].shape[0],dtype=float)
            IR = np.identity(Gs[site].shape[2],dtype=float)
            Sz_site = np.tensordot(np.tensordot(np.tensordot(np.tensordot(IL,
                    Gs[site],axes=(0,0)),stag**site*Sz,axes=(1,0)),
                    np.conjugate(Gs[site]),axes=([0,2],[0,1])),IR,axes=([0,1],[0,1]))
        return Sz_site.item() 
        
    if order == 'R': # state is right-normalized
        for site in xrange(N):
            state[site] = update_Sz(site)
            if site < N-1:
                Gs = tn._normalize_fmps(Gs,'L',site)
    elif order == 'L': # state is left-normalized
        for site in xrange(N-1,-1,-1):
            state[site] = update_Sz(site)
            if site > 0:
                Gs = tn._normalize_fmps(Gs,'R',site)
    return state 

def Sz_corr(Gs, m, n, staggering=False): 
    """
    0 < m < n < N-1
    """
    Sp, Sm, Sz, I2, O2 = operators.spin()
    N = len(Gs); order = tn.get_fmps_order(Gs)
    if staggering: stag = -1
    else: stag = 1
    
    if m >= n:
        raise ValueError('m must be smaller than n.')
    if m not in xrange(1,N-1) and n-m not in xrange(N-2):
        raise ValueError('|m-n| cannot exceed the system size.')
    
    if order == 'R':
        for site in xrange(m):
            Gs = tn._normalize_fmps(Gs,'L',site)
    elif order == 'L':
        for site in xrange(N-1,n,-1):
            Gs = tn._normalize_fmps(Gs,'R',site)       

    IL = np.identity(Gs[m].shape[0],dtype=float)
    IR = np.identity(Gs[n].shape[2],dtype=float)

    for site in xrange(m,n+1):
        if site == m:
            corr = np.tensordot(np.tensordot(np.tensordot(IL,
                 Gs[site],axes=(0,0)),stag**site*Sz,axes=(1,0)),
                 np.conjugate(Gs[site]),axes=([0,2],[0,1]))
        elif site == n:
            corr = np.tensordot(np.tensordot(np.tensordot(np.tensordot(corr,
                 Gs[site],axes=(0,0)),stag**site*Sz,axes=(1,0)),
                 np.conjugate(Gs[site]),axes=([0,2],[0,1])),IR,axes=([0,1],[0,1]))
        else:
            corr = np.tensordot(np.tensordot(corr,
                 Gs[site],axes=(0,0)),np.conjugate(Gs[site]),axes=([0,1],[0,1]))                  
    return corr.item()

"""
def bipartite_spin_fluctuations():
    return
"""    

def string_order_paras(Gs, m, n, sign=1):
    """
    0 < m < n < N-1
    """
    Sp, Sm, Sz, I2, O2 = operators.spin()
    N = len(Gs); order = tn.get_fmps_order(Gs)
 
    if m >= n:
        raise ValueError('m must be smaller than n.')
    if m not in xrange(1,N-1) and n-m not in xrange(N-2):
        raise ValueError('|m-n| cannot exceed the system size.')
        
    def string_operator():
        U = expm(sign*1j*np.pi*Sz)
        return U
        
    if order == 'R':
        for site in xrange(m):
            Gs = tn._normalize_fmps(Gs,'L',site)
    elif order == 'L':
        for site in xrange(N-1,n,-1):
            Gs = tn._normalize_fmps(Gs,'R',site)       

    IL = np.identity(Gs[m].shape[0],dtype=float)
    IR = np.identity(Gs[n].shape[2],dtype=float)
        
    for site in xrange(m,n+1):
        if site == m:
            SO = np.tensordot(np.tensordot(np.tensordot(IL,
               Gs[site],axes=(0,0)),Sp,axes=(1,0)),
               np.conjugate(Gs[site]),axes=([0,2],[0,1]))
        elif site == n:
            SO = np.tensordot(np.tensordot(np.tensordot(np.tensordot(SO,
               Gs[site],axes=(0,0)),Sm,axes=(1,0)),
               np.conjugate(Gs[site]),axes=([0,2],[0,1])),IR,axes=([0,1],[0,1]))
        else:
            SO = np.tensordot(np.tensordot(np.tensordot(SO,
               Gs[site],axes=(0,0)),string_operator(),axes=(1,0)),
               np.conjugate(Gs[site]),axes=([0,2],[0,1]))
    SO = np.real_if_close(SO).item()
    return SO

class BKT_corr:
    def __init__(self, Gs, g, discard_site): 
        self.Gs, self.SVM = tn.normalize_fmps(Gs,'mix')
        self.g = g
        self.N = len(Gs)
        self.discard_site = discard_site
        if self.discard_site < 2: raise ValueError('Must discard at least two site at each boundary.')    
        if self.discard_site%2 != 0: raise ValueError('Must discard even number of sites')
    
    def _bkt_operator(self):
        Sp, Sm, Sz, I2, O2= operators.spin()
        op = expm((self.g+np.pi)*(np.kron(Sp,Sm)-np.kron(Sm,Sp)))
        op = np.ndarray.reshape(op,(2,2,2,2)) 
        return op
    
    def _update_IL(self, IL, site):
        IL = np.tensordot(np.tensordot(np.tensordot(np.tensordot(
            IL,self.Gs[site],axes=(0,0)),
            np.conjugate(self.Gs[site]),axes=([0,1],[0,1])),
            self.Gs[site+1],axes=(0,0)),
            np.conjugate(self.Gs[site+1]),axes=([0,1],[0,1]))
        if site == self.N/2-2:
            IL = np.tensordot(np.tensordot(IL,self.SVM,axes=(0,0)),np.conjugate(self.SVM),axes=(0,0))
        return IL
    
    def _update_IR(self, IR, site):
        IR = np.tensordot(np.tensordot(np.tensordot(np.tensordot(
            IR,self.Gs[site],axes=(0,2)),
            np.conjugate(self.Gs[site]),axes=([0,2],[2,1])),
            self.Gs[site-1],axes=(0,2)),
            np.conjugate(self.Gs[site-1]),axes=([0,2],[2,1]))
        if site == self.N/2+1:
            IR = np.tensordot(np.tensordot(IR,self.SVM,axes=(0,1)),np.conjugate(self.SVM),axes=(0,1))
        return IR
    
    def _connected_part(self, m, n):        
        if m < self.N/2-2 and n < self.N/2+2:
            IL = np.identity(self.Gs[m].shape[0],dtype=float)
            IR = np.identity(self.Gs[self.N/2+1].shape[2],dtype=float)
            for site in xrange(self.N/2+1,n+1,-2):
                IR = self._update_IR(IR, site)      
        elif m <= self.N/2-2 and n >= self.N/2:
            IL = np.identity(self.Gs[m].shape[0],dtype=float)
            IR = np.identity(self.Gs[n+1].shape[2],dtype=float)
        elif m >= self.N/2-2 and n >= self.N/2+2:
            IL = np.identity(self.Gs[self.N/2-1].shape[2],dtype=float)
            IR = np.identity(self.Gs[n+1].shape[2],dtype=float)
            for site in xrange(self.N/2-2,m,2):
                IL = self._update_IL(IL, site)
        else:
            raise ValueError('Indices m and n out of range.')

        for site in xrange(m,n+1,2):
            if site == m:
                corr = np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(
                        IL,self.Gs[m],axes=(0,0)),np.conjugate(self.Gs[m]),axes=(0,0)),
                        self._bkt_operator(),axes=([0,2],[0,1])),
                        self.Gs[m+1],axes=([0,2],[0,1])),np.conjugate(self.Gs[m+1]),axes=([0,1],[0,1]))
            elif site == n:
                corr = np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(
                        corr,self.Gs[n],axes=(0,0)),np.conjugate(self.Gs[n]),axes=(0,0)),
                        self._bkt_operator(),axes=([0,2],[0,1])),
                        self.Gs[n+1],axes=([0,2],[0,1])),np.conjugate(self.Gs[n+1]),axes=([0,1],[0,1])),
                        IR,axes=([0,1],[0,1]))   
            else:
                corr = np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(
                        corr,self.Gs[site],axes=(0,0)),np.conjugate(self.Gs[site]),axes=(0,0)),
                        self._bkt_operator(),axes=([0,2],[0,1])),
                        self.Gs[site+1],axes=([0,2],[0,1])),np.conjugate(self.Gs[site+1]),axes=([0,1],[0,1]))
            if m <= self.N/2-2 and n >= self.N/2 and site == self.N/2-2:
                corr = np.tensordot(np.tensordot(corr,self.SVM,axes=(0,0)),np.conjugate(self.SVM),axes=(0,0))
        corr = np.real_if_close(corr).item()
        return corr
        
    def avg_corr(self):
        ls = np.arange(2,self.N-2*self.discard_site,2); corrs = []
        for l in ls:
            corr = 0.0; Nconf = 0.0
            for m in xrange(self.discard_site,self.N-self.discard_site-l,2):
                corr += self._connected_part(m,m+l)
                Nconf += 1
                print "For length {}, passing site {}".format(l,m)
            corr *= 1./Nconf
            corrs.append(np.real_if_close(corr))
        return ls, np.array(corrs)
