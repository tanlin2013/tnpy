"""
This file contains the physical quantities to be measured.
"""

import copy,warnings
import numpy as np
from scipy.linalg import expm
import operators,linalg
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

def bipartite_entanglement_entropy(Gs):
    N = len(Gs); order = tn.get_fmps_order(Gs)
    
    gs = np.copy(Gs); entrolist=[None]*(N-1)
    if order == 'R':
        for site in xrange(N-1):
            gs, S = tn._normalize_fmps(gs,'L',site,return_S=True)
            entrolist[site] = von_Neumann_entropy(S)
    elif order == 'L':
        for site in xrange(N-1,0,-1):
            gs, S = tn._normalize_fmps(gs,'R',site,return_S=True)
            entrolist[N-1-site] = von_Neumann_entropy(S)
    del gs
    return np.array(entrolist)

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

class Sz_corr:
    def __init__(self, Gs, discard_site=1, staggering=False): 
        self.Gs = Gs
        self.N = len(Gs)
        self.order = tn.get_fmps_order(Gs)
        self.discard_site = discard_site
        if self.discard_site < 1: raise ValueError('Must discard at least one site at each boundary.')
        if staggering: self.stag = -1
        else: self.stag = 1
    
    def _update_IL(self, IL, site):
        IL = np.tensordot(np.tensordot(IL,self.Gs[site],axes=(0,0)),
                          np.conjugate(self.Gs[site]),axes=([0,1],[0,1]))
        return IL
        
    def _update_IR(self, IR, site):
        IR = np.tensordot(np.tensordot(IR,self.Gs[site],axes=(0,2)),
                          np.conjugate(self.Gs[site]),axes=([0,2],[2,1]))
        return IR
    
    def _connected_part(self, m, n):
        Sp, Sm, Sz, I2, O2 = operators.spin()       
        if self.order == 'R':
            IL = np.identity(self.Gs[self.discard_site].shape[0],dtype=float)
            IR = np.identity(self.Gs[n].shape[2],dtype=float)
            for site in xrange(self.discard_site,m):
                IL = self._update_IL(IL,site)
        elif self.order == 'L':
            IL = np.identity(self.Gs[m].shape[0],dtype=float)
            IR = np.identity(self.Gs[self.N-1-self.discard_site].shape[2],dtype=float)
            for site in xrange(self.N-1-self.discard_site,n,-1):
                IR = self._update_IR(IR,site)
                
        for site in xrange(m,n+1):
            if site == m:
                corr = np.tensordot(np.tensordot(np.tensordot(IL,
                       self.Gs[site],axes=(0,0)),self.stag**site*Sz,axes=(1,0)),
                       np.conjugate(self.Gs[site]),axes=([0,2],[0,1]))
            elif site == n:
                corr = np.tensordot(np.tensordot(np.tensordot(np.tensordot(corr,
                       self.Gs[site],axes=(0,0)),self.stag**site*Sz,axes=(1,0)),
                       np.conjugate(self.Gs[site]),axes=([0,2],[0,1])),IR,axes=([0,1],[0,1]))
            else:
                corr = np.tensordot(np.tensordot(corr,
                       self.Gs[site],axes=(0,0)),np.conjugate(self.Gs[site]),axes=([0,1],[0,1]))                  
        return corr.item()
    
    def avg_corr(self):
        ls = np.arange(1,self.N-2*self.discard_site); corrs = []
        for l in ls:
            corr = 0.0; Nconf = 0.0
            for m in xrange(self.discard_site,self.N-self.discard_site-l):
                tmp = self._connected_part(m,m+l)
                corr += tmp
                Nconf += 1
                print "For length {}, passing site {}, corr = {}".format(l,m,tmp)
            corr *= 1./Nconf
            corrs.append(np.real_if_close(corr))
        return ls, np.array(corrs)
    
class Spm_corr:
    def __init__(self, Gs, discard_site=1): 
        self.Gs = Gs
        self.N = len(Gs)
        self.order = tn.get_fmps_order(Gs)
        self.discard_site = discard_site
        if self.discard_site < 1: raise ValueError('Must discard at least one site at each boundary.')
    
    def _update_IL(self, IL, site):
        IL = np.tensordot(np.tensordot(IL,self.Gs[site],axes=(0,0)),
                          np.conjugate(self.Gs[site]),axes=([0,1],[0,1]))
        return IL
        
    def _update_IR(self, IR, site):
        IR = np.tensordot(np.tensordot(IR,self.Gs[site],axes=(0,2)),
                          np.conjugate(self.Gs[site]),axes=([0,2],[2,1]))
        return IR
    
    def _connected_part(self, m, n):
        Sp, Sm, Sz, I2, O2 = operators.spin()       
        if self.order == 'R':
            IL = np.identity(self.Gs[self.discard_site].shape[0],dtype=float)
            IR = np.identity(self.Gs[n].shape[2],dtype=float)
            for site in xrange(self.discard_site,m):
                IL = self._update_IL(IL,site)
        elif self.order == 'L':
            IL = np.identity(self.Gs[m].shape[0],dtype=float)
            IR = np.identity(self.Gs[self.N-1-self.discard_site].shape[2],dtype=float)
            for site in xrange(self.N-1-self.discard_site,n,-1):
                IR = self._update_IR(IR,site)
                
        for site in xrange(m,n+1):
            if site == m:
                corr = np.tensordot(np.tensordot(np.tensordot(IL,
                       self.Gs[site],axes=(0,0)),Sp,axes=(1,0)),
                       np.conjugate(self.Gs[site]),axes=([0,2],[0,1]))
            elif site == n:
                corr = np.tensordot(np.tensordot(np.tensordot(np.tensordot(corr,
                       self.Gs[site],axes=(0,0)),Sm,axes=(1,0)),
                       np.conjugate(self.Gs[site]),axes=([0,2],[0,1])),IR,axes=([0,1],[0,1]))
            else:
                corr = np.tensordot(np.tensordot(corr,
                       self.Gs[site],axes=(0,0)),np.conjugate(self.Gs[site]),axes=([0,1],[0,1]))                  
        return corr.item()
    
    def avg_corr(self):
        ls = np.arange(1,self.N-2*self.discard_site); corrs = []
        for l in ls:
            corr = 0.0; Nconf = 0.0
            for m in xrange(self.discard_site,self.N-self.discard_site-l):
                tmp = self._connected_part(m,m+l)
                corr += tmp
                Nconf += 1
                print "For length {}, passing site {}, corr = {}".format(l,m,tmp)
            corr *= 1./Nconf
            corrs.append(np.real_if_close(corr))
        return ls, np.array(corrs)

"""
def bipartite_spin_fluctuations():
    return
"""    

class string_corr:
    def __init__(self, Gs, discard_site=1): 
        self.Gs = Gs
        self.N = len(Gs)
        self.order = tn.get_fmps_order(Gs)
        self.discard_site = discard_site
        if self.discard_site < 1: raise ValueError('Must discard at least one site at each boundary.')
    
    def _update_IL(self, IL, site):
        IL = np.tensordot(np.tensordot(IL,self.Gs[site],axes=(0,0)),
                          np.conjugate(self.Gs[site]),axes=([0,1],[0,1]))
        return IL
        
    def _update_IR(self, IR, site):
        IR = np.tensordot(np.tensordot(IR,self.Gs[site],axes=(0,2)),
                          np.conjugate(self.Gs[site]),axes=([0,2],[2,1]))
        return IR
    
    def _connected_part(self, m, n):
        Sp, Sm, Sz, I2, O2 = operators.spin()
        U = expm(1j*np.pi*Sz)
        if self.order == 'R':
            IL = np.identity(self.Gs[self.discard_site].shape[0],dtype=float)
            IR = np.identity(self.Gs[n].shape[2],dtype=float)
            for site in xrange(self.discard_site,m):
                IL = self._update_IL(IL,site)
        elif self.order == 'L':
            IL = np.identity(self.Gs[m].shape[0],dtype=float)
            IR = np.identity(self.Gs[self.N-1-self.discard_site].shape[2],dtype=float)
            for site in xrange(self.N-1-self.discard_site,n,-1):
                IR = self._update_IR(IR,site)
                
        for site in xrange(m,n+1):
            if site == m:
                corr = np.tensordot(np.tensordot(np.tensordot(IL,
                       self.Gs[site],axes=(0,0)),Sp,axes=(1,0)),
                       np.conjugate(self.Gs[site]),axes=([0,2],[0,1]))
            elif site == n:
                corr = np.tensordot(np.tensordot(np.tensordot(np.tensordot(corr,
                       self.Gs[site],axes=(0,0)),Sm,axes=(1,0)),
                       np.conjugate(self.Gs[site]),axes=([0,2],[0,1])),IR,axes=([0,1],[0,1]))
            else:
                corr = np.tensordot(np.tensordot(np.tensordot(corr,
                       self.Gs[site],axes=(0,0)),U,axes=(1,0)),np.conjugate(self.Gs[site]),axes=([0,2],[0,1]))                  
        return corr.item()
    
    def avg_corr(self):
        ls = np.arange(1,self.N-2*self.discard_site,2); corrs = []
        for l in ls:
            corr = 0.0; Nconf = 0.0
            for m in xrange(self.discard_site,self.N-self.discard_site-l):
                tmp = self._connected_part(m,m+l)
                corr += tmp
                Nconf += 1
                print "For length {}, passing site {}, corr = {}".format(l,m,tmp)
            corr *= 1./Nconf
            corrs.append(np.real_if_close(corr))
        return ls, np.array(corrs)

class vertex_corr:
    def __init__(self, Gs, discard_site=2):
        self.Gs = Gs
        self.N = len(Gs)
        self.order = tn.get_fmps_order(Gs)
        self.discard_site = discard_site
        if self.discard_site < 2: raise ValueError('Must discard at least two site at each boundary.')
    
    def _update_IL(self, IL, site):
        IL = np.tensordot(np.tensordot(IL,self.Gs[site],axes=(0,0)),
                          np.conjugate(self.Gs[site]),axes=([0,1],[0,1]))
        return IL
        
    def _update_IR(self, IR, site):
        IR = np.tensordot(np.tensordot(IR,self.Gs[site],axes=(0,2)),
                          np.conjugate(self.Gs[site]),axes=([0,2],[2,1]))
        return IR
    
    def _vertex_op(self, sign):
        Sp, Sm, Sz, I2, O2 = operators.spin()
        Op = np.kron(Sz,I2)-np.kron(I2,Sz)-1j*sign*(np.kron(Sp,Sm)+np.kron(Sm,Sp))
        return Op
    
    def _connected_part(self, m, n):
        if self.order == 'R':
            IL = np.identity(self.Gs[self.discard_site].shape[0],dtype=float)
            IR = np.identity(self.Gs[n].shape[2],dtype=float)
            for site in xrange(self.discard_site,m):
                IL = self._update_IL(IL,site)
        elif self.order == 'L':
            IL = np.identity(self.Gs[m].shape[0],dtype=float)
            IR = np.identity(self.Gs[self.N-1-self.discard_site].shape[2],dtype=float)
            for site in xrange(self.N-1-self.discard_site,n+1,-1):
                IR = self._update_IR(IR,site)
        
        for site in xrange(m,n+1,2):
            if site == m:
                corr = np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(
                        IL,self.Gs[m],axes=(0,0)),np.conjugate(self.Gs[m]),axes=(0,0)),
                        self._vertex_op(1.0),axes=([0,2],[0,1])),
                        self.Gs[m+1],axes=([0,2],[0,1])),np.conjugate(self.Gs[m+1]),axes=([0,1],[0,1]))
            elif site == n:
                corr = np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(
                        corr,self.Gs[n],axes=(0,0)),np.conjugate(self.Gs[n]),axes=(0,0)),
                        self._vertex_op(-1.0),axes=([0,2],[0,1])),
                        self.Gs[n+1],axes=([0,2],[0,1])),np.conjugate(self.Gs[n+1]),axes=([0,1],[0,1])),
                        IR,axes=([0,1],[0,1])) 
            else:
                corr = np.tensordot(np.tensordot(corr,
                       self.Gs[site],axes=(0,0)),np.conjugate(self.Gs[site]),axes=([0,1],[0,1]))      
        return np.real_if_close(corr).item()
    
    def avg_corr(self):
        ls = np.arange(2,self.N-2*self.discard_site,2); corrs = []
        for l in ls:
            corr = 0.0; Nconf = 0.0
            for m in xrange(self.discard_site,self.N-self.discard_site-l):
                tmp = self._connected_part(m,m+l)
                corr += tmp
                Nconf += 1
                print "For length {}, passing site {}, corr = {}".format(l,m,tmp)
            corr *= 1./Nconf
            corrs.append(np.real_if_close(corr))
        return ls, np.array(corrs)        
    
"""
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
"""

def fermion_momentum(Gs):
    Sp, Sm, Sz, I2, O2 = operators.spin()
    N = len(Gs); order = tn.get_fmps_order(Gs)
    
    op = -1j*(np.kron(np.kron(Sm,Sz),Sp)-np.kron(np.kron(Sp,Sz),Sm))
    op = np.ndarray.reshape(op,(2,2,2,2,2,2))
    op2 = np.tensordot(op,op,axes=([1,3,5],[0,2,4]))
    op2 = np.swapaxes(op2,1,3)
    op2 = np.swapaxes(op2,2,4)
    op2 = np.swapaxes(op2,2,3)
    
    def update_p(site):      
        if site == 0:
            IR = np.identity(Gs[site+2].shape[2],dtype=float)
            update_p = np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(
                    Gs[site],op2,axes=(0,0)),np.conjugate(Gs[site]),axes=(1,0)),
                    Gs[site+1],axes=([0,1],[0,1])),np.conjugate(Gs[site+1]),axes=([0,1],[1,0])),
                    Gs[site+2],axes=([0,2],[1,0])),np.conjugate(Gs[site+2]),axes=([0,1],[1,0])),
                    IR,axes=([0,1],[0,1]))
        elif site == N-3:
            IL = np.identity(Gs[site].shape[0],dtype=float)
            update_p = np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(
                    IL,Gs[site],axes=(0,0)),
                    op2,axes=(1,0)),np.conjugate(Gs[site]),axes=([0,2],[0,1])),
                    Gs[site+1],axes=([0,1],[0,1])),np.conjugate(Gs[site+1]),axes=([0,3],[1,0])),
                    Gs[site+2],axes=([0,2],[0,1])),np.conjugate(Gs[site+2]),axes=([0,1],[0,1]))
        else:
            IL = np.identity(Gs[site].shape[0],dtype=float)
            IR = np.identity(Gs[site+2].shape[2],dtype=float)
            update_p = np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(
                    IL,Gs[site],axes=(0,0)),
                    op2,axes=(1,0)),np.conjugate(Gs[site]),axes=([0,2],[0,1])),
                    Gs[site+1],axes=([0,1],[0,1])),np.conjugate(Gs[site+1]),axes=([0,3],[1,0])),
                    Gs[site+2],axes=([0,2],[1,0])),np.conjugate(Gs[site+2]),axes=([0,1],[1,0])),
                    IR,axes=([0,1],[0,1]))
        return np.real_if_close(update_p).item()
    
    p = 0.0
    if order == 'R':
        for site in xrange(N-2):
            p += update_p(site)
            Gs = tn._normalize_fmps(Gs,'L',site)          
    elif order == 'L':
        for site in xrange(N-1,1,-1):
            p += update_p(site-2)
            Gs = tn._normalize_fmps(Gs,'R',site)
    return p
      
class TEBD_corr:
    def __init__(self, Gs, d, chi, dt, maxstep, discard_site=2):
        self.Gs = Gs
        self.N = len(Gs)
        self.d = d
        self.chi = chi
        self.dt = dt
        self.maxstep = maxstep
        order = tn.get_fmps_order(self.Gs)
        if order == 'R': self.Gs = tn.normalize_fmps(self.Gs, 'L')
        self.SVMs = [None]*(self.N-1)
        for site in xrange(self.N-1,0,-1):
            self.Gs, S = tn._normalize_fmps(self.Gs,'R',site,return_S=True)
            self.SVMs[site-1] = S
        self.Gs0 = copy.copy(Gs)
        self.discard_site = discard_site
        if self.discard_site < 1: raise ValueError('Must discard at least one site at each boundary.')    
        self.config_ID = []
        self.Gs_config_dict = []
        self.SVMs_config_dict = []
        
    def _gate(self, site):
        Sp, Sm, Sz, I2, O2 = operators.spin()
        Op = np.kron(Sp,Sm)-np.kron(Sm,Sp)
        Op = np.ndarray.reshape(Op, (self.d, self.d, self.d, self.d))
        Op = np.swapaxes(Op,1,2)
        Op = np.ndarray.reshape(Op, (self.d**2, self.d**2))
        U = expm(Op*self.dt)
        U = np.ndarray.reshape(U, (self.d, self.d, self.d, self.d))
        U = np.swapaxes(U,1,2)
        return U
    
    def time_evolution(self, m, n, use_config=True, svd_method='numpy'):
        self.Gs = copy.copy(self.Gs0)
        for step in xrange(self.maxstep):
            k = n-2*step
            if use_config and n-m > 2 and k-m > 2:
                key = self.config_ID.index('point_{}_{}-layer_{}'.format(m,n-2,2*step+1))       
                self.Gs = self.Gs[:m] + self.Gs_config_dict[key] + self.Gs[k:]
                self.SVMs[k] = self.SVMs_config_dict[key]
            else:
                k = m
            for site in xrange(k,n+1,2):
                # contract MPS and U into theta
                theta_p = np.tensordot(np.tensordot(self.Gs[site],self._gate(site),axes=(1,0)),self.Gs[site+1],axes=([1,3],[0,1]))
                theta = np.tensordot(np.diagflat(self.SVMs[site-1]),theta_p,axes=(1,0))
                theta = np.ndarray.reshape(theta,(self.d*self.Gs[site].shape[0],self.d*self.Gs[site+1].shape[2]))
                X, S, Y = linalg.svd(theta, self.chi, method=svd_method)
                # form the new configurations
                self.SVMs[site] = S
                self.Gs[site+1] = np.ndarray.reshape(Y,self.Gs[site+1].shape)                  
                self.Gs[site] = np.tensordot(theta_p,self.Gs[site+1],axes=([2,3],[1,2]))                                   
            if use_config and n < self.N-2-self.discard_site:
                self.config_ID.append('point_{}_{}-layer_{}'.format(m,n,2*step+1))
                self.SVMs_config_dict.append(copy.copy(self.SVMs[n+1-2*step]))
                self.Gs_config_dict.append(copy.copy(self.Gs[m:n+2-2*step]))           
            
            k = n-2*step-1
            if use_config and n-m > 2 and k-m+1 > 2:  
                key = self.config_ID.index('point_{}_{}-layer_{}'.format(m,n-2,2*step+2))         
                self.Gs = self.Gs[:m] + self.Gs_config_dict[key] + self.Gs[k:]
                self.SVMs[k] = self.SVMs_config_dict[key]
            else:
                k = m+1
            for site in xrange(k,n,2):
                # contract MPS and U into theta
                theta_p = np.tensordot(np.tensordot(self.Gs[site],self._gate(site),axes=(1,0)),self.Gs[site+1],axes=([1,3],[0,1]))
                theta = np.tensordot(np.diagflat(self.SVMs[site-1]),theta_p,axes=(1,0))
                theta = np.ndarray.reshape(theta,(self.d*self.Gs[site].shape[0],self.d*self.Gs[site+1].shape[2]))
                X, S, Y = linalg.svd(theta, self.chi, method=svd_method)
                # form the new configurations
                self.SVMs[site] = S
                self.Gs[site+1] = np.ndarray.reshape(Y,self.Gs[site+1].shape)                  
                self.Gs[site] = np.tensordot(theta_p,self.Gs[site+1],axes=([2,3],[1,2]))                  
            if use_config and n < self.N-2-self.discard_site:
                self.config_ID.append('point_{}_{}-layer_{}'.format(m,n,2*step+2))
                self.SVMs_config_dict.append(copy.copy(self.SVMs[n-2*step]))
                self.Gs_config_dict.append(copy.copy(self.Gs[m:n+1-2*step]))          
            if use_config and n-m > 2:
                self._dict_cleaner(m,n,2*step+2)     
        return
    
    def _dict_cleaner(self, m, n, layer):
        key = self.config_ID.index('point_{}_{}-layer_{}'.format(m,n-2,layer-1))
        del self.SVMs_config_dict[key]
        del self.Gs_config_dict[key]
        del self.config_ID[key]
        key = self.config_ID.index('point_{}_{}-layer_{}'.format(m,n-2,layer))
        del self.SVMs_config_dict[key]       
        del self.Gs_config_dict[key]
        del self.config_ID[key]
        return
        
    def exp_value(self):
        for site in xrange(self.N):
            if site == 0:
                corr = np.tensordot(self.Gs0[site],self.Gs[site],axes=(0,0))
            elif site == self.N-1:
                corr = np.tensordot(np.tensordot(corr,self.Gs0[site],axes=(0,1)),self.Gs[site],axes=([0,1],[1,0]))
            else:
                corr = np.tensordot(np.tensordot(corr,self.Gs0[site],axes=(0,0)),self.Gs[site],axes=([0,1],[0,1]))
        corr = np.real_if_close(corr).item()
        return corr
    
    def no_avg_corr(self, svd_method='numpy'):
        ls = np.arange(2,self.N-2*self.discard_site,2); corrs = []
        for l in ls:
            m = (self.N-l-1)/2
            try:
                self.time_evolution(m,m+l,use_config=False,svd_method=svd_method)
                corr = self.exp_value()
            except:
                corr = np.nan
                warnings.simplefilter("always")        
                warnings.warn("ValueWarning: Encounter NaN in SVD, skip.")
            print "For length {}, passing site {}, corr = {}".format(l,m,corr)
            corrs.append(np.real_if_close(corr))
        return ls, np.array(corrs)
    
    def avg_corr(self, N_conf=None, use_config=True, svd_method='numpy'):
        ls = np.arange(2,self.N-2*self.discard_site,2); corrs = []
        for l in ls:
            corr = 0.0; Nconf = 0.0
            for m in range(self.discard_site,self.N-self.discard_site-l,2)[:N_conf]:
                try:    
                    self.time_evolution(m,m+l,use_config,svd_method)
                    tmp = self.exp_value()
                    Nconf += 1
                except:
                    tmp = 0.0
                    warnings.simplefilter("always")        
                    warnings.warn("ValueWarning: Encounter NaN in SVD, skip.")
                corr += tmp
                print "For length {}, passing site {}, corr = {}".format(l,m,tmp)
            if Nconf == 0:
                corrs.append(np.nan)
            else:
                corr *= 1./Nconf
                corrs.append(np.real_if_close(corr))
        return ls, np.array(corrs)
