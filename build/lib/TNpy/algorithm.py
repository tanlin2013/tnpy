"""
This file contains several algorithms which are based on the Matrix Product State (MPS) ansatz.

* Infinite Size Density Matrix Renormalization Group (iDMRG)
* Infinte Time Evolution Bond Decimation (iTEBD)
* Finite Size Density Matrix Renormalization Group (fDMRG)
* Finte Time Evolution Bond Decimation (fTEBD)

"""

import time
import warnings
import numpy as np
from scipy.linalg import expm
from . import linalg
from . import tnstate as tn

class iDMRG:
    def __init__(self, MPO, Gs, SVMs, N, d, chi):
        self.MPO = MPO
        self.Gs = Gs
        self.SVMs = SVMs
        self.N = N
        self.d = d
        self.chi = chi
    
    def _initialize_Env(self):
        D = self.MPO(0).shape[1]
        vL = np.zeros(D)
        vL[0] = 1.0
        L = np.kron(vL, np.identity(self.chi, dtype=float))      
        L = np.ndarray.reshape(L, (self.chi, D, self.chi))
        vR = np.zeros(D)
        vR[-1] = 1.0
        R = np.kron(np.identity(self.chi, dtype=float), vR)
        R = np.ndarray.reshape(R, (self.chi, D, self.chi))
        return L, R
    
    def _effH(self, L, R, A, B):        
        H = np.tensordot(np.tensordot(L,self.MPO(A),axes=(1,1)),
                        np.tensordot(self.MPO(B),R,axes=(3,1)),axes=(4,1))                                  
        H = np.swapaxes(H,1,4)
        H = np.swapaxes(H,3,6)
        H = np.swapaxes(H,1,2)
        H = np.swapaxes(H,5,6)
        H = np.ndarray.reshape(H, (self.chi**2*self.d**2, self.chi**2*self.d**2)) 
        return H
    
    def _effHpsi(self, L, R, A, B):
        def H_matvec(X):
            G = np.ndarray.reshape(X, (self.chi, self.d, self.d, self.chi))
            matvec = np.tensordot(np.tensordot(np.tensordot(np.tensordot(L,
                                G,axes=(0,0)),
                                self.MPO(A),axes=([0,2],[1,0])),
                                self.MPO(B),axes=([1,4],[0,1])),
                                R,axes=([1,4],[0,1]))
           
            matvec = np.ndarray.reshape(matvec,X.shape)
            return matvec
        psi = np.tensordot(np.tensordot(np.tensordot(np.tensordot(self.SVMs[B],
                         self.Gs[A],axes=(1,0)),self.SVMs[A],axes=(2,0)),
                         self.Gs[B],axes=(2,0)),self.SVMs[B],axes=(3,0))
        psi = np.ndarray.reshape(psi,(self.d**2*self.chi**2,1))                   
        return H_matvec, psi
    
    def warm_up_optimize(self, svd_method='numpy'):
        L, R = self._initialize_Env()
        for length in range(self.N/2):
            A = length%2
            B = (length+1)%2
            # optimize 2 new-added sites in the center
            H = self._effH(L,R,A,B)    
            evals, evec = np.linalg.eigh(H)
            E = evals[0]; theta = evec[:,0]
            #E,theta=linalg.eigshmv(*self._effHpsi(L,R,A,B))
            E /= 2*(length+1)
            print("length%d," % length,"E/N= %.12f" % E)
            # SVD and truncation
            theta = np.ndarray.reshape(theta, (self.chi*self.d, self.chi*self.d))
            X, S, Y = linalg.svd(theta, self.chi, svd_method)                 
            self.SVMs[A] = np.diagflat(S/np.linalg.norm(S))
            # form the new configuration
            X = np.ndarray.reshape(X, (self.chi, self.d, self.chi))
            Y = np.ndarray.reshape(Y, (self.chi, self.d, self.chi))
            if length == 1:
                self.Gs[A] = X
                self.Gs[B] = Y
            else:
                SVM_inv = linalg.inverse_SVM(self.SVMs[B])
                self.Gs[A] = np.tensordot(SVM_inv, X, axes=(1,0))
                self.Gs[B] = np.tensordot(Y, SVM_inv, axes=(2,0))
            # update the environment
            if length == 1:
                EnvL = np.tensordot(np.tensordot(self.Gs[A],self.MPO(A),axes=(1,0)),np.conjugate(self.Gs[A]),axes=(3,1))
                L = np.tensordot(L, EnvL, axes=([0,1,2], [0,2,4]))
                EnvR = np.tensordot(np.tensordot(self.Gs[B],self.MPO(B),axes=(1,0)),np.conjugate(self.Gs[B]),axes=(3,1))
                R = np.tensordot(EnvR, R, axes=([1,3,5], [0,1,2]))
            else:    
                EnvL = np.tensordot(np.tensordot(np.tensordot(self.SVMs[B],self.Gs[A],axes=(1,0)),self.MPO(A),axes=(1,0)),
                                  np.tensordot(self.SVMs[B],np.conjugate(self.Gs[A]),axes=(1,0)),axes=(3,1))                        
                L = np.tensordot(L, EnvL, axes=([0,1,2], [0,2,4]))
                EnvR = np.tensordot(np.tensordot(np.tensordot(self.Gs[B],self.SVMs[B],axes=(2,0)),self.MPO(B),axes=(1,0)),
                                  np.tensordot(np.conjugate(self.Gs[B]),self.SVMs[B],axes=(2,0)),axes=(3,1))
                R = np.tensordot(EnvR, R, axes=([1,3,5], [0,1,2]))
        return E

class iTEBD:
    def __init__(self, ham, Gs, SVMs, d, chi, dt, maxstep=1e+4):
        self.ham = ham
        self.Gs = Gs
        self.SVMs = SVMs
        self.d = d
        self.chi = chi
        self.dt = dt
        self.maxstep = maxstep
        
    def _gate(self, parity):
        U = expm(-self.ham(parity)*self.dt)
        U = np.ndarray.reshape(U, (self.d, self.d, self.d, self.d))
        return U
    
    def time_evolution(self, svd_method='numpy'):
        for step in range(self.maxstep):
            A = step%2
            B = (step+1)%2
            #contract to theta
            thetaL = np.tensordot(np.tensordot(self.SVMs[B],self.Gs[A],axes=(1,0)),self.SVMs[A],axes=(2,0))
            thetaR = np.tensordot(self.Gs[B], self.SVMs[B], axes=(2,0))
            theta = np.tensordot(thetaL, thetaR, axes=(2,0))
            #contract theta with U + SVD
            theta_t = np.tensordot(theta, self._gate(A), axes=([1,2], [0,1]))
            theta_t = np.swapaxes(theta_t,1,2)
            theta_t = np.swapaxes(theta_t,2,3)
            theta_t2 = np.ndarray.reshape(theta_t, (self.d*self.chi, self.d*self.chi)) 
            # SVD and truncation
            X, S, Y = linalg.svd(theta_t2, self.chi, svd_method)                                    
            self.SVMs[A] = np.diagflat(S/np.linalg.norm(S))
            #form the new configuration
            X = np.ndarray.reshape(X, (self.chi, self.d, self.chi))
            Y = np.ndarray.reshape(Y, (self.chi, self.d, self.chi))
            SVMB_inv = linalg.inverse_SVM(self.SVMs[B])
            self.Gs[A] = np.tensordot(SVMB_inv, X, axes=(1,0))
            self.Gs[B] = np.tensordot(Y, SVMB_inv, axes=(2,0))
            #expectation values
            H = np.ndarray.reshape(self.ham(A), (self.d, self.d, self.d, self.d))
            E_expect = np.tensordot(np.tensordot(theta,H,axes=([1,2],[0,1])),theta,axes=([0,2,3,1],[0,1,2,3]))                   
            norm = np.tensordot(theta, theta, axes=([0,1,2,3], [0,1,2,3]))       
            E = E_expect/norm
        return E

class fDMRG:
    def __init__(self, MPO, Gs, N, d, chi, tolerance=1e-12, maxsweep=200, projE=None, projGs=None):
        self.MPO = MPO
        self.Gs = Gs
        self.N = N
        self.d = d
        self.chi = chi
        self.tolerance = tolerance
        self.maxsweep = maxsweep
        order = tn.get_fmps_order(self.Gs)
        if order == 'L': self.Gs = tn.normalize_fmps(self.Gs, 'R')
        self.projE = projE
        self.projGs = projGs
        
    def _initialize_Env(self):
        L=[None]*(self.N-1); R=[None]*(self.N-1)
        for site in range(self.N-1):
            if site == 0:
                EnvL = tn.transfer_operator(self.Gs[site], self.MPO(site))             
            else:    
                EnvL = self._update_EnvL(EnvL, site)            
            L[site] = EnvL                       
        for site in range(self.N-1,0,-1):
            if site == self.N-1:
                EnvR = tn.transfer_operator(self.Gs[site], self.MPO(site))
            else:
                EnvR = self._update_EnvR(EnvR, site)            
            R[self.N-1-site] = EnvR      
        return L,R
        
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

    def _initialize_projEnv(self):
        projL=[None]*(self.N-1); projR=[None]*(self.N-1)
        for site in range(self.N-1):           
            if site == 0:               
                projEnvL = np.tensordot(self.Gs[site],self.projGs[site],axes=(0,0))          
            else:
                projEnvL = np.tensordot(np.tensordot(projEnvL,self.Gs[site],axes=(0,0)),self.projGs[site],axes=([0,1],[0,1]))            
            projL[site] = projEnvL                       
        for site in range(self.N-1,0,-1):
            if site == self.N-1:
                projEnvR = np.tensordot(self.Gs[site],self.projGs[site],axes=(0,0))   
            else:
                projEnvR = np.tensordot(np.tensordot(projEnvR,self.Gs[site],axes=(0,2)),self.projGs[site],axes=([0,2],[2,1]))           
            projR[self.N-1-site] = projEnvR    
        return projL, projR
    
    def _effH(self, L, R, site):
        M = self.MPO(site)
        if site == 0:
            dimH = self.d*self.Gs[site].shape[1]
            H = np.tensordot(M, R[self.N-2-site], axes=(1,1))
            H = np.swapaxes(H,1,2)
        elif site == self.N-1:
            dimH = self.d*self.Gs[site].shape[1]
            H = np.tensordot(L[site-1], M, axes=(1,1))
            H = np.swapaxes(H,1,2)              
        else:
            dimH = self.d*self.Gs[site].shape[0]*self.Gs[site].shape[2]
            H = np.tensordot(L[site-1],np.tensordot(M,R[self.N-2-site],axes=(3,1)),axes=(1,1))
            H = np.swapaxes(H,1,3)
            H = np.swapaxes(H,2,4)
            H = np.swapaxes(H,1,4)
        H = np.ndarray.reshape(H, (dimH, dimH))
        psi = np.ndarray.reshape(self.Gs[site], (dimH, 1))
        return H, psi
    
    def _effHpsi(self,L ,R, site):
        M = self.MPO(site)     
        def H_matvec(X):
            G = np.ndarray.reshape(X,self.Gs[site].shape)
            if site == 0:
                matvec = np.tensordot(np.tensordot(R[self.N-2-site],G,axes=(0,1)),M,axes=([2,0],[0,1]))
                matvec = np.swapaxes(matvec,0,1)
            elif site == self.N-1:
                matvec = np.tensordot(np.tensordot(L[site-1],G,axes=(0,1)),M,axes=([2,0],[0,1]))
                matvec = np.swapaxes(matvec,0,1)
            else:
                matvec = np.tensordot(np.tensordot(np.tensordot(L[site-1],G,axes=(0,0)),
                                    M,axes=([0,2],[1,0])),R[self.N-2-site],axes=([1,3],[0,1]))
            matvec = np.ndarray.reshape(matvec, X.shape)
            return matvec
        psi = np.ndarray.reshape(self.Gs[site], (self.Gs[site].size,1))
        return H_matvec, psi
    
    def _effprojHpsi(self, L, R, projL, projR, site):
        H_matvec, psi = self._effHpsi(L,R,site)
        def projH_matvec(X):
            G = np.ndarray.reshape(X,self.Gs[site].shape)
            if site == 0:
                proj_matvec = np.tensordot(projR[self.N-2-site],self.projGs[site],axes=(1,1))
                proj_matvec = np.swapaxes(proj_matvec,0,1)
                proj_matvec = proj_matvec * np.tensordot(np.tensordot(
                        projR[self.N-2-site],G,axes=(0,1)),self.projGs[site],axes=([0,1],[1,0]))
            elif site == self.N-1:
                proj_matvec = np.tensordot(projL[site-1],self.projGs[site],axes=(1,1))
                proj_matvec = np.swapaxes(proj_matvec,0,1)
                proj_matvec = proj_matvec * np.tensordot(np.tensordot(
                        projL[site-1],G,axes=(0,1)),self.projGs[site],axes=([0,1],[1,0]))
            else:
                proj_matvec = np.tensordot(np.tensordot(projL[site-1],self.projGs[site],axes=(1,0)),projR[self.N-2-site],axes=(2,1))
                proj_matvec = proj_matvec * np.tensordot(np.tensordot(np.tensordot(
                        projL[site-1],G,axes=(0,0)),self.projGs[site],axes=([0,1],[0,1])),projR[self.N-2-site],axes=([0,1],[0,1]))
            proj_matvec = np.ndarray.reshape(proj_matvec, X.shape)
            return H_matvec(X) - self.projE * proj_matvec
        return projH_matvec, psi
    
    def _modified_density_matrix(self, alpha, L, R, site):
        if site == 0:
            rho = np.tensordot(np.tensordot(R[self.N-2-site],self.Gs[site],axes=(0,1)),self.MPO(site),axes=([2,0],[0,1]))
            rho = np.swapaxes(rho,0,1)
        elif site == self.N-1:
            rho = np.tensordot(np.tensordot(L[site-1],self.Gs[site],axes=(0,1)),self.MPO(site),axes=([2,0],[0,1]))
            rho = np.swapaxes(rho,0,1)
        else:
            rho = np.tensordot(np.tensordot(np.tensordot(L[site-1],self.Gs[site],axes=(0,0)),
                  self.MPO(site),axes=([0,2],[1,0])),R[self.N-2-site],axes=([1,3],[0,1]))
        rho = np.ndarray.reshape(rho, (rho.size,1))
        return alpha * rho
    
    def variational_optimize(self, show_stats=True, return_stats=True, modified_DM=False, primme_method=2, svd_method='numpy'):
        L, R = self._initialize_Env()
        if self.projE is not None:
            projL, projR = self._initialize_projEnv()
        E0 = 0.0; t0 = time.clock(); alpha = self.tolerance
        for sweep in range(1,self.maxsweep):
            #--------------------------------------------------------------------------------------------
            # Right Sweep         
            for site in range(self.N-1):
                # construct effH & diagonalize it; psi is an initial guess of eigenvector    
                if self.projE is not None:
                    E, theta = linalg.eigshmv(*self._effprojHpsi(L,R,projL,projR,site),tol=0.1*self.tolerance,method=primme_method)
                else:
                    E, theta = linalg.eigshmv(*self._effHpsi(L,R,site),tol=0.1*self.tolerance,method=primme_method) 
                E /= self.N
                if show_stats:
                    print("site%d," % site,"E/N= %.12f" % E)
                if modified_DM:
                    theta += self._modified_density_matrix(alpha, L, R ,site)
                # SVD and truncation
                if site == 0:
                    theta = np.ndarray.reshape(theta,(self.d,self.Gs[site].shape[1])) 
                else:    
                    theta = np.ndarray.reshape(theta,(self.d*self.Gs[site].shape[0],self.Gs[site].shape[2]))                       
                X, S, Y = linalg.svd(theta, self.chi, method=svd_method)
                # form the new configuration               
                if site == self.N-2:
                    self.Gs[site+1] = np.tensordot(self.Gs[site+1],np.dot(np.diagflat(S),Y),axes=(1,1))
                else:
                    self.Gs[site+1] = np.tensordot(np.dot(np.diagflat(S),Y),self.Gs[site+1],axes=(1,0))                           
                if site==0:                    
                    self.Gs[site] = X
                    EnvL = tn.transfer_operator(self.Gs[site],self.MPO(site))
                    if self.projE is not None:
                        projEnvL = np.tensordot(self.Gs[site],self.projGs[site],axes=(0,0))
                else:
                    self.Gs[site] = np.ndarray.reshape(X,self.Gs[site].shape)                                       
                    EnvL = self._update_EnvL(EnvL,site)
                    if self.projE is not None:
                        projEnvL = np.tensordot(np.tensordot(projEnvL,self.Gs[site],axes=(0,0)),self.projGs[site],axes=([0,1],[0,1]))
                L[site] = EnvL
                if self.projE is not None: 
                    projL[site] = projEnvL
            # check convergence of right-sweep   
            dE = E0-E; E0 = E
            if modified_DM and sweep == 1: alpha = 0.1*self.tolerance
            if show_stats:
                print("sweep %.1f," % (sweep-0.5),"E/N= %.12f," % E,"dE= %.4e" % dE)                               
            if self._convergence(sweep-0.5,E,dE):
                break                       
            #--------------------------------------------------------------------------------------------               
            # Left Sweep
            for site in range(self.N-1,0,-1):
                # construct H & diagonalize it; psi is an initial guess of eigenvector                  
                if self.projE is not None:
                    E, theta = linalg.eigshmv(*self._effprojHpsi(L,R,projL,projR,site),tol=0.1*self.tolerance,method=primme_method)
                else:
                    E, theta = linalg.eigshmv(*self._effHpsi(L,R,site),tol=0.1*self.tolerance,method=primme_method) 
                E /= self.N
                if show_stats:
                    print("site%d," % site,"E/N= %.12f" % E)
                if modified_DM:
                    theta += self._modified_density_matrix(alpha, L, R ,site)
                # SVD and truncation
                if site == self.N-1:
                    theta = np.ndarray.reshape(theta,(self.Gs[site].shape[1],self.d))    
                else:    
                    theta = np.ndarray.reshape(theta,(self.Gs[site].shape[0],self.d*self.Gs[site].shape[2]))            
                X, S, Y = linalg.svd(theta, self.chi, method=svd_method)
                # form the new configuration              
                if site == 1:
                    self.Gs[site-1] = np.tensordot(self.Gs[site-1],np.dot(X,np.diagflat(S)),axes=(1,0))
                else:
                    self.Gs[site-1] = np.tensordot(self.Gs[site-1],np.dot(X,np.diagflat(S)),axes=(2,0))
                if site == self.N-1:                    
                    self.Gs[site] = np.transpose(Y)
                    EnvR = tn.transfer_operator(self.Gs[site],self.MPO(site))
                    if self.projE is not None:
                        projEnvR = np.tensordot(self.Gs[site],self.projGs[site],axes=(0,0))
                else:
                    self.Gs[site] = np.ndarray.reshape(Y,self.Gs[site].shape)                      
                    EnvR = self._update_EnvR(EnvR,site)
                    if self.projE is not None:
                        projEnvR = np.tensordot(np.tensordot(projEnvR,self.Gs[site],axes=(0,2)),self.projGs[site],axes=([0,2],[2,1]))
                R[self.N-1-site] = EnvR
                if self.projE is not None:
                    projR[self.N-1-site] = projEnvR 
            # check convergence of left-sweep
            dE = E0-E; E0 = E
            if modified_DM and sweep == 1: modified_DM = False
            if show_stats:
                print("sweep %d," % sweep,"E/N= %.12f," % E,"dE= %.4e" % dE)                   
            if self._convergence(sweep,E,dE):
                break
            #--------------------------------------------------------------------------------------------
        if return_stats:
            t = (time.clock()-t0)/(self.maxsweep*60.0)
            stats = dict(dE=dE, sweep=self.maxsweep, AvgProcT=t)
            return E, stats
        else:
            return E
        
    def _convergence(self, sweep, E, dE): # print warning messages and check the convergence of main routine
        warnings.simplefilter("always")        
        if dE < 0.0:
            warnings.warn("ValueWarning: Encounter negative dE=E(sweep-0.5)-E(sweep).")            
        if sweep == self.maxsweep-1 and dE > self.tolerance: 
            warnings.warn("ConvergenceWarning: Convergence is not yet achieved before the maxsweep has reached.")
            return True       
        if 0.0 <= dE < self.tolerance:
            self.maxsweep=sweep
            return True
        else:
            return False
        
"""
class fTEBD:
    def __init__(self, ham, Gs, d, chi, dt, maxstep):
        self.ham = ham
        self.Gs = Gs
        self.N = len(Gs)
        self.d = d
        self.chi = chi
        self.dt = dt
        self.maxstep = maxstep
    
    def _gate(self, site):
        U = expm(-self.ham(site)*self.dt)
        U = np.ndarray.reshape(U, (self.d, self.d, self.d, self.d))
        return U
        
    def time_evolution(self):
        for step in range(self.maxstep):
            for site in range(N/2+1):
                # contract theta and U
                if site == 0:
                    theta = np.tensordot(np.tensordot())
                    theta = np.ndarray.reshape()
                elif site == self.N-2:
                    theta = np.tensordot(np.tensordot())
                    theta = np.ndarray.reshape()
                else:
                    theta = np.tensordot(np.tensordot())
                    theta = np.ndarray.reshape()
                # SVD and truncation
                X, S, Y = linalg.svd(theta,self.chi)
                # form the new configuration
                S = np.square(S)
                S = np.diagflat(S/np.linalg.norm(S))
                X = np.dot(X,S); Y = np.dot(S,Y)
                if site == 0:
                
                elif site == self.N-2:
                
                else:
                
            for site in range(1,N/2+2):
                # contract theta and U
                
            
            if self._convergence():
                break
                
        return

    def _convergence(self):
        
        return

"""        
        
