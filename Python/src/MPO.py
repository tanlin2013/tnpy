import numpy as np

class MPO:
    def __init__(self,whichMPS,N,D,elem):
        self.whichMPS=whichMPS
        self.N=N
        self.D=D
        self.elem=elem
        self.Sp=np.array([[0,1],[0,0]],dtype=float)
        self.Sm=np.array([[0,0],[1,0]],dtype=float)
        self.Sz=0.5*np.array([[1,0],[0,-1]],dtype=float)
        self.I2=np.identity(2,dtype=float)
        self.O2=np.zeros((2,2),dtype=float)

    def MPO(self):
        MPOlist=[]
        for site in xrange(self.N):
            M=np.array(self.elem(site))
            if whichMPS=='f':
                if site==0:
                    M=M[0,:,:,:]                      
                    M=np.swapaxes(M,0,1)
                elif site==self.N-1:
                    M=M[:,self.D-1,:,:]
                    M=np.swapaxes(M,0,1)        
                else:               
                    M=np.swapaxes(M,0,3)
                    M=np.swapaxes(M,0,2)
                    M=np.swapaxes(M,1,3)
            elif whichMPS=='i':            
                M=np.swapaxes(M,0,3)
                M=np.swapaxes(M,0,2)
                M=np.swapaxes(M,1,3)
            MPOlist.append(M)
        return MPOlist
