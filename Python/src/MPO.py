import numpy as np

class MPO:
    def __init__(self,whichMPS,N,D):
        self.whichMPS=whichMPS
        self.N=N
        self.D=D
        self.elem=None
        
    def MPO(self):
        MPOlist=[]
        for site in xrange(self.N):
            M=np.array(self.elem)
            if self.whichMPS=='f':
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
            elif self.whichMPS=='i':            
                M=np.swapaxes(M,0,3)
                M=np.swapaxes(M,0,2)
                M=np.swapaxes(M,1,3)
            MPOlist.append(M)
        return MPOlist
