import numpy as np

class MPO:
    def __init__(self,whichMPS,N,D):
        self.whichMPS=whichMPS
        self.N=N
        self.D=D
        
    def assign_to_MPO(self,elem,site):
        M=np.array(elem)
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
        return M
    
def spin_operators(spin=0.5):
    Sp=np.array([[0,1],[0,0]],dtype=float)
    Sm=np.array([[0,0],[1,0]],dtype=float)
    Sz=spin*np.array([[1,0],[0,-1]],dtype=float)
    I2=np.identity(2,dtype=float)
    O2=np.zeros((2,2),dtype=float)
    return Sp,Sm,Sz,I2,O2
