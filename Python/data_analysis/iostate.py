import numpy as np

def save(path,state):
    np.savez(path,*state)
    return None
    
def read(path):
    npzfile=np.load(path)
    state=[npzfile['arr_%i'%i] for i in xrange(len(npzfile.files))]
    npzfile.close()
    return state    
