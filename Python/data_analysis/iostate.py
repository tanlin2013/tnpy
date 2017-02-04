import numpy as np

def save(path,state):
    np.savez(path,*state)
    return None
    
def read(path):
    npzfile=np.load(path)
    state=[npzfile[i] for i in npzfile]
    state.reverse() ; npzfile.close()
    return state    
