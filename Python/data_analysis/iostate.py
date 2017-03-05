import re
import numpy as np

def save(path,state):
    np.savez(path,*state)
    return None
    
def read(path):
    npzfile=np.load(path)
    state=[npzfile['arr_%i'%i] for i in xrange(len(npzfile.files))]
    npzfile.close()
    return state    

def grep(filename,pattern):
    found=False ; line_idx=[]
    for idx,line in enumerate(open(filename)):
        if pattern in line:   
            found=True ; line_idx.append(idx)
    if not found:
        print "pattern '{}' does not be found in file {}".format(pattern,filename)
        return None
    else:            
        return line_idx
