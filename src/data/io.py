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
    found=False
    for line_idx,line in enumerate(open(filename)):
        if pattern in line:   
            found=True; break
    if not found:
        print "pattern '{}' does not be found in file {}".format(pattern,filename)
    
    f=open(filename)
    titles=re.split('\s+',f.readlines()[line_idx])[1:-2]
    f=open(filename)
    datas=map(float,re.split('\s+',f.readlines()[line_idx+1])[1:-2])
    return titles,datas
