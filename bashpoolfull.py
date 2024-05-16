import os
import subprocess
import numpy as np

def testfun(x,y):
    z=y
    return z
    
def find_number(arr, target):
    return target in arr

def bashfun(run,tuple,selecttuples,clusterc):
        Rgit=tuple[0]
        Rinit=tuple[1]
        R0it=tuple[2]
        slopeit=tuple[3]
        inclinationit=tuple[4]
        posangleit=tuple[5]
        numpix=int(tuple[6])
        bashcommand="./ssdisk"+" -run "+str(run)+" -Rmax "+str(150)+" -np "+str(numpix)+" -cluster "+str(int(clusterc))
        if find_number(selecttuples, 0):
            bashcommand=bashcommand+" -Rg "+str(Rgit)
        if find_number(selecttuples, 1):
            bashcommand=bashcommand+" -Rin "+str(Rinit)
        if find_number(selecttuples, 2):
            bashcommand=bashcommand+" -R0 "+str(R0it)
        if find_number(selecttuples, 3):
            bashcommand=bashcommand+" -n "+str(slopeit)
        if find_number(selecttuples, 4):
            bashcommand=bashcommand+" -i "+str(np.arccos(inclinationit))
        if find_number(selecttuples, 5):
            bashcommand=bashcommand+" -phi "+str(np.arccos(posangleit))
       # bashcommand="./ssdisk"+" -run "+str(run)+" -Rg "+str(Rgit)+" -Rin "+str(Rinit)+" -R0 "+str(R0it)+" -n "+str(slopeit)+" -i "+str(np.arccos(inclinationit))+" -phi "+str(np.arccos(posangleit))
       # print(bashcommand)
        os.system(bashcommand)
        return run
        
def process_data(data):
    x, y = data
    return torch.squeeze(x.view(x.size(1) * x.size(2), -1), 1), y