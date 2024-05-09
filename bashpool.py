import os
import subprocess

def testfun(x,y):
    z=y
    return z

def bashfun(run,tuple):
        inclinationit=tuple[0]
        R0it=tuple[1]
        bashcommand="./ssdisk"+" -run "+str(run)+" -i "+str(inclinationit)+" -R0 "+str(R0it)+" -Rmax "+str(150)
   #     print(bashcommand)
        os.system(bashcommand)
        return run
