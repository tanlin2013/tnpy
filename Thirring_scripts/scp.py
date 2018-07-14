#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 21:57:38 2017

@author: davidtan
"""

import numpy as np
import pexpect,getpass,time

def ssh_cmd(cmd,password,timeout=40):
    for attempt in xrange(3):
        proc = pexpect.spawn(cmd,timeout=timeout)
        err = proc.expect(["password: ","Connection reset by peer"])
        if err==0:
            proc.sendline(password)
        elif err==1:
            print "Connetion reset by peer. Try again."
            proc.close; time.sleep(40); continue
        err = proc.expect([pexpect.EOF,pexpect.TIMEOUT,"No such file or directory","rm: cannot"])
        print "code: ",err
        proc.close
        if err==0:
            pass; break
        elif err==1:
            raise pexpect.TIMEOUT('timeout')
        elif err==2:
            raise IOError('scp: No such file or directory')
        elif err==3:
            raise IOError('rm: cannot remove the file')
    return

if __name__=='__main__':

    N=1000
    gs=np.linspace(-1.0,1.0,101)
    gs = np.array(gs.tolist()[5:20]+gs.tolist()[20::10])
    #gs=[-0.86,-0.78,-0.74,-0.72,-0.7,-0.68,-0.62]
    #gs=[-0.58,-0.56,-0.54,-0.52,-0.5]
    #gs=[-0.6, -0.2, 0.2, 0.8]
    mas=[0.0,0.1,0.2]
    #mas=[0.02,0.04,0.06,0.08,0.13,0.16]
    lamda=100.0
    S_target=0.0
    chi=600
    tolerance=1e-8
    suffix='eps'
    
    password = getpass.getpass(">>> pass: ")
    
    for g in gs:
        for ma in mas:
            err=0
            try:
                #cmd = "scp tanlin@spark.ep.nctu.edu.tw:/data3/tanlin/MPS_config/string_corr/N_{}/chi_{}/avg_10/string_corr-N_{}-g_{}-ma_{}-chi_{}.".format(N,chi,N,g,ma,chi)+suffix+" ."
                #cmd = "scp tanlin@spark.ep.nctu.edu.tw:data/XXZ/measurements/N_{}/chi_{}/E-N_{}-ma_{}-chi_{}.".format(N,chi,N,ma,chi)+suffix+" ."
                #cmd = "scp tanlin@spark.ep.nctu.edu.tw:/data3/tanlin/MPS_config/measurements/proj0/N_{}/chi_{}/E1-N_{}-ma_{}-chi_{}.".format(N,chi,N,ma,chi)+suffix+" ."
                #cmd = "scp tanlin@spark.ep.nctu.edu.tw:/data3/tanlin/MPS_config/entro/N_{}/chi_{}/EE-N_{}-g_{}-ma_{}-chi_{}.".format(N,chi,N,g,ma,chi)+suffix+" ."
                cmd = "scp /home/davidtan/Desktop/plots/XXZ/N_{}/entro/EE-N_{}-g_{}-ma_{}.".format(N,N,g,ma)+suffix+" tanlin@spark.ep.nctu.edu.tw:/data3/tanlin/plots/entro"
                #cmd = "scp kcichy@loewe-csc.hhlr-gu.de:/data01/mesonqcd/kcichy/thirring/MPS_config/string_corr/string_corr-N_{}-g_{}-ma_{}-chi_{}.".format(N,g,ma,chi)+suffix+" ."
                #cmd = "scp kcichy@loewe-csc.hhlr-gu.de:/data01/mesonqcd/kcichy/thirring/MPS_config/entro/N_{}/chi_{}/EE-N_{}-g_{}-ma_{}-chi_{}.".format(N,chi,N,g,ma,chi)+suffix+" ."
                #cmd = "scp kcichy@loewe-csc.hhlr-gu.de:/data01/mesonqcd/kcichy/thirring/MPS_config/measurements/proj0/E1-N_{}-ma_{}-chi_{}.".format(N,ma,chi)+suffix+" ."
                ssh_cmd(cmd,password,timeout=180)
                #cmd = "scp tanlin@spark.ep.nctu.edu.tw:data/XXZ/measurements/N_{}/chi_{}/ccs-N_{}-ma_{}-chi_{}.".format(N,chi,N,ma,chi)+suffix+" ."
                #ssh_cmd(cmd,password,timeout=180)
                #print "copy file MPS-N_{}-g_{}-ma_{}-lambda_{}-Starget_{}-chi_{}-tol_{}.".format(N,g,ma,lamda,S_target,chi,tolerance)+suffix+" completed!"
            except IOError:
                #print "copy file MPS-N_{}-g_{}-ma_{}-lambda_{}-Starget_{}-chi_{}-tol_{}.".format(N,g,ma,lamda,S_target,chi,tolerance)+suffix+" failed"
                err=1
                pass

