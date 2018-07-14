#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 21:35:46 2018

@author: davidtan
"""

import numpy as np
import itertools
import pandas as pd
from scipy import optimize
import warnings
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase

class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
                           linestyle=orig_handle[1], color='k')
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], 
                           color=orig_handle[0])
        return [l1, l2]

def read_N_extrap_table(path,key):
    df = pd.read_csv(path)
    key_values = df[key].values
    return key_values

def write_ma_extrap_table(mas, g_idx):
    ccs_mean = []; ccs_std = []
    mg_mean = []; mg_std = []
    for ma in mas:
        path = '/home/davidtan/Desktop/MPS_config/XXZ/N_extrap/'
        try:
            inf_mean = read_N_extrap_table(path+'/N_extrap-g_{}-ma_{}.csv'.format(gs[g_idx],ma),'ccs_mean')[-1]
            inf_std = read_N_extrap_table(path+'/N_extrap-g_{}-ma_{}.csv'.format(gs[g_idx],ma),'ccs_std')[-1]
            ccs_mean.append(inf_mean); ccs_std.append(inf_std)
        except:
            ccs_mean.append(np.nan); ccs_std.append(np.nan)
        try:
            inf_mean = read_N_extrap_table(path+'/N_extrap-g_{}-ma_{}.csv'.format(gs[g_idx],ma),'mg_mean')[-1]
            inf_std = read_N_extrap_table(path+'/N_extrap-g_{}-ma_{}.csv'.format(gs[g_idx],ma),'mg_std')[-1]
            mg_mean.append(inf_mean); mg_std.append(inf_std)
        except:
            mg_mean.append(np.nan); mg_std.append(np.nan)    
            
            
    dict = {"ma": mas, "ccs_mean": ccs_mean, "ccs_std": ccs_std,
                       "mg_mean": mg_mean, "mg_std": mg_std}
    df = pd.DataFrame(dict)
    mas += [0]
    out = zero_ma_fit(df,"ccs")
    ccs_mean.append(out[0]); ccs_std.append(out[1])
    out = zero_ma_fit(df,"mg")
    mg_mean.append(out[0]); mg_std.append(out[1])
    dict = {"ma": mas, "ccs_mean": ccs_mean, "ccs_std": ccs_std,
                       "mg_mean": mg_mean, "mg_std": mg_std}
    df = pd.DataFrame(dict)
    return df

def zero_ma_fit(df,key):
    mas = df['ma'].values
    mean = df[key+'_mean'].values
    std = df[key+'_std'].values   
    if np.isnan(sum(mean)):
        inf_mean = np.nan
        inf_std = np.nan
    else:
        fitfunc = lambda p, x: p[0] + p[1] * x #+ p[2] * x ** 2
        errfunc = lambda p, x, y, err: ((y - fitfunc(p, x)) / err) ** 2
        p_init = [1.0, 1.0]
        out = optimize.leastsq(errfunc, p_init, args=(np.array(mas), mean, std), full_output=1)

        p_final = out[0]
        covar = out[1]
        inf_mean = p_final[0]
        chi2 = np.sum(errfunc(p_final,np.array(mas),mean,std))
        if covar is not None:
            inf_std = np.sqrt(covar[0][0])
        else:
            warnings.simplefilter("always")
            warnings.warn("ValueWarning: Encounter singular matrix in scipy.optimize.leastsq. Try to fit without error estimation.")
            coeff = np.polyfit(np.array(mas),mean,1)
            inf_mean = coeff[1]
            inf_std = 1e-7
        if key == 'ccs' and chi2 > 1.:
            inf_std = np.sqrt(chi2/2.) * inf_std
    return [inf_mean, inf_std]

def read_ma_extrap_table(path,key):
    df = pd.read_csv(path)
    key_values = df[key].values
    return key_values

if __name__=='__main__':

    gs = np.linspace(-1.0,1.0,101)
    gs = np.array(gs.tolist()[5:20]+gs.tolist()[20::10])
    mas = [0.1,0.2,0.3,0.4]
    
    N = 800
    Ns = np.linspace(400,1000,4,dtype=int).tolist()
    g_idx = 10
    chis = [300,400,500]
    chi = 500

    """
    for g_idx in xrange(len(gs)):
        mas = [0.1,0.2,0.3,0.4]
        path = '/home/davidtan/Desktop/MPS_config/XXZ/ccs_ma_extrap'
        df = write_ma_extrap_table(mas, g_idx)
        df.to_csv(path+'/ma_extrap-g_{}.csv'.format(gs[g_idx]))
        print gs[g_idx],'\n',df
    """
    
    fontsize = 28; labelsize = 22
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    marker = itertools.cycle(('^', 'D', 'o', 's')) 
    """
    glist = []
    fig = plt.figure(figsize=(12,9))
    fig.patch.set_alpha(0.0)
    colors = [plt.cm.jet(i) for i in np.linspace(0, 1, 11)]
    i = 0
    for g_idx in [2,4,6,7,8,9,10,11,12,15,17]:
        ccs = []
        for N,ma in [(200,0.4),(400,0.2),(800,0.1)]:
            path = '/home/davidtan/Desktop/MPS_config/XXZ/chi_extrap'     
            ccs.append(np.load(path+'/ccs-N_{}-ma_{}-chi_{}.npy'.format(N,ma,chi))[g_idx])     
           
        glist.append(gs[g_idx])
        plt.plot(1./np.array(Ns),ccs,marker='o',linestyle='None',color=colors[i])
        m, b = np.polyfit(1./np.array(Ns),ccs,1)
        x = np.linspace(0,1./200,100)
        plt.plot(x,m*x+b,marker='None',linestyle='-',color=colors[i])
        i += 1
    plt.xlabel(r'$1/N$',fontsize=fontsize)
    #plt.ylabel(r'$|\sum_{n}(-1)^{n}\langle c_{n}^{\dagger}c_{n}\rangle|/N$',fontsize=fontsize)
    plt.ylabel(r'$\chi$',fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    plt.xlim(-1e-6,0.0052)
    plt.ylim(-1e-4,0.35)
    legendlist=map(str,glist)
    legendlist=[r'$\Delta(g) = $'+s for s in legendlist]
    #plt.legend((legendlist),loc=0,numpoints=1,fontsize=labelsize,ncol=2) 
    plt.legend([object], legendlist,handler_map={object: AnyObjectHandler()},fontsize=labelsize)
    ax=plt.gca()
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.74, 0.88, r'$m_0 L = 80$',transform=ax.transAxes,bbox=props,fontsize=fontsize)
    plt.grid()
    plt.show()
    """
    """
    colors = [plt.cm.tab10(i) for i in np.linspace(0, 1, 11)]
    cc = itertools.cycle(colors)
    glist = [-0.76,-0.72,-0.7,-0.68,-0.64,-0.62,-0.6,-0.4,-0.2,0.0]
    fig = plt.figure(figsize=(12,9))
    fig.patch.set_alpha(0.0)
    plotlist = [] 
    for g in glist:
        ccs = []
        for ma in mas:
            g_idx = np.argmin(np.abs(gs-g))
            path = '/home/davidtan/Desktop/MPS_config/XXZ/chi_extrap'     
            ccs.append(np.load(path+'/ccs-N_{}-ma_{}-chi_{}.npy'.format(N,ma,chi))[g_idx])   
        m, b = np.polyfit(mas,ccs,1)
        x = np.linspace(0.,0.45,100)
        
        plt.hold(True)
        c = next(cc)
        p1 = plt.plot(mas,ccs,marker='o',linestyle='None',color=c)
        p2 = plt.plot(x,m*x+b,marker='None',linestyle='-',color=c)
        plotlist.append(p1[0])
    plt.xlabel(r'$m_{0}a$',fontsize=fontsize)
    plt.ylabel(r'$\chi$',fontsize=fontsize) 
    plt.tick_params(labelsize=labelsize)
    plt.xlim(-1e-6,0.45)
    plt.ylim(-1e-4,0.38)#ccs[-1]+0.1)
    legendlist=map(str,glist)
    legendlist=[r'$\Delta(g) = $'+s for s in legendlist]
    plt.legend(plotlist,(legendlist),loc=0,numpoints=1,fontsize=labelsize,ncol=2)   
    ax=plt.gca()
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.6, 0.08, r'$N = {}, D = {}$'.format(N,chi),transform=ax.transAxes,bbox=props,fontsize=fontsize)
    plt.grid()  
    plt.show()
    """
    """
    for N in [600,800]:
        for g_idx in xrange(len(gs)):
            for ma in mas:
    #N = 1000; g_idx = 9; ma = 0.1
                fig = plt.figure(figsize=(12,9))
                fig.patch.set_alpha(0.0)         
                path = '/home/davidtan/Desktop/MPS_config/XXZ/chi_extrap/table'
                ccs = read_ma_extrap_table(path+'/chi_extrap-N_{}-g_{}-ma_{}.csv'.format(N,gs[g_idx],ma),'ccs')[:4]
                chis = read_ma_extrap_table(path+'/chi_extrap-N_{}-g_{}-ma_{}.csv'.format(N,gs[g_idx],ma),'chi')[:4]
                
                plt.plot(1./chis.astype(np.float),ccs,marker="o",linestyle="-")   
                plt.xlabel('$1/D$',fontsize=fontsize)
                plt.ylabel(r'$\chi$',fontsize=fontsize)
                plt.xlim(-1e-4,1./280)
                plt.tick_params(labelsize=labelsize)
                ax=plt.gca()
                props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                plt.text(0.1, 0.88, r'$N = {}, \Delta(g) = {}, \tilde{{m}}_0 a = {}$'.format(N,gs[g_idx],ma),transform=ax.transAxes,bbox=props,fontsize=fontsize)
                plt.grid()
                plt.savefig('/home/davidtan/Desktop/plots/XXZ/ccs_ma_extrap/N_extrap/chi_extrap/chi_extrap-N_{}-g_{}-ma_{}.pdf'.format(N,gs[g_idx],ma))
                plt.close()
    #plt.show()   
    """
    """
    for g_idx in xrange(len(gs)):
        for ma in [0.0,0.1,0.2,0.3,0.4]:
    #g_idx = 17; ma = 0.0
            fig = plt.figure(figsize=(12,9))
            fig.patch.set_alpha(0.0)
            mean = []; std = []
            for N in Ns:
                path = '/home/davidtan/Desktop/MPS_config/XXZ/chi_extrap/table'
                mean_value, std_value = read_ma_extrap_table(path+'/chi_extrap-N_{}-g_{}-ma_{}.csv'.format(N,gs[g_idx],ma),'ccs')[-2:]
                mean.append(mean_value); std.append(std_value)
            
            fitfunc = lambda p, x: p[0] + p[1] * x
            errfunc = lambda p, x, y, err: ((y - fitfunc(p, x)) / err) ** 2
            p_init = [1.0, 1.0]
            out = optimize.leastsq(errfunc, p_init, args=(1./np.array(Ns), mean, std), full_output=1)

            p_final = out[0]
            covar = out[1]
            inf_mean = p_final[0]
            chi2 = np.sum(errfunc(p_final,1./np.array(Ns),mean,std))
            if covar is not None:
                inf_std = np.sqrt(covar[0][0])
            else:
                warnings.simplefilter("always")
                warnings.warn("ValueWarning: Encounter singular matrix in scipy.optimize.leastsq. Try to fit without error estimation.")
                coeff = np.polyfit(1./np.array(Ns),mean,1)
                inf_mean = coeff[1]
                inf_std = 1e-7    
            if chi2 > 1.:
                inf_std = np.sqrt(chi2/2.) * inf_std
    
            x = np.linspace(0.,1./400,100)
            plt.errorbar(1./np.array(Ns),mean,yerr=std,fmt=marker.next(),elinewidth=2,capsize=8,capthick=2,markersize=8) 
            plt.errorbar([0.0],inf_mean,yerr=inf_std,fmt=marker.next(),elinewidth=2,capsize=8,capthick=2,markersize=8)
            plt.plot(x,p_final[1]*x+p_final[0],marker="None",linestyle='-')
            plt.xlabel('$1/N$',fontsize=fontsize)
            plt.ylabel(r'$\hat{\chi}$',fontsize=fontsize)
            #plt.ylabel(r'$E_1-E_0$',fontsize=fontsize)
            plt.tick_params(labelsize=labelsize)
            ax=plt.gca()
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            plt.text(0.5, 0.08, r'$\Delta(g) = {}, \tilde{{m}}_0 a = {}$'.format(gs[g_idx],ma),transform=ax.transAxes,bbox=props,fontsize=fontsize)
            plt.grid()
            plt.legend(("linear fit","data","linear fit result"),loc=0,numpoints=1,fontsize=fontsize,markerscale=1.5)
            #ax.yaxis.offsetText.set_fontsize(labelsize)
            plt.savefig('/home/davidtan/Desktop/plots/XXZ/ccs_ma_extrap/N_extrap/N_extrap-g_{}-ma_{}.pdf'.format(gs[g_idx],ma))
            plt.close()
            #plt.show()
    """
    
    quantity = "ccs"
    meanlist = []; stdlist = []
    for g_idx in xrange(len(gs)):
        #fig = plt.figure(figsize=(12,9))
        #fig.patch.set_alpha(0.0)         
        path = '/home/davidtan/Desktop/MPS_config/XXZ/ccs_ma_extrap'
        mean = read_ma_extrap_table(path+'/ma_extrap-g_{}.csv'.format(gs[g_idx]),quantity+'_mean')[:]
        std = read_ma_extrap_table(path+'/ma_extrap-g_{}.csv'.format(gs[g_idx]),quantity+'_std')[:]
            
        mas = [0.0,0.1,0.2,0.3,0.4]
        fitfunc = lambda p, x: p[0] + p[1] * x
        errfunc = lambda p, x, y, err: ((y - fitfunc(p, x)) / err) ** 2
        p_init = [1.0, 1.0]
        out = optimize.leastsq(errfunc, p_init, args=(np.array(mas)[1:4], mean[:3], std[:3]), full_output=1)
        p_final = out[0]
        covar = out[1]
        std1 = np.sqrt(np.linalg.norm(covar))
        chi2_1 = np.sum(errfunc(p_final,np.array(mas)[1:4],mean[:3],std[:3]))
        if chi2_1 > 1:
            std1 *= np.sqrt(chi2_1/2.)
        
        fitfunc = lambda p, x: p[0] + p[1] * x + p[2] * x**2
        errfunc = lambda p, x, y, err: ((y - fitfunc(p, x)) / err) ** 2
        p_init = [1.0, 1.0, 1.0]
        out = optimize.leastsq(errfunc, p_init, args=(np.array(mas)[1:5], mean[:4], std[:4]), full_output=1)        
        p_final2 = out[0]
        covar2 = out[1]
        std2 = np.sqrt(np.linalg.norm(covar2))
        chi2_2 = np.sum(errfunc(p_final2,np.array(mas)[1:5],mean[:4],std[:4]))
        if chi2_2 > 1: 
            std2 *= np.sqrt(chi2_2/2.)
        
        mean_value = p_final[0]
        meanlist.append(mean_value)
        std_value = np.sqrt(std1**2 + abs(p_final[0]-p_final2[0])**2)
        stdlist.append(std_value)
        """
        plt.errorbar(mas[1:],mean[:4],yerr=std[:4],fmt=marker.next(),elinewidth=2,capsize=8,capthick=2,markersize=10)  
        plt.errorbar(mas[0],p_final[0],yerr=std1,fmt=marker.next(),elinewidth=2,capsize=8,capthick=2,markersize=10,color='g')
        plt.errorbar(mas[0],p_final2[0],yerr=std2,fmt=marker.next(),elinewidth=2,capsize=8,capthick=2,markersize=10,color='r')
        x = np.linspace(0.,0.3,100)
        plt.plot(x,p_final[0]+p_final[1]*x,marker="None",linestyle='-',color='g')
        x = np.linspace(0.,0.4,100)
        plt.plot(x,p_final2[0]+p_final2[1]*x++p_final2[2]*x**2,marker="None",linestyle='-',color='r')
        plt.xlabel(r'$\tilde{{m}}_{0}a$',fontsize=fontsize)
        plt.ylabel(r'$\hat{\chi}$',fontsize=fontsize)
        #plt.ylabel(r'$E_1-E_0$',fontsize=fontsize)
        plt.tick_params(labelsize=labelsize)
        ax=plt.gca()
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        plt.text(0.1, 0.88, r'$\Delta(g) = {}$'.format(gs[g_idx]),transform=ax.transAxes,bbox=props,fontsize=fontsize)
        plt.grid()
        plt.legend(("linear fit","quadratic fit","data","linear fit result","quadratic fit result"),loc=4,numpoints=1,fontsize=fontsize,markerscale=1.5)
        plt.savefig('/home/davidtan/Desktop/plots/XXZ/ccs_ma_extrap/ma_extrap-g_{}.pdf'.format(gs[g_idx]))
        plt.close()
        #plt.show()
        """
    
    marker = itertools.cycle(('^', 'D', 'o', 's', 'v')) 
    quantity = "ccs"
    fig = plt.figure(figsize=(12,9))
    fig.patch.set_alpha(0.0)
    #mean = []; std = []
    #for g_idx in xrange(len(gs)):            
    #    path = '/home/davidtan/Desktop/MPS_config/XXZ/ccs_ma_extrap'      
    #    mean_value = read_ma_extrap_table(path+'/ma_extrap-g_{}.csv'.format(gs[g_idx]),quantity+'_mean')[-1]
    #    std_value = read_ma_extrap_table(path+'/ma_extrap-g_{}.csv'.format(gs[g_idx]),quantity+'_std')[-1]
    #    mean.append(mean_value); std.append(std_value)

    plt.errorbar(gs,meanlist,yerr=stdlist,fmt=marker.next(),elinewidth=2,capsize=8,capthick=2,markersize=12,markerfacecolor='None')    
    for ma in [0.1,0.2,0.3,0.4]:
        mean = []; std = []      
        for g in gs:
            path = '/home/davidtan/Desktop/MPS_config/XXZ/N_extrap'
            mean_value = read_N_extrap_table(path+'/N_extrap-g_{}-ma_{}.csv'.format(g,ma),quantity+'_mean')[-1]
            std_value = read_N_extrap_table(path+'/N_extrap-g_{}-ma_{}.csv'.format(g,ma),quantity+'_std')[-1]
            mean.append(mean_value); std.append(std_value)

        plt.errorbar(gs,mean,yerr=std,fmt=marker.next(),elinewidth=2,capsize=8,capthick=2,markersize=12,markerfacecolor='None')   
    
    plt.xlabel('$\Delta(g)$',fontsize=fontsize)
    plt.ylabel(r'$\hat{\chi}$',fontsize=fontsize)
    #plt.ylabel(r'$E_1 -E_0$',fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)
    plt.legend((r"$\tilde{{m}}_{0} a \to 0.0$",r"$\tilde{{m}}_{0} a = 0.1$",r"$\tilde{{m}}_{0} a = 0.2$",r"$\tilde{{m}}_{0} a = 0.3$",r"$\tilde{{m}}_{0} a = 0.4$"),loc=0,ncol=1,numpoints=1,fontsize=fontsize)
    plt.grid()
    plt.show()
    