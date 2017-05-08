# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:15:05 2017

@author: hazel.bain

    This module generates the PDFs for the
    Chen geoeffective magnetic cloud prediction Bayesian formulation. 
    
    P((Bzm, tau) n e|Bzm', tau' ; f) 
    = P(Bzm, tau|(Bzm, tau) n e ; f) * P(Bzm, tau|e) * P(e) / SUM_j( P(Bzm',tau' | c_j ; f) * P(c_j))
    
    where
    
    Bzm = actual value of Bz max for a magnetic cloud
    tau = actual duation of a magnetic cloud
    Bzm' = fitted/estimated value of Bz max at fraction (f) of an event
    tau' = fitted/estimated value of duration at fraction (f) of an event
    e = geoeffective event
    n = nongeoeffective event
    f = fraction of an event
    
    The data is stored as a pickle file and should be read in as:
    
        events_frac = pickle.load(open("events_frac.p","rb"))
    
    The top level create_pdfs function generates all the input PDFs
    and returns the posterior PDF P((Bzm, tau) n e|Bzm', tau' ; f) along with
    some other diagnostic variables. 
    
    Due to a relatively small smaple of geoeffective events, kernel density estimation
    is used to smooth the data and generate a non parametric PDFs.


"""


from sklearn.neighbors import KernelDensity
from scipy import stats
import pickle as pickle
import scipy.integrate as integrate
   
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D


def create_pdfs(events_frac, kernel_alg = 'scipy_stats', \
                ranges = [-150, 150, -250, 250], nbins = [50j, 100j],\
                ew = 2, nw = 0.5):

    """
    Create the PDFs for the
    Chen geoeffective magnetic cloud prediction Bayesian formulation. 
    
    P((Bzm, tau) n e|Bzm', tau' ; f) 
    = P(Bzm, tau|(Bzm, tau) n e ; f) * P(Bzm, tau|e) * P(e) / SUM_j( P(Bzm',tau' | c_j ; f) * P(c_j))
    
    where
    
    Bzm = actual value of Bz max for a magnetic cloud
    tau = actual duation of a magnetic cloud
    Bzm' = fitted/estimated value of Bz max at fraction (f) of an event
    tau' = fitted/estimated value of duration at fraction (f) of an event
    e = geoeffective event
    n = nongeoeffective event
    f = fraction of an event
       
    inputs:
        
    events_frac = dataframe
        contains output variables from a fit to solar wind magnetic field data
    kernel_alg = string
        choose between scikit learn and scipy stats python KDE algorithms
    ranges = 4 element array 
        defines the axis ranges for Bzm and tau [bmin, bmax, tmin, tmax]
    nbins = 2 elements array 
        defines the number of bins along bzm and tau [nbins_Bzm, nbins_tau]
    ew = float
        defines the kernel smoothing width for the geoeffective events
    nw = float
        defines the kernel smoothing width for the nongeoeffective events    
       
    
    """ 
    
    #width of smoothing box
    #ew = 2
    #nw = 0.5
    
    
    #range of Bzm and tau to define PDFs 
    #note - tmin is negative due to requirement to reflect raw data points in 
    #the tau = 0 axis to combat boundary effects when implementing kernel density
    #estimate smoothing
    #bmin = -150
    #bmax = 150
    #tmin = -250
    #tmax = 250
    
    #ranges = [bmin, bmax, tmin, tmax]
    
    #number of data bins in each dimension (dt takes into account the reflection)
    #db = 5j
    #dt = 10j
    
    #nbins = [db, dt]
    
    
    #create input PDFS
    Pe = P_e(events_frac)
    Pn = P_n(events_frac)
    Pbzm_tau_e, norm_bzm_tau_e = P_bzm_tau_e(events_frac, ranges=ranges, nbins=nbins)
    Pbzmp_taup_e, norm_bzmp_taup_e = P_bzmp_taup_e(events_frac, ranges=ranges, nbins=nbins)
    Pbzmp_taup_n, norm_bzmp_taup_n = P_bzmp_taup_n(events_frac, ranges=ranges, nbins=nbins)
    Pbzmp_taup_bzm_tau_e, norm_bzmp_taup_bzm_tau_e, P0 = P_bzmp_taup_bzm_tau_e(events_frac, ranges=ranges, nbins=nbins)
    
    Pbzm_tau_e_bzmp_taup, norm_bzm_tau_e_bzmp_taup, P1, P1_map       = P_bzm_tau_e_bzmp_taup(Pe, \
                                                    Pn,\
                                                    Pbzm_tau_e, \
                                                    Pbzmp_taup_e,\
                                                    Pbzmp_taup_n,\
                                                    Pbzmp_taup_bzm_tau_e)
    
    return Pbzm_tau_e_bzmp_taup, norm_bzm_tau_e_bzmp_taup, P0, P1, P1_map
    
    
def P_e(events_frac):  
    
    """
    Determine the prior PDF P(e) for input into the following Chen geoeffective 
    magnetic cloud prediction Bayesian formulation
    
    P((Bzm, tau) n e|Bzm', tau' ; f) 
    = P(Bzm, tau|(Bzm, tau) n e ; f) * P(Bzm, tau|e) * P(e) / SUM_j( P(Bzm',tau' | c_j ; f) * P(c_j))
    
    where c_j can be e or n
    
    Bzm = actual value of Bz max for a magnetic cloud
    tau = actual duation of a magnetic cloud
    Bzm' = fitted/estimated value of Bz max at fraction (f) of an event
    tau' = fitted/estimated value of duration at fraction (f) of an event
    e = geoeffective event
    n = nongeoeffective event
    f = fraction of an event
    
    inputs:
        
    events_frac = dataframe
        contains output variables from a fit to solar wind magnetic field data
        as a function of the fraction of time through an event
            
    """
    
    #----P(e) - probability of a geoefective event 
    #n_events = len(events_frac.iloc[np.where(events_frac.frac == 1.0)])
    #n_geoeff_events = len(events_frac.iloc[np.where((events_frac.frac == 1.0) & (events_frac.geoeff == 1.0))[0]])
    
    #values from Chen paper
    n_nongeoeff_events = 8600
    n_geoeff_events = 56
    n_events = n_nongeoeff_events + n_geoeff_events
    
    P_e = n_geoeff_events / n_events
    
    return P_e


def P_n(events_frac):  
    
    """
    Determine the prior PDF P(n) for input into the following Chen geoeffective 
    magnetic cloud prediction Bayesian formulation
    
    P((Bzm, tau) n e|Bzm', tau' ; f) 
    = P(Bzm, tau|(Bzm, tau) n e ; f) * P(Bzm, tau|e) * P(e) / SUM_j( P(Bzm',tau' | c_j ; f) * P(c_j))
    
    where c_j can be e or n
    
    Bzm = actual value of Bz max for a magnetic cloud
    tau = actual duation of a magnetic cloud
    Bzm' = fitted/estimated value of Bz max at fraction (f) of an event
    tau' = fitted/estimated value of duration at fraction (f) of an event
    e = geoeffective event
    n = nongeoeffective event
    f = fraction of an event
    
    inputs:
        
    events_frac = dataframe
        contains output variables from a fit to solar wind magnetic field data
        as a function of the fraction of time through an event
            
    """
    
    #----P(n) probability of nongeoeffective events
    #n_events = len(events_frac.iloc[np.where(events_frac.frac == 1.0)])
    #n_nongeoeff_events = len(events_frac.iloc[np.where((events_frac.frac == 1.0) & (events_frac.geoeff != 1.0))[0]])
    
    #values from Chen paper
    n_nongeoeff_events = 8600
    n_geoeff_events = 56
    n_events = n_nongeoeff_events + n_geoeff_events
    
    P_n = n_nongeoeff_events / n_events

    return P_n


def P_bzm_tau_e(events_frac, kernel_alg = 'scipy_stats', \
                ranges = [-150, 150, -250, 250], nbins = [50j, 100j],\
                kernel_width = 2, plotfig = 0):  
    
    """
    Determine the prior PDF P(Bzm, tau|e), the probability of geoeffective event with observered bzm and tau
    for input into the following Chen geoeffective magnetic cloud prediction Bayesian formulation
    
    P((Bzm, tau) n e|Bzm', tau' ; f) 
    = P(Bzm, tau|(Bzm, tau) n e ; f) * P(Bzm, tau|e) * P(e) / SUM_j( P(Bzm',tau' | c_j ; f) * P(c_j))
    
    where
    
    Bzm = actual value of Bz max for a magnetic cloud
    tau = actual duation of a magnetic cloud
    Bzm' = fitted/estimated value of Bz max at fraction (f) of an event
    tau' = fitted/estimated value of duration at fraction (f) of an event
    e = geoeffective event
    n = nongeoeffective event
    f = fraction of an event
    
    Due to a relatively small smaple of geoeffective events, kernel density estimation
    is used to smooth the data and generate a non parametric PDF. An artifact of 
    the KDE method is that the PDF will be smoothed to produce probabilities 
    extending to negative values of tau. To prevent this effect at the boundary 
    the raw data values are reflected in the tau = 0 axis and then the KDE is applied, 
    producing a symmetric density estimates about the axis of reflection - see 
    Silverman 1998 section 2.10. The required output PDF is obtained by selecting array 
    elements corresponding to postive values of tau 
    
    f'(x) = 2f(x) for x >= 0 and f'(x) = 0 for x < 0
    
    As a result of this reflection, the input range for tau extends to negative 
    values and nbins_tau is double nbins_bzm. The output PDF will be for 
    tau = [0, tmax]
    
    inputs:
        
    events_frac = dataframe
        contains output variables from a fit to solar wind magnetic field data
    kernel_alg = string
        choose between scikit learn and scipy stats python KDE algorithms        
    ranges = 4 element array 
        defines the axis ranges for Bzm and tau [bmin, bmax, tmin, tmax]
    nbins = 2 elements array 
        defines the number of bins along bzm and tau [nbins_Bzm, nbins_tau]
    kernel_widths = float
        defines the kernel smoothing width
    plotfit = int
        plot a figure of the distribution         
    
    """ 
    #range of Bzm and tau to define PDFs 
    bmin = ranges[0]
    bmax = ranges[1]
    tmin = ranges[2]
    tmax = ranges[3]
    
    #number of data bins in each dimension (dt takes into account the reflection)
    db = nbins[0]
    dt = nbins[1]
    
    #true boundary for tau and the corresponding index in the pdf array
    taumin = 0
    #dt0 = dt/2 
    
    #kernel smoothing width 
    ew = kernel_width
    
    #extract raw data points from dataframe of "actual" bzm and tau for geoeffective events
    #note bzm and tau do not change as a function of fraction of the events, so we use 
    #frac == 1.0, but any other fratcion could have been used
    gbzm = events_frac.bzm.iloc[np.where((events_frac.frac == 1.0) & (events_frac.geoeff == 1.0) & (events_frac.bzm < 0.0))[0]]
    gtau = events_frac.tau.iloc[np.where((events_frac.frac == 1.0) & (events_frac.geoeff == 1.0) & (events_frac.bzm < 0.0))[0]]

    #to handle boundary conditions and limit the density estimate to positve 
    #values of tau: reflect the data points along the tau axis, perform the 
    #density estimate and then set negative values of tau to 0 and double the 
    #density of positive values of tau
    gbzm_r = np.concatenate([gbzm, gbzm]) 
    gtau_r = np.concatenate([gtau, -gtau])
    
    #grid containing x, y positions 
    X_bzm, Y_tau = np.mgrid[bmin:bmax:db, tmin:tmax:dt]
    dt0 = int(len(Y_tau[1])/2.)
    Y_tau = Y_tau-((tmax-tmin)/len(Y_tau[1])/2.)      #to make sure the resulting PDF tau 
                                            #axis will start at 0. 

    #option to use scikit learn or scipy stats kernel algorithms
    if kernel_alg == 'sklearn':
        
        positions = np.vstack([X_bzm.ravel(), Y_tau.ravel()]).T
        values = np.vstack([gbzm_r, gtau_r]).T
        kernel_bzm_tau_e = KernelDensity(kernel='gaussian', bandwidth=ew).fit(values)
        Ptmp_bzm_tau_e = np.exp(np.reshape(kernel_bzm_tau_e.score_samples(positions).T, X_bzm.shape))
        
    elif kernel_alg == 'scipy_stats':
        
        positions = np.vstack([X_bzm.ravel(), Y_tau.ravel()])
        values = np.vstack([gbzm_r, gtau_r])
        kernel_bzm_tau_e = stats.gaussian_kde(values, bw_method = ew)
        Ptmp_bzm_tau_e = np.reshape(kernel_bzm_tau_e(positions).T, X_bzm.shape)

    #set the density estimate to 0 for negative tau, and x2 for positve tau 
    P_bzm_tau_e = Ptmp_bzm_tau_e[:,dt0::]*2

    #check the normalization, should normalize to 1
    b = X_bzm[:,0]
    t = Y_tau[dt0::]
    
    norm_bzm_tau_e = integrate.simps(integrate.simps(P_bzm_tau_e,Y_tau[0,dt0::]), X_bzm[:,0])
    print('\n\n Normalization for P_bzm_tau_e: ' + str(norm_bzm_tau_e) + '\n\n')

    if plotfig == 1:
        fig, ax = plt.subplots()
        c = ax.imshow(np.rot90(P_bzm_tau_e), extent=(bmin,bmax,taumin,tmax), cmap=plt.cm.gist_earth_r)
        ax.plot(gbzm, gtau, 'k.', markersize=4, label = 'bzm, tau, geoeff = 1')
        ax.set_xlim([bmin, bmax])
        ax.set_ylim([taumin, tmax])
        ax.set_xlabel('Bzm')
        ax.set_ylabel('Tau')
        ax.set_title('P_bzm_tau_e, bandwidth = '+str(ew))
        fig.colorbar(c)
        ax.legend(loc='upper right', prop = fontP, fancybox=True)


    return P_bzm_tau_e, norm_bzm_tau_e


def P_bzmp_taup_e(events_frac, kernel_alg = 'scipy_stats', \
                ranges = [-150, 150, -250, 250], nbins = [50j, 100j],\
                kernel_width = 2, plotfig = 0):  
    
    """
    Determine the PDF P(Bzm', tau'|e ; f), the probability of geoeffective event 
    with estimated values bzm' and tau' at fraction f throughout an event for 
    input into the following Chen geoeffective magnetic cloud prediction 
    Bayesian formulation.  This PDF contributes to the Bayesian "evidence".
    
    P((Bzm, tau) n e|Bzm', tau' ; f) 
    = P(Bzm, tau|(Bzm, tau) n e ; f) * P(Bzm, tau|e) * P(e) / SUM_j( P(Bzm',tau' | c_j ; f) * P(c_j))
    
    where
    
    Bzm = actual value of Bz max for a magnetic cloud
    tau = actual duation of a magnetic cloud
    Bzm' = fitted/estimated value of Bz max at fraction (f) of an event
    tau' = fitted/estimated value of duration at fraction (f) of an event
    e = geoeffective event
    n = nongeoeffective event
    f = fraction of an event
    
    Due to a relatively small smaple of geoeffective events, kernel density estimation
    is used to smooth the data and generate a non parametric PDF. An artifact of 
    the KDE method is that the PDF will be smoothed to produce probabilities 
    extending to negative values of tau. To prevent this effect at the boundary 
    the raw data values are reflected in the tau' = 0 axis and then the KDE is applied, 
    producing a symmetric density estimates about the axis of reflection - see 
    Silverman 1998 section 2.10. The required output PDF is obtained by selecting array 
    elements corresponding to postive values of tau 
    
    f'(x) = 2f(x) for x >= 0 and f'(x) = 0 for x < 0
    
    As a result of this reflection, the input range for tau extends to negative 
    values and nbins_tau is double nbins_bzm. The output PDF will be for 
    tau' = [0, tmax]
    
    inputs:
        
    events_frac = dataframe
        contains output variables from a fit to solar wind magnetic field data
    kernel_alg = string
        choose between scikit learn and scipy stats python KDE algorithms
    ranges = 4 element array 
        defines the axis ranges for Bzm and tau [bmin, bmax, tmin, tmax]
    nbins = 2 elements array 
        defines the number of bins along bzm and tau [nbins_Bzm, nbins_tau]
    kernel_width = float
        defines the kernel smoothing width
    plotfit = int
        plot a figure of the distribution 
        
    
    """ 
    #range of Bzm and tau to define PDFs 
    bmin = ranges[0]
    bmax = ranges[1]
    tmin = ranges[2]
    tmax = ranges[3]
    
    #number of data bins in each dimension (dt takes into account the reflection)
    db = nbins[0]
    dt = nbins[1]
    
    #true boundary for tau and the corresponding index in the pdf array
    taumin = 0
    #dt0 = dt/2 
    
    #kernel smoothing width
    ew = kernel_width
    
    #P_bzmp_taup_e is a function of the fraction of time f throughout an event
    #currently the fit to the data considers every 5th of an event
    for i in np.arange(6)*0.2:
    
        #extract raw data points from dataframe of estimates bzm' and tau' for 
        #fraction f throughout geoeffective events
        gbzmp = events_frac.bzm_predicted.iloc[np.where((events_frac.frac == i) & (events_frac.geoeff == 1.0))[0]]
        gtaup = events_frac.tau_predicted.iloc[np.where((events_frac.frac == i) & (events_frac.geoeff == 1.0))[0]]
        
        gbzmp.iloc[np.where(np.isnan(gbzmp))[0]] = 0.0
        gtaup.iloc[np.where(np.isnan(gtaup))[0]] = 0.0
        
        #to handle boundary conditions and limit the density estimate to positve 
        #values of tau': reflect the data points along the tau' axis, perform the 
        #density estimate and then set negative values of tau' to 0 and double the 
        #density of positive values of tau'
        gbzmp_r = np.concatenate([gbzmp, gbzmp])
        gtaup_r = np.concatenate([gtaup, -gtaup])
    
        #grid containing x, y positions 
        X_bzmp, Y_taup = np.mgrid[bmin:bmax:db, tmin:tmax:dt]
        dt0 = int(len(Y_taup[1])/2.)
        Y_taup = Y_taup-((tmax-tmin)/len(Y_taup[1])/2.)        #to make sure the resulting PDF tau 
                                                    #axis will start at 0. 
        
        #option to use scikit learn or scipy stats kernel algorithms
        if kernel_alg == 'sklearn':
        
            positions = np.vstack([X_bzmp.ravel(), Y_taup.ravel()]).T
            values = np.vstack([gbzmp_r, gtaup_r]).T             
            kernel_bzmp_taup_e = KernelDensity(kernel='gaussian', bandwidth=ew).fit(values)
            
            if i < 0.1:
                Ptmp_bzmp_taup_e = np.exp(np.reshape(kernel_bzmp_taup_e.score_samples(positions).T, X_bzmp.shape))
            else: 
                tmp = np.exp(np.reshape(kernel_bzmp_taup_e.score_samples(positions).T, X_bzmp.shape))
                Ptmp_bzmp_taup_e = np.dstack([Ptmp_bzmp_taup_e, tmp])
            
        elif kernel_alg == 'scipy_stats'  : 
            
            positions = np.vstack([X_bzmp.ravel(), Y_taup.ravel()])
            values = np.vstack([gbzmp_r, gtaup_r])
            kernel_bzmp_taup_e = stats.gaussian_kde(values, bw_method=ew)
        
            if i < 0.1:
                Ptmp_bzmp_taup_e = np.reshape(kernel_bzmp_taup_e(positions).T, X_bzmp.shape)
            else: 
                tmp = np.reshape(kernel_bzmp_taup_e(positions).T, X_bzmp.shape)
                Ptmp_bzmp_taup_e = np.dstack([Ptmp_bzmp_taup_e, tmp])
       
    #set the density estimate to 0 for negative tau, and x2 for positve tau 
    P_bzmp_taup_e = Ptmp_bzmp_taup_e[:,dt0::,:]*2

    #check the normalization
    norm_bzmp_taup_e = integrate.simps(integrate.simps(P_bzmp_taup_e[:,:,5],Y_taup[0,dt0::]), X_bzmp[:,0])
    print('\n\n Normalization for P_bzmp_taup_e: ' + str(norm_bzmp_taup_e) + '\n\n')

    if plotfig == 1:                   
        fig, ax = plt.subplots()
        c = ax.imshow(np.rot90(P_bzmp_taup_e[:,:,5]), extent=(bmin,bmax,taumin,tmax), cmap=plt.cm.gist_earth_r)
        ax.plot(gbzmp, gtaup, 'k.', markersize=4, label = 'bzm_p, tau_p, geoeff = 1')
        ax.set_xlim([bmin, bmax])
        ax.set_ylim([taumin, tmax])
        ax.set_xlabel('Bzm')
        ax.set_ylabel('Tau')
        ax.set_title('P_bzmp_taup_e, bandwidth = '+str(ew))
        fig.colorbar(c)
        ax.legend(loc='upper right', prop = fontP, fancybox=True)

    return P_bzmp_taup_e, norm_bzmp_taup_e



def P_bzmp_taup_n(events_frac, kernel_alg = 'scipy_stats', \
                ranges = [-150, 150, -250, 250], nbins = [50j, 100j],\
                kernel_width = 0.5, plotfig = 0):  
    
    """
    Determine the PDF P(Bzm', tau'|n ; f), the probability of nongeoeffective event 
    with estimated values bzm' and tau' at fraction f throughout an event for 
    input into the following Chen geoeffective magnetic cloud prediction 
    Bayesian formulation. This PDF contributes to the Bayesian "evidence".
    
    P((Bzm, tau) n e|Bzm', tau' ; f) 
    = P(Bzm, tau|(Bzm, tau) n e ; f) * P(Bzm, tau|e) * P(e) / SUM_j( P(Bzm',tau' | c_j ; f) * P(c_j))
    
    where
    
    Bzm = actual value of Bz max for a magnetic cloud
    tau = actual duation of a magnetic cloud
    Bzm' = fitted/estimated value of Bz max at fraction (f) of an event
    tau' = fitted/estimated value of duration at fraction (f) of an event
    e = geoeffective event
    n = nongeoeffective event
    f = fraction of an event
    
    Due to a relatively small smaple of events, kernel density estimation
    is used to smooth the data and generate a non parametric PDF. An artifact of 
    the KDE method is that the PDF will be smoothed to produce probabilities 
    extending to negative values of tau. To prevent this effect at the boundary 
    the raw data values are reflected in the tau' = 0 axis and then the KDE is applied, 
    producing a symmetric density estimates about the axis of reflection - see 
    Silverman 1998 section 2.10. The required output PDF is obtained by selecting array 
    elements corresponding to postive values of tau 
    
    f'(x) = 2f(x) for x >= 0 and f'(x) = 0 for x < 0
    
    As a result of this reflection, the input range for tau extends to negative 
    values and nbins_tau is double nbins_bzm. The output PDF will be for 
    tau' = [0, tmax]
    
    inputs:
        
    events_frac = dataframe
        contains output variables from a fit to solar wind magnetic field data
    kernel_alg = string
        choose between scikit learn and scipy stats python KDE algorithms
    ranges = 4 element array 
        defines the axis ranges for Bzm and tau [bmin, bmax, tmin, tmax]
    nbins = 2 elements array 
        defines the number of bins along bzm and tau [nbins_Bzm, nbins_tau]
    kernel_width = float
        defines the kernel smoothing width    
    plotfit = int
        plot a figure of the distribution 
        
    
    """ 
    #range of Bzm and tau to define PDFs 
    bmin = ranges[0]
    bmax = ranges[1]
    tmin = ranges[2]
    tmax = ranges[3]
    
    #number of data bins in each dimension (dt takes into account the reflection)
    db = nbins[0]
    dt = nbins[1]
    
    #true boundary for tau and the corresponding index in the pdf array
    taumin = 0
    #dt0 = dt/2 
    
    #kernel smoothing width
    nw = kernel_width
    
    #P_bzmp_taup_n is a function of the fraction of time f throughout an event
    #currently the fit to the data considers every 5th of an event
    for i in np.arange(6)*0.2:

        #extract raw data points from dataframe of estimates bzm' and tau' for 
        #fraction f throughout nongeoeffective events
        #note geoeff label can have values between 0-3. Anything that is not 1 is
        #considered nongeoeffective
        
        #gbzm_n = events_frac.bzm.iloc[np.where((events_frac.frac == i) & (events_frac.geoeff != 1.0))[0]]
        #gtau_n = events_frac.tau.iloc[np.where((events_frac.frac == i) & (events_frac.geoeff != 1.0))[0]]
        
        gbzmp_n = events_frac.bzm_predicted.iloc[np.where((events_frac.frac == i) & (events_frac.geoeff != 1.0))[0]]
        gtaup_n = events_frac.tau_predicted.iloc[np.where((events_frac.frac == i) & (events_frac.geoeff != 1.0))[0]]
        
        gbzmp_n.iloc[np.where(np.isnan(gbzmp_n))[0]] = 0.0
        gtaup_n.iloc[np.where(np.isnan(gtaup_n))[0]] = 0.0
 
        #to handle boundary conditions and limit the density estimate to positve 
        #values of tau': reflect the data points along the tau' axis, perform the 
        #density estimate and then set negative values of tau' to 0 and double the 
        #density of positive values of tau'
        gbzmp_nr = np.concatenate([gbzmp_n, gbzmp_n])
        gtaup_nr = np.concatenate([gtaup_n, -gtaup_n])
    
        X_bzmp_n, Y_taup_n = np.mgrid[bmin:bmax:db, tmin:tmax:dt]
        dt0 = int(len(Y_taup_n[1])/2.)
        Y_taup_n = Y_taup_n-((tmax-tmin)/len(Y_taup_n[1])/2.)    #to make sure the resulting PDF tau 
                                                    #axis will start at 0. 
        
        #option to use scikit learn or scipy stats kernel algorithms
        if kernel_alg == 'sklearn':
        
            positions = np.vstack([X_bzmp_n.ravel(), Y_taup_n.ravel()]).T
            values = np.vstack([gbzmp_nr, gtaup_nr]).T
            kernel_bzmp_taup_n = KernelDensity(kernel='gaussian', bandwidth=nw).fit(values)
            
            if i < 0.1:
                Ptmp_bzmp_taup_n = np.exp(np.reshape(kernel_bzmp_taup_n.score_samples(positions).T, X_bzmp_n.shape))
            else: 
                tmp = np.exp(np.reshape(kernel_bzmp_taup_n.score_samples(positions).T, X_bzmp_n.shape))
                Ptmp_bzmp_taup_n = np.dstack([Ptmp_bzmp_taup_n, tmp])
            
        elif kernel_alg == 'scipy_stats'  :
            
            positions = np.vstack([X_bzmp_n.ravel(), Y_taup_n.ravel()])
            values = np.vstack([gbzmp_nr, gtaup_nr])  
            kernel_bzmp_taup_n = stats.gaussian_kde(values, bw_method=nw)
        
            if i < 0.1:
                Ptmp_bzmp_taup_n = np.reshape(kernel_bzmp_taup_n(positions).T, X_bzmp_n.shape)
            else: 
                tmp = np.reshape(kernel_bzmp_taup_n(positions).T, X_bzmp_n.shape)
                Ptmp_bzmp_taup_n = np.dstack([Ptmp_bzmp_taup_n, tmp])
   
    #set the density estimate to 0 for negative tau, and x2 for positve tau 
    P_bzmp_taup_n = Ptmp_bzmp_taup_n[:,dt0::,:]*2

    #check the normalization
    norm_bzmp_taup_n = integrate.simps(integrate.simps(P_bzmp_taup_n[:,:,5],Y_taup_n[0,dt0::]), X_bzmp_n[:,0])
    print('\n\n Normalization for P_bzmp_taup_n: ' + str(norm_bzmp_taup_n) + '\n\n')

    if plotfig == 1:
        fig, ax = plt.subplots()
        c = ax.imshow(np.rot90(P_bzmp_taup_n[:,:,5]), extent=(bmin,bmax,taumin,tmax), cmap=plt.cm.gist_earth_r)
        ax.plot(gbzmp_n, gtaup_n, 'k.', markersize=4, label = 'bzm_p, tau_p, geoeff = 0')
        ax.set_xlim([bmin, bmax])
        ax.set_ylim([taumin, tmax])
        ax.set_xlabel('Bzm')
        ax.set_ylabel('Tau')
        ax.set_title('P_bzmp_taup_n, bandwidth = '+str(nw))
        fig.colorbar(c)
        ax.legend(loc='upper right', prop = fontP, fancybox=True)

    return P_bzmp_taup_n, norm_bzmp_taup_n


def P_bzmp_taup_bzm_tau_e(events_frac, kernel_alg = 'scipy_stats', \
                ranges = [-150, 150, -250, 250], nbins = [50j, 100j],\
                kernel_width = 2, plotfig = 0):  
    
    """
    Determine the prior PDF P(Bzm', tau'|(Bzm, tau) n e ; f), the probability 
    of a geoeffective event with estimates Bzm' and tau' for a MC with actual 
    values Bzm and tau, at fraction f throughout an event for input into the following 
    Chen geoeffective magnetic cloud prediction Bayesian formulation. This is 
    the bayesian likelihood PDF, relating the model to the data.  
    
    P((Bzm, tau) n e|Bzm', tau' ; f) 
    = P(Bzm, tau|(Bzm, tau) n e ; f) * P(Bzm, tau|e) * P(e) / SUM_j( P(Bzm',tau' | c_j ; f) * P(c_j))
    
    where
    
    Bzm = actual value of Bz max for a magnetic cloud
    tau = actual duation of a magnetic cloud
    Bzm' = fitted/estimated value of Bz max at fraction (f) of an event
    tau' = fitted/estimated value of duration at fraction (f) of an event
    e = geoeffective event
    n = nongeoeffective event
    f = fraction of an event
    
    Due to a relatively small smaple of geoeffective events, kernel density estimation
    is used to smooth the data and generate a non parametric PDF. An artifact of 
    the KDE method is that the PDF will be smoothed to produce probabilities 
    extending to negative values of tau. To prevent this effect at the boundary 
    the raw data values are reflected in the tau' = 0 axis and then the KDE is applied, 
    producing a symmetric density estimates about the axis of reflection - see 
    Silverman 1998 section 2.10. The required output PDF is obtained by selecting array 
    elements corresponding to postive values of tau 
    
    f'(x) = 2f(x) for x >= 0 and f'(x) = 0 for x < 0
    
    As a result of this reflection, the input range for tau extends to negative 
    values and nbins_tau is double nbins_bzm. The output PDF will be for 
    tau' = [0, tmax]
    
    inputs:
        
    events_frac = dataframe
        contains output variables from a fit to solar wind magnetic field data
    kernel_alg = string
        choose between scikit learn and scipy stats python KDE algorithms
    ranges = 4 element array 
        defines the axis ranges for Bzm and tau [bmin, bmax, tmin, tmax]
    nbins = 2 elements array 
        defines the number of bins along bzm and tau [nbins_Bzm, nbins_tau]
    kernel_width = float
        defines the kernel smoothing width
    plotfit = int
        plot a figure of the distribution 
        
    
    """ 
    #range of Bzm and tau to define PDFs 
    bmin = ranges[0]
    bmax = ranges[1]
    tmin = ranges[2]
    tmax = ranges[3]
    
    #number of data bins in each dimension (dt takes into account the reflection)
    db = nbins[0]
    dt = nbins[1]
    
    #true boundary for tau and the corresponding index in the pdf array
    taumin = 0
    #dt0 = dt/2 
    
    #kernel smoothing width
    ew = kernel_width
    
    #hack to get array size
    X_bzmp, Y_taup, XX_bzm, YY_tau = np.mgrid[bmin:bmax:db, tmin:tmax:dt, bmin:bmax:db, tmin:tmax:dt]   
    db2 = int(len(Y_taup[:,0,:,:]))
    dt2 = int(len(Y_taup[0,:,:,:]))
    
    #P_bzmp_taup_bzm_tau_e is a function of the fraction of time f throughout an event
    #currently the fit to the data considers every 5th of an event 
    Ptmp_bzmp_taup_bzm_tau_e = np.zeros((db2,dt2,db2,dt2,6))
    for i in np.arange(6)*0.2:
        
        #extract raw data points from dataframe of estimates bzm' and tau' for 
        #fraction f throughout eoeffective events
        gbzmp = events_frac.bzm_predicted.iloc[np.where((events_frac.frac == i) & (events_frac.geoeff == 1.0))[0]]
        gtaup = events_frac.tau_predicted.iloc[np.where((events_frac.frac == i) & (events_frac.geoeff == 1.0))[0]]
        
        gbzmp.iloc[np.where(np.isnan(gbzmp))[0]] = 0.0
        gtaup.iloc[np.where(np.isnan(gtaup))[0]] = 0.0
        
        #extract raw data points from dataframe of estimates bzm and tau for 
        #fraction f throughout eoeffective events
        gbzm = events_frac.bzm.iloc[np.where((events_frac.frac == i) & (events_frac.geoeff == 1.0))[0]]
        gtau = events_frac.tau.iloc[np.where((events_frac.frac == i) & (events_frac.geoeff == 1.0))[0]]
    
        #to handle boundary conditions and limit the density estimate to positve 
        #values of tau' and tau: reflect the data points along the tau' and tau
        #axes, perform the ensity estimate and then set negative values of 
        #tau' and tau to 0 and x4 the density of positive values of tau'
        gbzmp_r = np.concatenate([gbzmp, gbzmp, gbzmp, gbzmp])
        gtaup_r = np.concatenate([gtaup, -gtaup, gtaup, -gtaup])        
        gbzm_r = np.concatenate([gbzm, gbzm, gbzm, gbzm]) 
        gtau_r = np.concatenate([gtau, gtau, -gtau, -gtau])
        
        #grid containing x, y positions 
        X_bzmp, Y_taup, XX_bzm, YY_tau = np.mgrid[bmin:bmax:db, tmin:tmax:dt, bmin:bmax:db, tmin:tmax:dt]
        dt0 = int(len(Y_taup[1])/2)
        Y_taup = Y_taup-((tmax-tmin)/len(Y_taup[1])/2.)
        YY_tau = YY_tau-((tmax-tmin)/len(Y_taup[1])/2.)
        
        #option to use scikit learn or scipy stats kernel algorithms
        if kernel_alg == 'sklearn':
            
            positions = np.vstack([X_bzmp.ravel(), Y_taup.ravel(), XX_bzm.ravel(), YY_tau.ravel()]).T
            values = np.vstack([gbzmp_r, gtaup_r, gbzm_r, gtau_r]).T        
            kernel_bzmp_taup_bzm_tau_e = KernelDensity(kernel='gaussian', bandwidth=ew).fit(values)
            Ptmp_bzmp_taup_bzm_tau_e[:,:,:,:,int(i*5)] = np.exp(np.reshape(kernel_bzmp_taup_bzm_tau_e.score_samples(positions).T, X_bzmp.shape))
            
        elif kernel_alg == 'scipy_stats':    
            
            positions = np.vstack([X_bzmp.ravel(), Y_taup.ravel(), XX_bzm.ravel(), YY_tau.ravel()])
            values = np.vstack([gbzmp_r, gtaup_r, gbzm_r, gtau_r])            
            kernel_bzmp_taup_bzm_tau_e = stats.gaussian_kde(values, bw_method = ew)
            Ptmp_bzmp_taup_bzm_tau_e[:,:,:,:,int(i*5)] = np.reshape(kernel_bzmp_taup_bzm_tau_e(positions).T, X_bzmp.shape)
 
    #set the density estimate to 0 for negative tau, and x4 for positve tau 
    P_bzmp_taup_bzm_tau_e = Ptmp_bzmp_taup_bzm_tau_e[:,dt0::,:,dt0::,:]*4             

    #check the normalization of the 4D space   
    bp = X_bzmp[:,0,0,0]
    tp = Y_taup[0,dt0::,0,0]
    b = XX_bzm[0,0,:,0]
    t = YY_tau[0,0,0,dt0::]
    
    norm_bzmp_taup_bzm_tau_e = integrate.simps(integrate.simps(integrate.simps(integrate.simps(P_bzmp_taup_bzm_tau_e[:,:,:,:,5],
                                tp),\
                                bp), \
                                t),\
                                b)
    
    #check the normalization of the 4D space - should be 1   
    predicted_duration = 15.0
    predicted_bzmax = -26.0
    indt = np.min(np.where(t > predicted_duration))
    indb = np.max(np.where(b < predicted_bzmax))
    
    
    P0 = integrate.simps(integrate.simps(P_bzmp_taup_bzm_tau_e[:,:,indb,indt,5],\
                                t),\
                                b)  
    
    print('\n\n Normalization for P_bzmp_taup_bzm_tau_e: ' + str(norm_bzmp_taup_bzm_tau_e) )
    print('\n Normalization for P_bzmp_taup_bzm_tau_e plane Bzm =-26nT, tau = 15 hrs, frac = 1.0: ' \
          + str(P0) + '\n\n')
                                
    if plotfig == 1:
        fig, ax = plt.subplots()
        c = ax.imshow(np.rot90(P_bzmp_taup_bzm_tau_e[:,:,20,3,5]), extent=(bmin,bmax,taumin,tmax), cmap=plt.cm.gist_earth_r)
        #ax.plot(gbzm, gtau, 'k.', markersize=4, c='r')
        #ax.plot(gbzm_n, gtau_n, 'k.', markersize=4, c='b')
        #ax.set_xlim([bmin, bmax])
        #ax.set_ylim([taumin, tmax])
        ax.set_xlabel('Bzm')
        ax.set_ylabel('Tau')
        fig.colorbar(c)

    
    
    ############## FOLLOWING CODE IS MESSING ABOUT TO GET THE PLANES TO NORM TO 1 #############

#==============================================================================
#     #normalize each plane
#     P_bzmp_taup_bzm_tau_e2 = np.zeros((50,50,50,50,6))
#     for i in np.arange(6):
#         for j in range(50):
#             for k in range(50):
#                 
#                 tmp = P_bzmp_taup_bzm_tau_e[:,:,j,k,i]
#                 tmpsum = integrate.simps(integrate.simps(tmp, t), b)   
#                 #P_bzmp_taup_bzm_tau_e2[:,:,j,k,i] = tmp/tmpsum/((b[1]-b[0]) * (t[1]-t[0]))
#                 P_bzmp_taup_bzm_tau_e2[:,:,j,k,i] = tmp *(50*(b[1]-b[0]) * 50*(t[1]-t[0]))
#                 
#                 #if np.isnan(integrate.simps(integrate.simps(P_bzmp_taup_bzm_tau_e2[:,:,j,k,i], t), b)):
#                     #print(j,k,i,integrate.simps(integrate.simps(P_bzmp_taup_bzm_tau_e2[:,:,j,k,i],\
#                     #            t),\
#                     #            b) )
#==============================================================================

    #voxel = (49*(b[1]-b[0]) * 49*(t[1]-t[0]))
    #voxel = 300*250
    #P_bzmp_taup_bzm_tau_e2 = P_bzmp_taup_bzm_tau_e * voxel
    
    #P_bzmp_taup_bzm_tau_e2 = P_bzmp_taup_bzm_tau_e * (49*49)
    
#==============================================================================
#     P_bzmp_taup_bzm_tau_e2 = np.zeros((db2,db2,db2,db2,6))
#     for j in range(db2):
#         for k in range(db2):
#             for i in range(6):
#                 P_bzmp_taup_bzm_tau_e2[:,:,j,k,i] = P_bzmp_taup_bzm_tau_e[:,:,j,k,i] * 1/ P_bzmp_taup_bzm_tau_e[:,:,j,k,i]
#==============================================================================
    
    #P0_2 = integrate.simps(integrate.simps(P_bzmp_taup_bzm_tau_e2[:,:,20,3,5],\
    #                            t),\
    #                            b)          
    #print(P0_2)            
    
    ############## END MESSING ABOUT WITH CODE #############
    
    return P_bzmp_taup_bzm_tau_e, norm_bzmp_taup_bzm_tau_e, P0    


def P_bzm_tau_e_bzmp_taup(P_e, P_n, P_bzm_tau_e, P_bzmp_taup_e, P_bzmp_taup_n, \
                P_bzmp_taup_bzm_tau_e, plotfig = 0):  
    
    """
    Determine the posterior PDF P((Bzm, tau) n e |Bzm', tau' ; f), the probability 
    of a geoeffective event with parameters Bzm and tau given estimates  
    Bzm' and tau', at fraction f throughout an event for input into the following 
    Chen geoeffective magnetic cloud prediction Bayesian formulation. This is 
    the bayesian posterior PDF.
    
   P((Bzm, tau) n e|Bzm', tau' ; f) 
    = P(Bzm, tau|(Bzm, tau) n e ; f) * P(Bzm, tau|e) * P(e) / SUM_j( P(Bzm',tau' | c_j ; f) * P(c_j))
    
    where
    
    Bzm = actual value of Bz max for a magnetic cloud
    tau = actual duation of a magnetic cloud
    Bzm' = fitted/estimated value of Bz max at fraction (f) of an event
    tau' = fitted/estimated value of duration at fraction (f) of an event
    e = geoeffective event
    n = nongeoeffective event
    f = fraction of an event
       
    inputs:
    
    P_e = float
        Prior P(e)
    P_n = float
        Prior P(n)
    P_bzm_tau_e = data array
        Prior PDF P(bzm, tau|e)
    P_bzmp_taup_e = data array
        Evidence PDF P(bzm', tau'|e; f)
    P_bzmp_taup_n = data array
        Evidence PDF P(bzm', tau'|n; f)
    P_bzmp_taup_bzm_tau_e = data array
        Likelihood PDF P(Bzm', tau' | (Bzm, tau) n e ; f)
        
    output:
        
    P_bzm_tau_e_bzmp_taup = data array
        Posterior PDF P((Bzm, tau) n e | Bzm', tau' ; f)
    
        
    """ 
    #range of Bzm and tau to define PDFs 
    bmin = ranges[0]
    bmax = ranges[1]
    tmin = ranges[2]
    tmax = ranges[3]
    
    #hack to get array size
    X_bzmp, Y_taup, XX_bzm, YY_tau = np.mgrid[bmin:bmax:db, tmin:tmax:dt, bmin:bmax:db, tmin:tmax:dt]   
    db2 = int(len(Y_taup[:,0,:,:]))
    dt2 = int(len(Y_taup[0,:,:,:]))
    
    #true boundary for tau and the corresponding index in the pdf array
    taumin = 0
    #dt0 = dt/2 
    
    #P_bzm_tau_e_bzmp_taup_e is a function of the fraction of time f throughout an event
    #currently the fit to the data considers every 5th of an event     
    P_bzm_tau_e_bzmp_taup = np.zeros((db,dt2,db,dt2,6))
    for i in np.arange(6)*0.2:

        num = np.multiply(P_bzmp_taup_bzm_tau_e[:,:,:,:,int(i*5)], P_bzm_tau_e) * P_e
        denom = (P_bzmp_taup_e[:,:,int(i*5)] * P_e) + (P_bzmp_taup_n[:,:,int(i*5)] * P_n)
        
        P_bzm_tau_e_bzmp_taup[:,:,:,:,int(i*5)] = np.divide(num, denom)
        
        #num = np.multiply(Ptmp_bzmp_taup_bzm_tau_e[:,:,:,:,int(i*5)], Ptmp_bzm_tau_e) * P_e
        #denom = (Ptmp_bzmp_taup_e[:,:,int(i*5)] * P_e) + (Ptmp_bzmp_taup_n[:,:,int(i*5)] * P_n)
        #Ptmp_bzm_tau_e_bzmp_taup[:,:,:,:,int(i*5)] = np.divide(num, denom)

    #check the normalization of the 4D space 
    norm_bzm_tau_e_bzmp_taup = integrate.simps(integrate.simps(integrate.simps(integrate.simps(P_bzm_tau_e_bzmp_taup[:,:,:,:,5],\
                                tp),\
                                bp), \
                                t),\
                                b)
    #check the normalization of the 4D space - should be 1   
    # corresponds to P1 in Chen 97 paper
    b = XX_bzm[0,0,:,0]
    t = YY_tau[0,0,0,dt0::]
    
    predicted_duration = 15.0
    predicted_bzmax = -26.0
    indt = np.min(np.where(t > predicted_duration))
    indb = np.max(np.where(b < predicted_bzmax))
    
    P1 = integrate.simps(integrate.simps(P_bzm_tau_e_bzmp_taup[:,:,indb,indt,5],\
                                t),\
                                b)

    #map of P1 for all planes
    P1_map = np.zeros((db,dt/2))
    for j in range(dt):
        for k in range(db):
            P1_map[j,k] = integrate.simps(integrate.simps(P_bzm_tau_e_bzmp_taup[:,:,j,k,5],\
                                t),\
                                b)
    
    print('\n\n Normalization for P_bzm_tau_e_bzmp_taup: ' + str(norm_bzm_tau_e_bzmp_taup) )
    print('\n Normalization for P_bzm_tau_e_bzmp_taup plane Bzmp =-26nT, taup = 15 hrs, frac = 1.0: ' \
          + str(P1) + '\n\n')
    print('\n\n Max P1_map: ' + str(P1_map.max()) )
    
    if plotfig == 1: 
        fig, ax = plt.subplots()
        c = ax.imshow(np.rot90(P_bzm_tau_e_bzmp_taup[:,:,20,3,5]), extent=(bmin,bmax,taumin,tmax), cmap=plt.cm.gist_earth_r)
        #ax.imshow(np.rot90(num[:,:,20,3]), extent=(bmin,bmax,taumin,tmax), cmap=plt.cm.gist_earth_r)
        ax.set_xlim([bmin, bmax])
        ax.set_ylim([taumin, tmax])
        ax.set_xlabel('Bzm')
        ax.set_ylabel('Tau')
        fig.colorbar(c)


    return P_bzm_tau_e_bzmp_taup, norm_bzm_tau_e_bzmp_taup, P1, P1_map    