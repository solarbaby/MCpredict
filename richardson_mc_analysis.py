# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 12:05:15 2017

@author: hazel.bain
"""

import Richardson_ICME_list as icme
import read_database as rddb

import plot_ace as pa
import MCpredict as MC
import read_dst as dst

import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import datetime

import matplotlib.pyplot as plt

def MCpredict_all_Richardson(plotting = 0):
    
    """
    Tests the Chen magnetic cloud fitting model using known magnetic cloud events
    from Richardson and Cane list. 
    
    inputs
    
    plotting: int
        outputs plots of the solar wind plasma and magnetic field data, along 
        with values of the Dst. Indicated whether or not the event is geoeffective
        (red), non geoeffective (green) or ambigous (orange)
    
    """
    
    

    #read in Richardson and Cane ICME list 
    mc_list = icme.read_richardson_icme_list()
    
    #read in the dst data
    dst_data = dst.read_dst_df()
    
    #choose icmes that have MC_flag classification 2 and 1
    #
    # (2) = indicates that a magnetic cloud has been reported in association with
    #the ICME (see (d) above) or (occasionally, or for recent events) that by 
    #our assessment, the ICME has the clear features of a magnetic cloud but a 
    #magnetic cloud may not have been reported. 
    #
    # (1) indicates that the ICME shows evidence of a rotation in field direction
    #but lacks some other characteristics of a magnetic cloud, for example an
    #enhanced magnetic field
    
    mc_list12 = mc_list[np.where(mc_list['MC_flag'] == 2)]

    #run the Chen magnetic cloud fitting routine to obtain fits to solar wind 
    #events to predict durtaion and max/min Bzm. 
    
    events = pd.DataFrame()             #observational event characteristics for all MCs
    events_frac = pd.DataFrame()        #predicted events characteristics split into fraction of an event
    
    errpredict = []                     # keep a note of any events where there were errors
    
    for i in range(0,len(mc_list12['mc_start'])):
        
        if mc_list12['mc_start'][i] == None:
            continue
        
        #known events with data issues. Still to fix this
        if i == 3:
            continue
        if i == 49:
            continue
        if i == 58:
            continue
        if i == 96:
            continue
        
        
        #get mc times +/- 24 hours
        st = mc_list12['mc_start'][i] - timedelta(hours=24)
        et = mc_list12['mc_start'][i] + timedelta(hours=48)
        
        #format time strings
        stf = datetime.strftime(st, "%Y-%m-%d")
        etf = datetime.strftime(et, "%Y-%m-%d")
        
        #run the MC fit and prediction routine
        try:

            data,events_tmp, events_frac_tmp = MC.Chen_MC_Prediction(stf, etf, dst_data[st - timedelta(1):et + timedelta(1)], smooth_num = 100,\
                line = [mc_list12['mc_start'][i], mc_list12['mc_end'][i]], plotting = plotting,\
                plt_outfile = 'mcpredict_'+ datetime.strftime(mc_list12['mc_start'][i], "%Y-%m-%d_%H%M") + '.pdf' ,\
                plt_outpath = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/richardson_mcpredict_plots_2_smooth100_bzmfix/')
            
                            
            print("----appending event----")
            events = events.append(events_tmp)
            events_frac = events_frac.append(events_frac_tmp)
            
        except:
            errpredict.append(i)
  
            
    events = events.reset_index() 

    #drop duplicate events 
    events_uniq = events.drop_duplicates('start')       
            
    print("--------Error predict------------")
    print(errpredict)

    #plot_obs_bz_tau(events_uniq, 'bzm_vs_tau_smooth100.pdf')
    #plot_predict_bz_tau_frac(events_frac, outname = 'bztau_predict.pdf')
    
    return events_uniq, events_frac



def plot_predict_bz_tau_frac(events_frac, outname = 'bztau_predict.pdf'):
    
    """
    
    Plots the fitted magnetic cloud predicted Bzm vs predicted duration tau as
    a function of the fraction of the event. 
    
    input
    
    events: dataframe
        dataframe containing events determined from historical data as a fraction
        of the event 
    
    """
    
    
    from matplotlib.font_manager import FontProperties
        
    ##plot Bzm vs tau
    w2 = np.where((events_frac['frac'] == 0.2) & (events_frac['geoeff'] == 1.0))[0]
    w4 = np.where((events_frac['frac'] == 0.4) & (events_frac['geoeff'] == 1.0))[0]
    w6 = np.where((events_frac['frac'] == events_frac.frac.iloc[3]) & (events_frac['geoeff'] == 1.0))[0]
    w8 = np.where((events_frac['frac'] == 0.8) & (events_frac['geoeff'] == 1.0))[0]
    w10 = np.where((events_frac['frac'] == 1.0) & (events_frac['geoeff'] == 1.0))[0]
    
    bt = events_frac['bzm'].iloc[w2]*events_frac['tau'].iloc[w2]  
                                 
    bt2_predict = events_frac['bzm_predicted'].iloc[w2]*events_frac['tau_predicted'].iloc[w2]  
    bt4_predict = events_frac['bzm_predicted'].iloc[w4]*events_frac['tau_predicted'].iloc[w4]  
    bt6_predict = events_frac['bzm_predicted'].iloc[w6]*events_frac['tau_predicted'].iloc[w6]  
    bt8_predict = events_frac['bzm_predicted'].iloc[w8]*events_frac['tau_predicted'].iloc[w8] 
    bt10_predict = events_frac['bzm_predicted'].iloc[w10]*events_frac['tau_predicted'].iloc[w10] 


    fontP = FontProperties()                #legend
    fontP.set_size('medium')                       
                           
    plt.scatter(bt, bt2_predict, c = 'purple', label = '0.2 event')
    plt.scatter(bt, bt4_predict, c = 'b', label = '0.4 event')
    plt.scatter(bt, bt6_predict, c = 'g', label = '0.6 event')
    plt.scatter(bt, bt8_predict, c = 'orange', label = '0.8 event')  
    plt.scatter(bt, bt8_predict, c = 'r', label = '1.0 event')                         
    
    plt.plot(bt, bt, c = 'black')      
    
    #plt.ylim(0,60)
    #plt.xlim(-60,60)
    plt.title("BzmTau obs vs predicted as fraction \n of event duration (Geoeff = 1)")
    plt.xlabel("$\mathrm{B_{zm} tau (obs)}$")
    plt.ylabel("$\mathrm{B_{zm} tau (predict)}$")
    leg = plt.legend(loc='upper left', prop = fontP, fancybox=True, \
                     frameon=True, scatterpoints = 1 )
    leg.get_frame().set_alpha(0.5)
    
    #plt.show()
    plt.savefig(outname, format='pdf')
    
    plt.close()
    
    return None
    
def plot_obs_bz_tau(events, outname = 'bzm_vs_tau.pdf'):
    
    """
    Plots the magnetic cloud actual bzm vs tau
    
    input
    
    events: dataframe
        dataframe containing events determined from historical data
    
    
    """
    
    from matplotlib.font_manager import FontProperties
        
    ##plot Bzm vs tau
    w_geoeff = np.where(events['geoeff'] == 1.0)[0]
    w_no_geoeff = np.where(events['geoeff'] == 0)[0]

                           
    fontP = FontProperties()                #legend
    fontP.set_size('medium')                       
                           
                           
    plt.scatter(events['bzm'].iloc[w_no_geoeff], events['tau'].iloc[w_no_geoeff], c = 'b', label = 'Not Geoeffecive')                         
    plt.scatter(events['bzm'].iloc[w_geoeff], events['tau'].iloc[w_geoeff], c = 'r', label = 'Geoeffective')
    plt.ylim(0,60)
    plt.xlim(-60,60)
    plt.xlabel("$\mathrm{B_{zm}}$ (nT)")
    plt.ylabel("Duration (hr)")
    leg = plt.legend(loc='upper right', prop = fontP, fancybox=True, \
                     frameon=True, scatterpoints = 1 )
    leg.get_frame().set_alpha(0.5)
    
    plt.savefig(outname, format='pdf')
    
    plt.close()
    
def plot_all_Richardson_MC():

    #read in Richardson and Cane ICME list 
    mc_list = icme.read_richardson_icme_list()
    
    #choose icmes that have MC_flag classification 2 and 1
    #
    # (2) = indicates that a magnetic cloud has been reported in association with
    #the ICME (see (d) above) or (occasionally, or for recent events) that by 
    #our assessment, the ICME has the clear features of a magnetic cloud but a 
    #magnetic cloud may not have been reported. 
    #
    # (1) indicates that the ICME shows evidence of a rotation in field direction
    #but lacks some other characteristics of a magnetic cloud, for example an
    #enhanced magnetic field
    
    mc_list12 = mc_list[np.where(mc_list['MC_flag'] > 0)]

    #get the ace_mag_1m and ace_swepam_1m data for these events
    
    errplt = []
    for i in range(0,len(mc_list12['mc_start'])):
        
        if mc_list12['mc_start'][i] == None:
            continue
        
        
        #get mc times +/- 24 hours
        st = mc_list12['mc_start'][i] - timedelta(hours=24)
        et = mc_list12['mc_start'][i] + timedelta(hours=48)
        
        #format time strings
        stf = datetime.strftime(st, "%Y-%m-%d")
        etf = datetime.strftime(et, "%Y-%m-%d")
        
        try:

            pa.plot_ace_dst(stf, etf,\
                line = [mc_list12['mc_start'][i], mc_list12['mc_end'][i]],\
                plt_outfile = 'ace_dst_'+ datetime.strftime(mc_list12['mc_start'][i], "%Y-%m-%d_%H%M") + '.pdf')
            
        except:
            print("\nErrorPlot. recording event index and moving on\n")
            errplt.append(i)
        
        

            