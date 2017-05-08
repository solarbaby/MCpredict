# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 09:29:08 2017

@author: hazel.bain

 Chen_MC_Prediction.pro

 This module contains functions to read in either real time or historical 
 solar wind data and determine geoeffective and non geoeffective "events" 
 present in the magnetic field data. An event is defined to be > 120 minutes 
 long and with start and end times determined when Bz component of the magentic
 field changes sign. 
 
 A sinusoid is fitted to the Bz component to predict the event Bz maximum 
 (or minimum) value and the event duration, therefore giving some predictive 
 diagnostic of whether of not we expect the event to be geoeffective.
 
 When applied to historical data, classification of the geoeffectiveness of the events is 
 determined by comparison with the Dst value during that event. If Dst < -80 
 the event is considered to be geoeffective. If the Dst is > -80 then the event
 is considered to be non geoeffective. Events occuring in the wake of geoeffective
 events, where the Dst is still recovering are classed as ambigous. 
 
 The top level function Chen_MC_Prediction is called to run the model either to
 real-time data or to a historical dataset e.g.
 
 data, events, events_frac = Chen_MC_Prediction(start_date, end_date, dst_data, pdf)


 The original version of this code is from Jim Chen and Nick Arge
 and is called DOACE_hr.pro. This version is the python translation of
 IDL code written by Michele Cash of DOACE.pro modifed to 
 run in near real-time, to read in data from the SWPC database,
 and to make the code more readable by removing goto statements
 and 5 minute data averages.

 Reference Papers: Chen et al. 1996, 1997, 2012
                   Arge et al. 2002

 INPUTS
 sdate = start date in the format '2011-09-09'
 edate = end date in the format '2011-09-10'


"""

from read_database import get_data

import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta

def Chen_MC_Prediction(sdate, edate, dst_data, pdf, smooth_num = 25, pdf = pdf, resultsdir='', \
                       real_time = 0, spacecraft = 'ace',\
                       plotting = 1, plt_outfile = 'mcpredict.pdf',\
                       plt_outpath = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/richardson_mcpredict_plots/',\
                       line = [], dst_thresh = -80):

"""
 This function read in either real time or historical 
 solar wind data and determine geoeffective and non geoeffective "events" 
 present in the magnetic field data. An event is defined to be > 120 minutes 
 long and with start and end times determined when Bz component of the magentic
 field changes sign. Classification of the geoeffectiveness of the events is 
 determined by comparison with the Dst value during that event. If Dst < -80 
 the event is considered to be geoeffective. If the Dst is > -80 then the event
 is considered to be non geoeffective. Events occuring in the wake of geoeffective
 events, where the Dst is still recovering are classed as ambigous. 


"""


    
    #running in real_time mode
    if real_time == 1:
        print("todo: real-time data required - determine dates for last 24 hours")
    
    print("Start date: " + sdate )
    print("End date  : " + edate + "\n")
    
    #min_duration of at least 120 minutes in order to be considered as an event
    min_duration=120.
    
    #read in mag and solar wind data
    if spacecraft == 'ace':
        
        #read in ace_mag_1m data
        mag_data = get_data(sdate, edate, view = 'ace_mag_1m')
        
        sw_data = get_data(sdate, edate, view = 'ace_swepam_1m')
        
        #convert to pandas DataFrame
        #MAYBE MOVE THIS STEP INTO THE GET DATA FUNCTION!!!!
        mag = pd.DataFrame(data = mag_data, columns = mag_data.dtype.names)
        sw = pd.DataFrame(data = sw_data, columns = sw_data.dtype.names)
        
        
    elif spacecraft == 'dscovr':
        print("todo: dscovr data read functions still todo")
        
    #clean data
    mag_clean, sw_clean = clean_data(mag, sw)
    
    #Create stucture to hold smoothed data
    col_names = ['date', 'bx', 'by', 'bz', 'bt']        
    data = pd.concat([mag_clean['date'], \
            pd.Series(mag_clean['gsm_bx']).rolling(window = smooth_num).mean(), \
            pd.Series(mag_clean['gsm_by']).rolling(window = smooth_num).mean(),\
            pd.Series(mag_clean['gsm_bz']).rolling(window = smooth_num).mean(), \
            pd.Series(mag_clean['bt']).rolling(window = smooth_num).mean()], axis=1, keys = col_names)
    
    data['theta_z'] = pd.Series(180.*np.arcsin(mag_clean['gsm_bz']/mag_clean['bt'])/np.pi)\
                        .rolling(window = smooth_num).mean()   #in degrees
    data['theta_y'] = pd.Series(180.*np.arcsin(mag_clean['gsm_by']/mag_clean['bt'])/np.pi)\
                        .rolling(window = smooth_num).mean()   #in degrees

    data['sw_v'] = pd.Series(sw_clean['v']).rolling(window = smooth_num).mean()
    data['sw_n'] = pd.Series(sw_clean['n']).rolling(window = smooth_num).mean()    

    #add empty columns for the predicted data values at each step in time
    data['istart'] = 0
    data['iend'] = 0
    data['tau_predicted'] = 0
    data['tau_actual'] = 0
    data['bzm_predicted'] = 0
    data['i_bzmax'] = 0
    data['bzm_actual'] = 0
    
    #Incrementally step through the data and look for mc events.
    #An event is defined as the time bounded by sign changes of Bz.
    #An event needs to have a min_durtaion of min_duration.
    #Once a valid event is found, predict it's duration 

    ##check to make sure starting Bz data isn't NaN
    first_good_bz_data = np.min(np.where(np.isnan(data['bz']) == False)) + 1
    
    iend = first_good_bz_data
    iStartNext = first_good_bz_data
    
    for i in np.arange(first_good_bz_data, len(data['date'])):
        istart = iStartNext
        
        #check for bz sign change to signal end of an event, if not
        #move on to next data step
        
        if not event_end(data, i):
            continue 
        
        #print("----Event found----")
        
        iend = i-1
        iStartNext = i
        
        #now we have an event, check that is meets the min_duration
        #if not move to the next data step and look for new event
        if not long_duration(istart, iend, min_duration):
            continue
        
        #ignore the first input data step
        if istart == first_good_bz_data:
            continue

        #print("----Event of correct duration found----")
        
        #print('\n Sign changed: Bz[i-1] = ' + \
        #    str(data['bz'][i-1]) + \
        #    ', Bz[i] =' + str(data['bz'][i]) + \
        #    ', i = ' + str(i) + \
        #    ', hours = ' + str((iend-istart)/90.) )#hours
        
        start_date  = datetime.strftime(data['date'][i], "%Y-%m-%d %H:%M")
        #print('start date: ' + start_date + '\n')
        
        #now try and predict the duration
        predict_duration(data, istart, iend, pdf)
        
        if icme_event(istart, iend, len(data['date'])):
            validation_stats, data, resultsdir, istart, iend
    

    
    #create new dataframe to record event characteristics
    events, events_frac = create_event_dataframe(data, dst_data)
        
    #plot some stuff   
    if plotting == 1:
        evt_times = events[['start','end']].values
        mcpredict_plot(data, events_frac, dst_data, line=line, bars = evt_times, plt_outfile = plt_outfile, plt_outpath = plt_outpath)
    
    return data, events, events_frac



def create_event_dataframe(data, dst_data, t_frac = 5):

    #start and end times for each event
    #evt_times, evt_indices = find_event_times(data)
    evt_indices = np.transpose(np.array([data['istart'].drop_duplicates().values[1::], \
                                         data['iend'].drop_duplicates().values[1::]]))
    
    #start data frame to record each event's characteristics    
    evt_col_names = ['start', 'bzm', 'tau', 'istart', 'iend']        
    events = pd.concat([data['date'][evt_indices[:,0]],\
                    data['bzm_actual'][evt_indices[:,0]],\
                    data['tau_actual'][evt_indices[:,0]],\
                    data['istart'][evt_indices[:,0]],\
                    data['iend'][evt_indices[:,0]] ], axis=1, keys = evt_col_names)
    events['end'] =  data['date'][evt_indices[:,1]].values  #needs to be added separately due to different index

    #get min dst and geoeffective flags
    events = dst_geo_tag(events, dst_data, dst_thresh = -80, dst_dur_thresh = 2.0)
    
    #split the event into fractions for bayesian stats
    events_frac = events.loc[np.repeat(events.index.values, t_frac+1)]
    events_frac.reset_index(inplace=True)
    
    #remame the column headers to keep track of things
    events_frac.rename(columns={'level_0':'evt_index', 'index':'data_index'}, inplace=True)
    
    frac = pd.DataFrame({'frac':np.tile(np.arange(t_frac+1)*(100/t_frac/100), len(events)),\
                            'bzm_predicted':0.0,\
                            'tau_predicted':0.0,\
                            'i_bzmax':0})
    
    events_frac = pd.concat([events_frac, frac], axis = 1)  
    
    ##bzm at each fraction of an event
    for i in range(len(evt_indices)):
        
        #determine the indices in data for each fraction of an event
        frac_ind = evt_indices[i,0] + (np.arange(t_frac+1)*(100/t_frac/100) * \
                    float(evt_indices[i,1]-evt_indices[i,0])).astype(int)
        
        events_frac['bzm_predicted'].iloc[np.where(events_frac['evt_index'] == i)] = data['bzm_predicted'].iloc[frac_ind].values
        events_frac['tau_predicted'].iloc[np.where(events_frac['evt_index'] == i)] = data['tau_predicted'].iloc[frac_ind].values
        events_frac['i_bzmax'].iloc[np.where(events_frac['evt_index'] == i)] = data['i_bzmax'].iloc[frac_ind].values
        


    return events, events_frac
    
    
    
def mcpredict_plot(data, events_frac, dst_data, line= [], bars = [], plot_fit = 1, dst_thresh = -80, \
            plt_outpath = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/richardson_mcpredict_plots_2/',\
            plt_outfile = 'mcpredict.pdf'):
    
    """
    Plot the ACE_MAG_1m and ACE_SWEPAM_1M data

    Parameters
    ----------
    tstart : string, required 
        Start time for the database query.
    tend: string, required 
        End time for the database query.
    server: string, optional
        default server is swds-st
    database: string, optional
        default database is RA
    view: string, optional
        default view is ace_mag_1m
    csv: int, optional
        output csv file keyword. default is 0
    outpath: string, optional
        csv file path

    Returns
    -------
    None
    
    """
    
    import read_dst as dst
    
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.dates import DayLocator
    from matplotlib.dates import HourLocator
    from matplotlib.dates import DateFormatter
            
    #start and end times for plot to make sure all plots are consistent
    #st = datetime.strptime(data['date'][0]), "%Y-%m-%d")
    #et = datetime.strptime(data['date'][-1], "%Y-%m-%d")

    st = data['date'][0]
    et = data['date'].iloc[-1]
    
    #read in the dst data
    #dst_data = dst.read_dst(str(st), str(et))

    #plot the ace data
    f, (ax0, ax1, ax1b, ax2, ax3, ax4, ax5, ax6) = plt.subplots(8, figsize=(11,10))
 
    plt.subplots_adjust(hspace = .1)       # no vertical space between subplots
    fontP = FontProperties()                #legend
    fontP.set_size('medium')
    
    dateFmt = DateFormatter('%d-%b')
    hoursLoc = HourLocator()
    daysLoc = DayLocator()
    
    color = {0.0:'green', 1.0:'red', 2.0:'grey',3.0:'orange'}
    fitcolor = {0.2:'purple', 0.4:'blue', events_frac.frac.iloc[3]:'green',0.8:'orange', 1.0:'red'}
  
    #----By
    ax0.plot(data['date'], data['by'], label='By (nT)')
    ax0.hlines(0.0, data['date'][0], data['date'].iloc[-1], linestyle='--',color='grey')
    ax0.set_xticklabels(' ')
    ax0.xaxis.set_major_locator(daysLoc)
    ax0.xaxis.set_minor_locator(hoursLoc)
    ax0.set_xlim([st, et])
    for l in line:
        ax0.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for b in range(len(bars)):
        ax0.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15)        
    leg = ax0.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #----Bz
    ax1.plot(data['date'], data['bz'], label='Bz (nT)')
    ax1.hlines(0.0, data['date'][0], data['date'].iloc[-1], linestyle='--',color='grey')
    ax1.set_xticklabels(' ')
    ax1.xaxis.set_major_locator(daysLoc)
    ax1.xaxis.set_minor_locator(hoursLoc)
    ax1.set_xlim([st, et])
    for l in line:
        ax1.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for b in range(len(bars)):
        ax1.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
    leg = ax1.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #plot the position of max bz
    for i in np.arange(0, len(events_frac), 6):
        if (events_frac['geoeff'].iloc[i] == 1.0):
            wmax_bz = np.where( data['bz'].iloc[events_frac['istart'].iloc[i] : events_frac['iend'].iloc[i]] == events_frac['bzm'].iloc[i])[0]

            ax1.axvline(x=data['date'].iloc[events_frac['istart'].iloc[i] + wmax_bz].values[0], \
                     linewidth=1, linestyle='--', color='grey')

    #plot the fitted profile at certain intervals through the event  
    if plot_fit == 1:
        for i in range(len(events_frac)): 
            #only plot the fits for the geoeffective events
            if (events_frac['geoeff'].iloc[i] == 1.0) & (events_frac['frac'].iloc[i] >0.1):
                 
                #for each fraction of an event, determine the current fit to the profile up to this point
                pred_dur = events_frac['tau_predicted'].iloc[i] * 60.
                fit_times = [ events_frac['start'].iloc[i] + timedelta(seconds = j*60) for j in np.arange(pred_dur)]
                fit_profile = events_frac['bzm_predicted'].iloc[i] * np.sin(np.pi*np.arange(0,1,1./(pred_dur)) )          
                
                ax1.plot(fit_times, fit_profile, color=fitcolor[events_frac['frac'].iloc[i]])    

    #----theta_z
    ax1b.plot(data['date'], data['theta_z'], label='theta_z')
    ax1b.set_xticklabels(' ')
    ax1b.xaxis.set_major_locator(daysLoc)
    ax1b.xaxis.set_minor_locator(hoursLoc)
    ax1b.set_xlim([st, et])
    for l in line:
        ax1b.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for b in range(len(bars)):
        ax1b.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
    leg = ax1b.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
                              
    #----density
    ax2.plot(data['date'], data['sw_n'], label='n ($\mathrm{cm^-3}$)')
    ax2.set_xticklabels(' ')
    ax2.xaxis.set_major_locator(daysLoc)
    ax2.xaxis.set_minor_locator(hoursLoc)
    ax2.set_xlim([st, et])
    for l in line:
        ax2.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for b in range(len(bars)):
        ax2.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
    leg = ax2.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #----velocity
    maxv = max(  data['sw_v'].loc[np.where(np.isnan(data['sw_v']) == False )] ) + 50
    minv =  min(  data['sw_v'].loc[np.where(np.isnan(data['sw_v']) == False )] ) - 50
    ax3.plot(data['date'], data['sw_v'], label='v ($\mathrm{km s^-1}$)')
    ax3.set_ylim(top = maxv, bottom = minv)
    ax3.set_xticklabels(' ')
    ax3.xaxis.set_major_locator(daysLoc)
    ax3.xaxis.set_minor_locator(hoursLoc)
    ax3.set_xlim([st, et])
    for l in line:
        ax3.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for b in range(len(bars)):
        ax3.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15)       
    leg = ax3.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #----predicted and actual duration
    ax4.plot(data['date'], data['tau_predicted'], label='$\mathrm{\tau predicted (hr)}$', ls='solid',c='b')
    ax4.plot(data['date'], data['tau_actual'], label='$\mathrm{\tau actual (hr)}$', ls='dotted', c='r')
    ax4.set_xticklabels(' ')
    ax4.xaxis.set_major_locator(daysLoc)
    ax4.xaxis.set_minor_locator(hoursLoc)
    ax4.set_xlim([st, et])
    for l in line:
        ax4.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for b in range(len(bars)):
        ax4.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
    leg = ax4.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
        
    #----Bz max predicted and actual
    ax5.plot(data['date'], data['bzm_predicted'], label='Bzm predict (nT)', ls='solid', c='b')
    ax5.plot(data['date'], data['bzm_actual'], label='Bzm actual (nT)', ls='dotted', c='r')
    #ax3.hlines(0.0, data['date'][0], data['date'][-1], linestyle='--',color='grey')
    ax5.set_xticklabels(' ')
    ax5.xaxis.set_major_locator(daysLoc)
    ax5.xaxis.set_minor_locator(hoursLoc)
    ax5.set_xlim([st, et])
    for l in line:
        ax5.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for b in range(len(bars)):
        ax5.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
    leg = ax5.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
       
    #----dst
    ax6.plot(dst_data[st:et].index, dst_data[st:et]['dst'], label='Dst')
    ax6.hlines(dst_thresh, data['date'][0], data['date'].iloc[-1], linestyle='--',color='grey')
    ax6.set_xticklabels(' ')
    ax6.xaxis.set_major_formatter(dateFmt)
    ax6.xaxis.set_major_locator(daysLoc)
    ax6.xaxis.set_minor_locator(hoursLoc)
    ax6.set_xlim([st, et])
    for l in line:
        ax6.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for b in range(len(bars)):
        ax6.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
    ax6.set_xlabel("Start Time "+ str(st)+" (UTC)")
    leg = ax6.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    
    #plt.show()

    plt.savefig(plt_outpath + plt_outfile, format='pdf')

    plt.close()          
          
    return None

def dst_geo_tag(events, dst_data, dst_thresh = -80, dst_dur_thresh = 2.0):
    
    #add min Dst value and geoeffective tag for each event
    dstmin = pd.DataFrame({'dst':[]})
    dstdur = pd.DataFrame({'dstdur':[]})
    geoeff = pd.DataFrame({'geoeff':[]})
    for j in range(len(events)):
                
        #dst values for event time period
        dst_evt = dst_data[events['start'].iloc[j] : events['end'].iloc[j]]

        #if there are no dst data values then quit and move onto the next event interval
        if len(dst_evt) == len(dst_evt.iloc[np.where(dst_evt['dst'] == False)]):
            geoeff.loc[j] = 2           #unknown geoeff tag  
            dstdur.loc[j] = 0.0
            continue

        # the min dst value regardless of duration
        dstmin.loc[j] = dst_evt['dst'].min()
               
        #determine periods where dst is continuously below threshold dst < -80
        dst_evt['tag'] = dst_evt['dst'] < dst_thresh

        fst = dst_evt.index[dst_evt['tag'] & ~ dst_evt['tag'].shift(1).fillna(False)]
        lst = dst_evt.index[dst_evt['tag'] & ~ dst_evt['tag'].shift(-1).fillna(False)]
        pr = np.asarray([[i, j] for i, j in zip(fst, lst) if j > i])
        
        #if the event never reaches dst < -80 then it's not geoeffective
        if len(pr) == 0:
            geoeff.loc[j] = 0  
            dstdur.loc[j] = 0.0
        else:                               #at some point during event, dst < -80
            for t in pr:
                time_below_thresh = (t[1] - t[0] + timedelta(seconds = 3600)).seconds/60./60.
                        
                #event is considered geoeffictive if dst < -80 for more than 2 hours 
                if time_below_thresh >= dst_dur_thresh:
                    
                    #now question if the dst is just recovering from previous event being geoeffective
                    if (dst_evt['dst'].iloc[-1] > dst_evt['dst'].iloc[0] + 2):

                        # if there the previous event interval also decreases then it could be recovering from that
                        #if j > 0 & geoeff.loc[j-1] == 1:
                        geoeff.loc[j] = 3                       #dst still rising from previous event -> ambiguous
                    else:
                        geoeff.loc[j] = 1

                    dstdur.loc[j] = time_below_thresh

                else: 
                    geoeff.loc[j] = 0       # not below dst threshhold for long enough -> it's not geoeffective
                    dstdur.loc[j] = time_below_thresh

    events = events.reset_index()
    events = pd.concat([events, dstmin, dstdur, geoeff], axis = 1)  
    
    return events
        
    
    
    
#==============================================================================
# def find_event_times(data):
#     
#     """
#     Find the start and end time of the events
#     """
# 
#     import copy
#     
#     w = copy.deepcopy(data['bzm_predicted'])
#     w.loc[np.where(data['bzm_predicted'] !=0.)] = 1
#     w = w - np.roll(w,1)
#           
#     st = np.where(w == 1)[0]
#     et = np.where(w == -1)[0]-1
# 
#     evt_indices = np.transpose([st,et])
# 
#     evt_times = np.transpose(np.array([data['date'][st], data['date'][et]]))
#    
#     return evt_times, evt_indices
#==============================================================================
    
def event_end(data, i):
    
    """ 
    An event is defined as the times bounded by B_z passing through 0.
    
    This function determines the end of an event by checking whether Bz
    is changing sign.
    
    Parameters
    ----------
    data : pandas dataframe, required
        Mag and sw data 
    i: int, required
         current array index 
        
    Returns
    -------
    event_end_yes: int
        flag to signal event end
    
    """
    
    if data['bz'][i] * data['bz'][i-1] >= 0:
        event_end_yes = 0
    
    if data['bz'][i] * data['bz'][i-1] < 0:
        event_end_yes = 1
        
    if i == len(data['date'])-1:
        event_end_yes = 1
        
    return event_end_yes
    
    
    
def long_duration(istart, iend, min_duration):

    """
    Check to see if the event last at least min_duration
    
    Parameters
    ----------
    istart, iend : int, required 
        event start and end indexes
    min_duration: int, required 
        min_duration required for event
        
    Returns
    -------
    long_duration: int
        flag indicating a long_duration event
    
    """  
    
    if (iend - istart) <= min_duration:
        long_duration_yes = 0
        
    if (iend - istart) > min_duration:
        long_duration_yes = 1
        
    return long_duration_yes
    
    

def icme_event(istart, iend, npts):
    
    """
    Parameters
    ----------
    istart, iend : int, required 
        event start and end indices
    npts: int, required
        length of data array
        
    Returns
    -------
    main_event: int
        flag indicating XXX
    """
    
    if (istart < npts/2.) & (iend > npts/2.):
        main_event = 1
    else:
        main_event = 0
        
    return main_event

    

def predict_duration(data, istart, iend, pdf):
    
    """
    The original version of this code is from Jim Chen and Nick Arge
    and is called predall.pro. This version has been modifed to 
    make the code more readable by removing goto statements
    and 5 minute data averages.
    
    Reference Papers: Chen et al. 1996, 1997, 2012
                      Arge et al. 2002
    
    Parameters
    ----------
    istart, iend : int, required 
        event start and end indices

    Returns
    -------
    None
    
    """   

    #Extract data from structure needed for prediction routine
    bz = data['bz']             #in nT
    theta = data['theta_z']     #in degrees

    theta_start = theta[istart]
    theta_prev_step = theta_start
    
    tot_dtheta = 0

    #print, istart, bz_start, theta_start
    #print, iend, bz_end, theta_end
    #j=0.
    #rate_of_rotation_decreasing = fltarr((iend-istart)/20., /NOZERO)

    step = 20
    
    for i in np.arange(istart+step, iend, step):
        
        bz_current = bz[i]
        theta_current = theta[i]

        #max bz and theta up until current time
        bz_max = np.max(abs(bz[istart:i]))
        index_bz_max = np.where(abs(bz[istart:i]) == bz_max) 
        bz_max = bz[istart + index_bz_max[0]].values      #to account for sign of Bz

        theta_max = np.max(abs(theta[istart:i]))
        index_theta_max = np.where(abs(theta[istart:i]) == theta_max)

        i_bzmax = istart + index_bz_max[0]
        i_thetamax = istart + index_theta_max[0]
        
        #calculate the rate of duration and the predicted duration of the event
        dtheta = theta_max - theta_start
        dduration = i_thetamax - istart
        rate_of_rotation = dtheta/dduration      #in degrees/minutes
        
        predicted_duration = abs(180./rate_of_rotation)/60.           #in hours
                
        #current_dtheta = theta_current - theta_prev_step
        #current_duration = step
        #current_rate_of_rotation = current_dtheta/current_duration
        
        #predicted_duration = abs((180.-abs(theta_current - theta_start))/current_rate_of_rotation)/60.  + ((i - istart)/60.)     #in hours

        #theta_prev_step = theta_current

        #average_decreasing_rotation = rate_of_rotation

        #print('rate of rotation: ' + str(rate_of_rotation))
        #print('predicted duration: ' + str(predicted_duration))
        #print('\n')
        
        if value_increasing(bz_current, bz_max):
            
            form_function = np.sin(np.pi*((i_bzmax - istart)/60.)/predicted_duration) #Sin function in radians
                
            predicted_bzmax = bz_max/form_function
            #predicted_bzmax = predicted_bzmax[0][0]
            
            #if (form_function < 0) & (i >= step-1):
            #    predicted_bzmax = data['bzm_predicted'][i-step-1]
        else:
            predicted_bzmax = bz_max
                           
        if np.abs(predicted_bzmax) > 30.:
            predicted_bzmax = bz_max 
        
        data.loc[i-step:i, 'istart'] = istart
        data.loc[i-step:i, 'iend'] = iend   
            
        data.loc[i-step:i, 'tau_predicted'] = predicted_duration    #[0][0]
        data.loc[i-step:i, 'tau_actual'] = (iend-istart)/60.
        data.loc[i-step:i, 'bzm_predicted'] = predicted_bzmax

        #index of max bz up to the current time - used for fitting bz profile
        data.loc[i-step:i, 'i_bzmax'] = i_bzmax

        #max value of Bz with sign
        bz_max_val = np.max(abs(bz[istart:iend]))
        index_bz_max_val = np.where(abs(bz[istart:iend]) == bz_max_val)
        data.loc[i-step:i, 'bzm_actual'] = bz.loc[istart + index_bz_max_val[0]].values         

        #Using Bayesian statistics laid out in Chen papers, determine the probability 
        #of a geoeffective event given the estimated Bzm and tau
        bzmp_ind = np.max(np.where(pdf['axis_vals'][2::] < predicted_bzmax)[1])
        taup_ind = np.min(np.where(pdf['axis_vals'][3::] > predicted_duration)[1]) 
        
        P1 = np.sum(pdf['pdf'][:,:,bzmp_ind, taup_ind])

    
        #fill in rest of data record for remaining portion if what is left is less
        #than one step size
        if (iend-i) < step:
            data.loc[i:iend, 'istart'] = istart
            data.loc[i:iend, 'iend'] = iend         
            data.loc[i:iend, 'tau_predicted'] = predicted_duration    #[0][0]
            data.loc[i:iend, 'tau_actual'] = (iend-istart)/60.
            data.loc[i:iend, 'bzm_predicted'] = predicted_bzmax
            data.loc[i:iend, 'bzm_actual'] = bz.loc[istart + index_bz_max_val[0]].values  

        #if (i_thetamax > i-step): data['duration_actual'] = 0.
        #;if (i_bzmax > i-step): data['bzm_actual'] = 0.
    

def value_increasing(value_current, value_max):
    """
    function determines if the input value is 
    increasing or not
    
    Parameters
    ----------
    value_current, value_max : float, required 
        current and maximum value

    Returns
    -------
    value_increasing: int
        flag determining whether value is increasing
    
    """ 
    
    
    if abs(value_current) < 0.8*abs(value_max):
        value_increasing = 0 
      
    if abs(value_current) > 0.8*abs(value_max): 
        value_increasing = 1
      
    return value_increasing
    
    
    
def validation_stats(data, istart, iend, outdir=''):
    
    duration = (iend-istart)
    
    #compute the unsigned fractional diviation between the predicted and observed Bzm and tau
    fraction_of_event = np.arange(9)/10.
    index_of_event = np.floor((duration*fraction_of_event)+istart)
    print, istart, iend
    print, index_of_event
    
    Bzm_fractional_deviation = abs((data['bzm_predicted'][index_of_event] -data['bzm_actual'][istart]) \
                                  /data['bzm_actual'][istart])
    Tau_fractional_deviation = abs(data['tau_predicted'][index_of_event] - data['tau_actual'][istart]) \
                                  /data['tau_actual'][istart]

    start_date = datetime.strftime(data['date'][istart], "%Y-%m-%d %H:%M")
    
    print, start_date
    print, fraction_of_event
    print, Bzm_fractional_deviation
    print, Tau_fractional_deviation
    
    #fname1='Prediction_Results_Duration.txt'
    #OPENW, dunit, outdir + fname1, /GET_LUN, /APPEND
    #PRINTF, dunit, format='(a-25,10(2x,f5.2))', start_date, Tau_fractional_deviation
    #FREE_LUN, dunit
    
    #fname2='Prediction_Results_BzMax.txt'
    #OPENW, bunit, outdir + fname2, /GET_LUN, /APPEND
    #PRINTF, bunit, format='(a-25,10(2x,f5.2))', start_date, Bzm_fractional_deviation
    #FREE_LUN, bunit
    
    
def clean_data(mag, sw):
    
    """
    Clean solar wind plasma and mag data to remove bad data values and fix 
    data gaps (Michele Cash version translated to python)
    
    Parameters
    ----------
    mag : data array, required 
    sw: data, required

    Returns
    -------
    None
    
    """
    
    nevents_sw = len(sw['v'])
    nevents_mag = len(mag['gsm_bz'])
    
    
    #---check them magnetic field data
    bad_mag = np.where((abs(mag['gsm_bx']) > 90.) & (abs(mag['gsm_by']) > 90.) & (abs(mag['gsm_bz']) > 90.))
    nbad_mag = len(bad_mag[0])   
    
    #no good mag data
    if nevents_mag - nbad_mag < 2:
        print("******* No valid magnetic field data found *******")
        return None

        
    #if there is some bad data, set to NaN and interpolate
    if nbad_mag > 0:   
        mag['gsm_bx'][bad_mag] = np.nan
        mag['gsm_by'][bad_mag] = np.nan
        mag['gsm_bz'][bad_mag] = np.nan
        mag['bt'][bad_mag]     = np.na
        mag['gsm_lat'][bad_mag] = np.nan
        mag['gsm_lon'][bad_mag] = np.nan

    mag['gsm_bx'] = mag['gsm_bx'].interpolate()
    mag['gsm_by'] = mag['gsm_by'].interpolate()
    mag['gsm_bz'] = mag['gsm_bz'].interpolate()
    mag['bt']     = mag['bt'].interpolate()
    mag['gsm_lat'] = mag['gsm_lat'].interpolate()
    mag['gsm_lon'] = mag['gsm_lon'].interpolate()
    
    
    #---check solar wind velocity
    badsw_v = np.where((sw['v'] < 0.) & (sw['v'] > 3000.))
    nbadsw_v = len(badsw_v[0])
    
    #no valid sw data
    if nevents_sw - nbadsw_v < 2:
        print("******* No valid solar wind plasma data found *******")
        return None

    #if there are some bad sw velcotiy data values, set to NaN and interpolate
    if nbadsw_v > 0:
        print('******* Some bad SWE velocity data *******')
        sw['v'][badsw_v[0]] = np.nan
        
    sw['v'] = sw['v'].interpolate()

    #---check solar wind density which can be good even where the velocity was good
    badsw_n = np.where((sw['n'] < 0.) & (sw['n'] > 300.))
    nbadsw_n = len(badsw_n[0])
    
    if nbadsw_n > 0:
        print('******* Some bad SWE density data *******')
        
        #if there are no good density values, set all density to 4.0
        if nevents_sw - nbadsw_n == 0:
            sw['n'][:] = 4.0
        else:
            sw['n'][badsw_n[0]] = np.nan
    
    sw['n'] = sw['n'].interpolate()

            
    #---check solar wind temperature which can be good even where the velocity was good
    badsw_t = np.where(sw['t'] < 0.) 
    nbadsw_t = len(badsw_t[0])
    
    if nbadsw_t > 0:
        print('******* Some bad SWE temperature data *******')
        
        #if there are no good density values, set all temperature to 0.0
        if nevents_sw - nbadsw_t == 0:
            sw['t'][:] = 0.0
        else:
            sw['t'][badsw_n[0]] = np.nan

    sw['t'] = sw['t'].interpolate()
        
            

    #---interpolate the solar wind velocity to the mag time
    #SWVel=INTERPOL(swe.Speed,swe.jdate,mag.jdate)
    #Np=INTERPOL(swe.Np,swe.jdate,mag.jdate)
    
    #return SWVel, Np  -----QUERY????
    
    return mag, sw
    

    
    
    
    
      
    


        
    