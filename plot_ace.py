# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:38:48 2017

@author: hazel.bain

Plotting routines for ACE_MAG_1m and ACE_SWEPAM_1M data

"""

from read_database import get_data
import read_dst as dst

from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.dates import DayLocator
from matplotlib.dates import HourLocator
from matplotlib.dates import DateFormatter

def plot_ace_dst(tstart, tend, server = 'swds-st', \
             database = 'RA', view = 'ace_mag_1m',\
             csv = 1, outpath = 'C:/Users/hazel.bain/data/', \
             line = [], plt_outpath = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/richardson_mc_plots/',\
             plt_outfile = 'ace_dst.pdf'):
    
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
    
    #read in the ace data  
    ace_mag_data = get_data(tstart, tend, view = 'ace_mag_1m')
    ace_swepam_data = get_data(tstart, tend, view = 'ace_swepam_1m')
    
    #read in the dst data
    dst_data = dst.read_dst(tstart, tend)
        
    #start and end times for plot to make sure all plots are consistent
    st = datetime.strptime(tstart, "%Y-%m-%d")
    et = datetime.strptime(tend, "%Y-%m-%d")

    #plot the ace data
    f, (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(8, figsize=(11,9))
 
    plt.subplots_adjust(hspace = .1)       # no vertical space between subplots
    fontP = FontProperties()                #legend
    fontP.set_size('medium')
    
    dateFmt = DateFormatter('%d-%b')
    hoursLoc = HourLocator()
    daysLoc = DayLocator()
    
    #----Bx
    ax0.plot(ace_mag_data['date'], ace_mag_data['gsm_bx'], label='Bx (nT)')
    ax0.hlines(0.0, ace_mag_data['date'][0], ace_mag_data['date'][-1], linestyle='--',color='grey')
    ax0.set_title('ACE MAG + SWEPAM 1m', loc='Right')
    ax0.set_xticklabels(' ')
    ax0.xaxis.set_major_locator(daysLoc)
    ax0.xaxis.set_minor_locator(hoursLoc)
    ax0.set_xlim([st, et])
    for l in line:
        ax0.axvline(x=l, linewidth=2, linestyle='--', color='black')
    leg = ax0.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #----By
    ax1.plot(ace_mag_data['date'], ace_mag_data['gsm_by'], label='By (nT)')
    ax1.hlines(0.0, ace_mag_data['date'][0], ace_mag_data['date'][-1], linestyle='--',color='grey')
    ax1.set_xticklabels(' ')
    ax1.xaxis.set_major_locator(daysLoc)
    ax1.xaxis.set_minor_locator(hoursLoc)
    ax1.set_xlim([st, et])
    for l in line:
        ax1.axvline(x=l, linewidth=2, linestyle='--', color='black')
    leg = ax1.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #----Bz
    ax2.plot(ace_mag_data['date'], ace_mag_data['gsm_bz'], label='Bz (nT)')
    ax2.hlines(0.0, ace_mag_data['date'][0], ace_mag_data['date'][-1], linestyle='--',color='grey')
    ax2.set_xticklabels(' ')
    ax2.xaxis.set_major_locator(daysLoc)
    ax2.xaxis.set_minor_locator(hoursLoc)
    ax2.set_xlim([st, et])
    for l in line:
        ax2.axvline(x=l, linewidth=2, linestyle='--', color='black')
    leg = ax2.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #----|B|
    ax3.plot(ace_mag_data['date'], ace_mag_data['bt'], label='|B| (nT)')
    ax3.set_xticklabels(' ')
    ax3.xaxis.set_major_locator(daysLoc)
    ax3.xaxis.set_minor_locator(hoursLoc)
    ax3.set_xlim([st, et])
    for l in line:
        ax3.axvline(x=l, linewidth=2, linestyle='--', color='black')
    leg = ax3.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #----density
    ax4.plot(ace_swepam_data['date'], ace_swepam_data['n'], label='n ($\mathrm{cm^-3}$)')
    ax4.set_xticklabels(' ')
    ax4.xaxis.set_major_locator(daysLoc)
    ax4.xaxis.set_minor_locator(hoursLoc)
    ax4.set_xlim([st, et])
    for l in line:
        ax4.axvline(x=l, linewidth=2, linestyle='--', color='black')
    leg = ax4.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #----velocity
    ax5.plot(ace_swepam_data['date'], ace_swepam_data['v'], label='v ($\mathrm{km s^-1}$)')
    ax5.set_ylim(top = max(ace_swepam_data['v'])+50, bottom = min(ace_swepam_data['v'])-50)
    ax5.set_xticklabels(' ')
    ax5.xaxis.set_major_locator(daysLoc)
    ax5.xaxis.set_minor_locator(hoursLoc)
    ax5.set_xlim([st, et])
    for l in line:
        ax5.axvline(x=l, linewidth=2, linestyle='--', color='black')
    leg = ax5.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #----temp
    ax6.plot(ace_swepam_data['date'], ace_swepam_data['t'], label='t')
    ax6.set_xticklabels(' ')
    ax6.xaxis.set_major_locator(daysLoc)
    ax6.xaxis.set_minor_locator(hoursLoc)
    ax6.set_xlim([st, et])
    for l in line:
        ax6.axvline(x=l, linewidth=2, linestyle='--', color='black')
    leg = ax6.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #----dst
    ax7.plot(dst_data['datetime'], dst_data['dst'], label='Dst')
    ax7.set_xticklabels(' ')
    ax7.xaxis.set_major_formatter(dateFmt)
    ax7.xaxis.set_major_locator(daysLoc)
    ax7.xaxis.set_minor_locator(hoursLoc)
    ax7.set_xlim([st, et])
    for l in line:
        ax7.axvline(x=l, linewidth=2, linestyle='--', color='black')
    ax7.set_xlabel("Start Time "+datetime.strftime(dst_data['datetime'][0].astype(datetime),\
            "%Y-%d-%m %H:%M")+" (UTC)")
    leg = ax7.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    
    #plt.show()

    plt.savefig(plt_outpath + plt_outfile, format='pdf')



def plot_ace_mag_1m(tstart, tend, server = 'swds-st', \
        database = 'RA', view = 'ace_mag_1m',\
        csv = 1, outpath = 'C:/Users/hazel.bain/data/',\
        line = []):
    
    """
    Plot the ACE_MAG_1m data
    
    """
    
    #read in the ace data
    ace_mag_data = get_data(tstart, tend, server = 'swds-st', \
             database = 'RA', view = 'ace_mag_1m',\
             csv = 1, outpath = 'C:/Users/hazel.bain/data/')
    
    #plot the ace data
    f, (ax0, ax1, ax2, ax3) = plt.subplots(4, figsize=(10,8))
 
    plt.subplots_adjust(hspace = .05)       # no vertical space between subplots
    fontP = FontProperties()                #legend
    fontP.set_size('medium')
    
    dateFmt = DateFormatter('%H:%M')
    hoursLoc = HourLocator()
    daysLoc = DayLocator()
    
    #----Bx
    ax0.plot(ace_mag_data['date'], ace_mag_data['gsm_bx'], label='Bx (nT)')
    ax0.hlines(0.0, ace_mag_data['date'][0], ace_mag_data['date'][-1], linestyle='--',color='grey')
    ax0.set_title('ACE MAG 1m', loc='Right')
    ax0.set_xticklabels(' ')
    ax0.xaxis.set_major_locator(daysLoc)
    ax0.xaxis.set_minor_locator(hoursLoc)
    for l in line:
        ax0.axvline(x=l, linewidth=2, linestyle='--', color='black')
    leg = ax0.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #----By
    ax1.plot(ace_mag_data['date'], ace_mag_data['gsm_by'], label='By (nT)')
    ax1.hlines(0.0, ace_mag_data['date'][0], ace_mag_data['date'][-1], linestyle='--',color='grey')
    ax1.set_xticklabels(' ')
    ax1.xaxis.set_major_locator(daysLoc)
    ax1.xaxis.set_minor_locator(hoursLoc)
    for l in line:
        ax1.axvline(x=l, linewidth=2, linestyle='--', color='black')
    leg = ax1.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #----Bz
    ax2.plot(ace_mag_data['date'], ace_mag_data['gsm_bz'], label='Bz (nT)')
    ax2.hlines(0.0, ace_mag_data['date'][0], ace_mag_data['date'][-1], linestyle='--',color='grey')
    ax2.set_xticklabels(' ')
    ax2.xaxis.set_major_locator(daysLoc)
    ax2.xaxis.set_minor_locator(hoursLoc)
    for l in line:
        ax2.axvline(x=l, linewidth=2, linestyle='--', color='black')
    leg = ax2.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #----|B|
    ax3.plot(ace_mag_data['date'], ace_mag_data['bt'], label='|B| (nT)')
    ax3.xaxis.set_major_formatter(dateFmt)
    ax3.xaxis.set_major_locator(daysLoc)
    ax3.xaxis.set_minor_locator(hoursLoc)
    ax3.set_xlabel("Start Time "+datetime.strftime(ace_mag_data['date'][0].astype(datetime),\
            "%Y-%d-%m %H:%M")+" (UTC)")
    for l in line:
        ax3.axvline(x=l, linewidth=2, linestyle='--', color='black')
    leg = ax3.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    plt.show()
    

    
def plot_ace_swepam_1m(tstart, tend, server = 'swds-st', \
        database = 'RA', view = 'ace_mag_1m',\
        csv = 1, outpath = 'C:/Users/hazel.bain/data/',\
        line=[]):
    
    """
    Plot the ACE_SWEPAM_1m data
    
    """
    
    #read in the ace data
    ace_swepam_data = get_data(tstart, tend, server = 'swds-st', \
        database = 'RA', view = 'ace_mag_1m',\
        csv = 1, outpath = 'C:/Users/hazel.bain/data/')
    
    #plot the ace data
    f, (ax0, ax1, ax2) = plt.subplots(3, figsize=(10,6))
 
    plt.subplots_adjust(hspace = .05)       # no vertical space between subplots
    fontP = FontProperties()                #legend
    fontP.set_size('medium')
    
    dateFmt = DateFormatter('%H:%M')
    hoursLoc = HourLocator()
    daysLoc = DayLocator()
    
    #----density
    ax0.plot(ace_swepam_data['date'], ace_swepam_data['n'], label='n ($\mathrm{cm^-3}$)')
    ax0.set_title('ACE SWEPAM 1m', loc='Right')
    ax0.set_xticklabels(' ')
    ax0.xaxis.set_major_locator(daysLoc)
    ax0.xaxis.set_minor_locator(hoursLoc)
    for l in line:
        ax0.axvline(x=l, linewidth=2, linestyle='--', color='black')
    leg = ax0.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #----velocity
    ax1.plot(ace_swepam_data['date'], ace_swepam_data['v'], label='v')
    ax1.set_ylim(top = max(ace_swepam_data['v'])+50, bottom = min(ace_swepam_data['v'])-50)
    ax1.set_xticklabels(' ')
    ax1.xaxis.set_major_locator(daysLoc)
    ax1.xaxis.set_minor_locator(hoursLoc)
    for l in line:
        ax1.axvline(x=l, linewidth=2, linestyle='--', color='black')
    leg = ax1.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #----temp
    ax2.plot(ace_swepam_data['date'], ace_swepam_data['t'], label='t ($\mathrm{km s^-1}$)')
    ax2.set_xticklabels(' ')
    ax2.xaxis.set_major_formatter(dateFmt)
    ax2.xaxis.set_major_locator(daysLoc)
    ax2.xaxis.set_minor_locator(hoursLoc)
    for l in line:
        ax2.axvline(x=l, linewidth=2, linestyle='--', color='black')
    ax2.set_xlabel("Start Time "+datetime.strftime(ace_swepam_data['date'][0].astype(datetime),\
            "%Y-%d-%m %H:%M")+" (UTC)")
    leg = ax2.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    
    plt.show()
    

     