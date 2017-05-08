# -*- coding: utf-8 -*-


import numpy as np
from datetime import datetime
from datetime import timedelta

def read_richardson_icme_list(indir = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/.spyproject/'):
 
    """
    Reads in Richardson and Cane's ICME list
    
    inputs: 
        
    indir: working directory path

    """

    #indir = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/.spyproject/'
    file = 'Richardson_and_Cane_ICME_list.csv'

    #Richardson and Cane spreedsheet column names and format
    col_name =  ('Year', 'Disturbance', 'ICME_plasma_field_start', \
                 'ICME_plasma_field_end', 'Comp_start', 'Comp_end', \
                 'MC_start', 'MC_end', 'BDE', 'BIF', 'Qual', 'dV', 'v_ICME',\
                 'v_max', 'B', 'MC', 'DST', 'v_transit', 'LASCO_CME_time')
    col_fmt = ('i4','S10', 'S10', 'S10', 'S10', 'S10', 'S10', 'S10', 'S10', \
                'S10', 'S10', 'S10', 'S10', 'S10', 'S10', 'S10', 'S10', 'S10', 'S10')

    #col_fmt_converter = {1:lambda x: x.decode("utf-8") }

    #read the spreedsheet
    indata = np.loadtxt(indir + file, dtype = {'names': col_name, \
        'formats': col_fmt},delimiter = ',')
    nevents = len(indata)

    #reformat the times to datetime
    #---disturbance time
    disturbance_time = read_date_format(indata['Year'], indata['Disturbance'])
                 
    #---plasma_time
    plasma_start = read_date_format(indata['Year'], indata['ICME_plasma_field_start'])
    plasma_end   = read_date_format(indata['Year'], indata['ICME_plasma_field_end'])
    plasma_time  = np.column_stack((plasma_start, plasma_end))
    
    #---comp_time -- to do
    
    #---MC_time
    mc_start = []
    mc_end = []
    for i in range(nevents):
        
        #not all ICMEs listed have MC associated (... = no MC)
        if indata['MC_start'][i].decode("utf-8") != '...':
            
            #times are +/- hours relative to the plasma start end times
            mc_start.append(plasma_time[i][0] + timedelta(hours = int(indata['MC_start'][i])))
            mc_end.append(plasma_time[i][1] + timedelta(hours = int(indata['MC_end'][i])))
        
        else:
            
            #no MC with ICME
            mc_start.append(None)
            mc_end.append(None)
            
    mc_start = np.asarray(mc_start)
    mc_end = np.asarray(mc_end)
    
    mc_time  = np.column_stack((mc_start, mc_end))
    
            
            
    #---BDE -- to do
    #---BIF -- to do
    #---Qual -- to do
    #---dv -- to do
    #---v_icme -- to do
    #---v_max -- to do
    #---B -- to do
    
    #---MC flag - 2 = MC, 1 = some rotation but not all characteristics, 0 = no MC
    #remove indexes with H referring to events reported by Huttunen
    h_ind = np.where(indata['MC'] == b'2H')
    MC_flag_temp = indata['MC']
    MC_flag_temp[h_ind] = '2'
        
    #record as int
    MC_flag = MC_flag_temp.astype(int)
    
    
    #---Dst -- to do!!!
    Dst = indata['DST']
    
    
    #---v_transit -- to do
    #---lasco CME time -- to do
    
    outcol_name =  ('year','disturbance_time', 'plasma_start', 'plasma_end', \
                 'mc_start', 'mc_end', 'MC_flag', 'dst')
    outcol_fmt = ('i4', disturbance_time.dtype, plasma_start.dtype, plasma_end.dtype,\
                 mc_start.dtype, mc_end.dtype, 'i4', 'S10')
    
    dataout = np.empty(len(indata['Year']),dtype={'names': outcol_name, \
        'formats': outcol_fmt})
    dataout['year']             = indata['Year']
    dataout['disturbance_time'] = disturbance_time
    dataout['plasma_start']     = plasma_start
    dataout['mc_start']         = mc_start 
    dataout['mc_end']           = mc_end
    dataout['MC_flag']          = MC_flag
    dataout['dst']              = Dst
    
    return dataout


def read_date_format(year, mdt):

    """
    Format datetime array from column formatted in 'mm/dd tttt' style
    
    year = column with year data
    mdt = column with mm/dd tttt formatted data
    """                      

    nrows = len(year)
    
    date_temp = []
    for i in range(nrows):
        
        #convert to datetime
        date_temp.append(datetime(year[i], \
                       int(mdt[i].decode("utf-8").split('/')[0]),\
                       int(mdt[i].decode("utf-8").split('/')[1].split(' ')[0]),\
                       int(mdt[i].decode("utf-8").split(' ')[1][0:2]),\
                       int(mdt[i].decode("utf-8").split(' ')[1][2:4])))
            
    #convert list to numpy array
    date = np.asarray(date_temp)
    
    return date
    
    