# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 13:36:40 2017

@author: hazel.bain

Reads in hourly DST data from txt file to either a pandas dataframe or
a structured numpy data array.

Use read_dst_df as default

"""

import numpy as np
import pandas as pd
from datetime import datetime
import numpy.lib.recfunctions


def read_dst_df(path = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/.spyproject/'):
    """
    Reads in hourly DST data from txt file to pandas dataframe
    
    inputs:
        
    path: string
        path to working directory
    
    """
    
    file = 'Dst_hourly.csv'
    
    col_name =  ('date', 'time', 'doy', 'dst')
    
    parse = lambda x, y: datetime.strptime(x + ' ' + y, '%m/%d/%Y %H:%M')
    df = pd.read_csv(path + file, sep = ',', names = col_name, \
                parse_dates={'date0': [0,1]}, date_parser=parse, \
                skiprows = 1, index_col = 0)  
    
    return df
    
def read_dst(tstart, tend, path = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/.spyproject/'):    

    
    """
    Reads in hourly DST data from txt file
    
    inputs:
        
    path: string
        path to working directory
    
    """
    
    file = 'Dst_hourly.csv'

    #Richardson and Cane spreedsheet column names and format
    col_name =  ('date', 'time', 'doy', 'dst')
    col_fmt = ('|S10', '|S10', 'i4', 'i4')

    #read the file
    indata = np.loadtxt(path + file, dtype = {'names': col_name, \
        'formats': col_fmt},delimiter = ',', skiprows=1)
   
    #append a new column which converts time tag to datetime
    date_temp = np.asarray([ datetime(int(indata['date'][i].decode("utf-8").split('/')[2].split(' ')[0]), \
        int(indata['date'][i].decode("utf-8").split('/')[0]),\
        int(indata['date'][i].decode("utf-8").split('/')[1]), \
        int(indata['time'][i].decode("utf-8").split(':')[0]), \
        int(indata['time'][i].decode("utf-8").split(':')[1])  \
        )
        for i in range(len(indata['date']))])
    
    data = numpy.lib.recfunctions.append_fields(indata, 'datetime', date_temp, dtypes='datetime64[us]', usemask=False, asrecarray=True)

    #limit to tstart to tend
    st = datetime.strptime(tstart, "%Y-%m-%d %H:%M:%S")
    et = datetime.strptime(tend, "%Y-%m-%d %H:%M:%S")
    
    stidx = np.where(data['datetime'] >= st)[0][0] - 1
    etidx = np.where(data['datetime'] <= et)[0][-1] + 2
    
    outdata = data[stidx:etidx]
    
    
    return outdata