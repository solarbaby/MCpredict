# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:30:38 2017

@author: hazel.bain

Routines for reading in solar wind data from the database or csv file 

"""

import numpy as np
from datetime import datetime
import numpy.lib.recfunctions

def get_data(tstart, tend, server = 'swds-st', \
             database = 'RA', view = 'ace_mag_1m',\
             csv = 1, outpath = 'C:/Users/hazel.bain/data/'):
      
    """
    Checks to see if the data is already stored locally, if not
    it calls read_database to retrieve the data
    
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
        output csv file keyword. default is 1
    outpath: string, optional
        csv file path

    Returns
    -------
    None
    
    """

    from datetime import timedelta
    import os
    
    #convert times to datetime
    st = datetime(int(tstart.split('-')[0]), int(tstart.split('-')[1]), int(tstart.split('-')[2]))
    et = datetime(int(tend.split('-')[0]), int(tend.split('-')[1]), int(tend.split('-')[2]))
    
    #datafile dates to check for
    dates_to_check = [st + timedelta(int(i)) for i in np.arange(0,(et-st).days+1)]
    for d in dates_to_check:
        
        #if there is no datafile for that day, get the data from the databacse
        if not os.path.isfile(outpath + '/' + view + '/' + \
                              view + '_' + datetime.strftime(d, '%Y%m%d') + '.csv'):
            t = datetime.strftime(d, "%Y-%m-%d")
            
            print("datafile missing for day " + t + ": accessing databasee")
            data = read_database(t, t, server = server, \
                                 database = database, view=view, \
                                 csv=csv, outpath=outpath+ '/' + view + '/')
            
    #once all data is stored locally, read from csv file            
    print("All files stored locally: reading from csv")
            
    if view == 'ace_mag_1m':
        for f in dates_to_check:
            file = view + '_' + datetime.strftime(f, "%Y%m%d") + '.csv'
            if f == dates_to_check[0]:
                data = read_ace_mag_1m_database_csv(file)
            else:
                data_tmp = read_ace_mag_1m_database_csv(file)
                data = np.hstack((data,data_tmp))
                    
    elif view == 'ace_swepam_1m':
        for f in dates_to_check:
            file = view + '_' + datetime.strftime(f, "%Y%m%d") + '.csv'
            if f == dates_to_check[0]:
                data = read_ace_swepam_1m_database_csv(file)
            else:
                data_tmp = read_ace_swepam_1m_database_csv(file)
                data = np.hstack((data,data_tmp))
        
    
    return data

def read_database(tstart, tend, server = 'swds-st', \
                                 database = 'RA', view = 'ace_mag_1m',\
                                 csv = 1, outpath = 'C:/Users/hazel.bain/data/'):    

    """
    Pulls data from database
    
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
        output csv file keyword. default is 1
    outpath: string, optional
        csv file path

    Returns
    -------
    data : ndarray
        Array of data from database
    
    """
    
    import pyodbc
    
    #connect to server 
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER='+server+\
                      '; DATABASE='+database+'; Trusted_Connection=yes;')
    cursor = cnxn.cursor()
   
    #execute database call
    cursor.execute("SELECT * FROM [" + database + "].[dbo].[" + view + "] \
               where time_tag > '"+ tstart +"' and time_tag <= '"+ tend +" 23:59';")

    #get column names
    col_name = [column[0] for column in cursor.description]
    #col_fmt = (datetime, '<i4', '|S10', '|S10', '|S10', '|S10', '|S10', '|S10')
    col_fmt = [column[1] for column in cursor.description]
    
    #return all rows from search
    all_rows = cursor.fetchall()
    
    #close connection to the database    
    cnxn.close
    
    #convert to array and change None values 
    indata = np.asarray(all_rows)
    
    #for float formats change None to nan
    flt_indx = np.where(np.asarray(col_fmt).astype(str) == "<class 'float'>")
    for i in flt_indx[0]:
        indata[:,i][np.where(indata[:,i] == np.array(None))] = np.nan

    #for int formats change None to 99999
    int_indx = np.where(np.asarray(col_fmt).astype(str) == "<class 'int'>")
    for i in int_indx[0]:
        indata[:,i][np.where(indata[:,i] == np.array(None))] = 99999
    
    #store in data structure
    indata2 = list(map(tuple, indata))        #has to be list of tuples
    data = np.array(indata2, dtype = {'names': col_name,'formats': col_fmt})
    
    
    #record as daily csv files
    if csv == 1:
        database2csv(data, view, outpath)
    
    return data
    
 
def database2csv(data, file, path, days = 1):    

    """
    Write database data to csv file
    
    Parameters
    ----------
    data : structured data array, required 
    file: string, required
        filename for csv files
    path: string, required 
        Path to csv output directory
    days: int, optional
        keyword to save file in daily subgroups, default yes
        
        
    Returns
    -------
    None
    

    """
    
    import csv 

    #split the data into daily subgroups
    day = [d.day for d in data['time_tag']]
    day_idx = np.where(np.roll(day,1) != day)
    
    #if there are mulptiple days data in array and want to record serpartely
    if days == 1 and len(day_idx[0]) > 0:
        
        #multiple days in record
        for i in np.arange(len(day_idx[0])):
                          
            fname = file + '_' + datetime.strftime(data['time_tag'][day_idx[0][i]], '%Y%m%d') \
                + '.csv'
                
            #write daily csv file
            with open(path + fname, "w", newline='') as csv_file:
                csv_file.write(', '.join(data.dtype.names)+'\n')
                writer = csv.writer(csv_file, delimiter=',')
                
                if i == len(day_idx[0])-1: 
                    for line in data[day_idx[0][i]::]:
                        writer.writerow(line)
                else:
                    for line in data[day_idx[0][i]:day_idx[0][i+1]-1]:
                        writer.writerow(line)
            csv_file.close()
         
    #if not required to seprate data into daily subgroups or only 
    #one days worth of data in array dump all data into the one csv file            
    else:
        
        #generate file name on view and date
        fname = file + '_' + datetime.strftime(data['time_tag'][0], '%Y%m%d')\
            + '.csv'
        
        #write data to csv file
        with open(path + fname, "w", newline='') as csv_file:
            csv_file.write(', '.join(data.dtype.names)+'\n')
            writer = csv.writer(csv_file, delimiter=',')
            for line in data:
                writer.writerow(line)
        csv_file.close()

        print("saved data to csv file")
        
    
    return None
    
    

def read_ace_mag_1m_database_csv(file, path = 'C:/Users/hazel.bain/data/ace_mag_1m/'):    

    
    """
    Reads in ace_mag_1m data from csv files saved from RA database
    
    Parameters
    ----------
    file : string, required 
        CSV file containing the ace_mag_1m data to read.
    dir: string, optional 
        path to csv file. Default is MCpredict folder


    Returns
    -------
    data : ndarray
        Array of data from database
    
    """
    
    #file = 'ace_mag_test.csv'

    #Richardson and Cane spreedsheet column names and format
    col_name =  ('time_tag', 'dsflag', 'gsm_bx', \
                 'gsm_by', 'gsm_bz', 'bt', 'gsm_lat', 'gsm_lon')
    col_fmt = ('S30', '<i4', '|S10', '|S10', '|S10', '|S10', '|S10', '|S10')

    #read the file
    indata = np.loadtxt(path + file, dtype = {'names': col_name, \
        'formats': col_fmt},delimiter = ',', skiprows=1)
   
    #change Null data to None
    data = np.copy(indata)
    
    data['gsm_bx'][np.where(data['gsm_bx'] == b'NULL')] = np.nan
    data['gsm_by'][np.where(data['gsm_by'] == b'NULL')] = np.nan
    data['gsm_bz'][np.where(data['gsm_bz'] == b'NULL')] = np.nan
    data['bt'][np.where(data['bt'] == b'NULL')] = np.nan
    data['gsm_lat'][np.where(data['gsm_lat'] == b'NULL')] = np.nan
    data['gsm_lon'][np.where(data['gsm_lon'] == b'NULL')] = np.nan

    #convert values to float   
    col_fmt_new = ('S30', 'i4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4')
    data = data.astype({'names':col_name, 'formats':col_fmt_new})
    
    #append a new column which converts time tag to datetime    
    date_temp = np.asarray([ datetime(int(x.decode("utf-8").split('-')[0]), \
        int(x.decode("utf-8").split('-')[1]),\
        int(x.decode("utf-8").split('-')[2].split(' ')[0]), \
        int(x.decode("utf-8").split(' ')[1].split(':')[0]), \
        int(x.decode("utf-8").split(' ')[1].split(':')[1])  \
        ) 
        for x in data['time_tag']])
    
    data = numpy.lib.recfunctions.append_fields(data, 'date', date_temp, dtypes='datetime64[us]', usemask=False, asrecarray=True)

    return data
    
def read_ace_swepam_1m_database_csv(file, path = 'C:/Users/hazel.bain/data/ace_swepam_1m/'):    

    
    """
    Reads in ace_swepam_1m data from csv files saved from RA database
    
    Parameters
    ----------
    file : string, required 
        CSV file containing the ace_swepam_1m data to read.
    path: string, optional 
        path to csv file. Default is MCpredict folder


    Returns
    -------
    data : ndarray
        Array of data from database
    
    """
    
    #file = 'swepam_test.csv'

    #Richardson and Cane spreedsheet column names and format
    col_name =  ('time_tag', 'dsflag', 'n', 'v', 't')
    col_fmt = ('S30', '|S10', '|S10', '|S10', '|S10')

    #read the file
    indata = np.loadtxt(path + file, dtype = {'names': col_name, \
        'formats': col_fmt},delimiter = ',', skiprows=1)
   
    #change Null data to None
    data = np.copy(indata)
    
    data['dsflag'][np.where(data['dsflag'] == b'NULL')] = 999
    data['n'][np.where(data['n'] == b'NULL')] = np.nan
    data['v'][np.where(data['v'] == b'NULL')] = np.nan
    data['t'][np.where(data['t'] == b'NULL')] = np.nan

    #convert values to float   
    col_fmt_new = ('S30', 'i4', 'f4', 'f4', 'f4')
    data = data.astype({'names':col_name, 'formats':col_fmt_new})
    
    #append a new column which converts time tag to datetime    
    date_temp = np.asarray([ datetime(int(x.decode("utf-8").split('-')[0]), \
        int(x.decode("utf-8").split('-')[1]),\
        int(x.decode("utf-8").split('-')[2].split(' ')[0]), \
        int(x.decode("utf-8").split(' ')[1].split(':')[0]), \
        int(x.decode("utf-8").split(' ')[1].split(':')[1])  \
        ) 
        for x in data['time_tag']])
    
    data = numpy.lib.recfunctions.append_fields(data, 'date', date_temp, dtypes='datetime64[us]', usemask=False, asrecarray=True)

    return data    