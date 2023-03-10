# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 08:51:40 2023

@author: mazzarettoom
"""
#------------------------------------------------------------------------------
# Estas son las funciones para el curso en Panama, del IH cantabria.
# Desarroladores: Ruiz Alberti Jesus, Mazzaretto Ottavio Mattia
# Marta Ramirez, Melisa Menendez
#------------------------------------------------------------------------------

# %% IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import xarray as xr
from scipy import sin, cos, tan, arctan, arctan2, arccos, pi, radians
import os
import natsort
import glob
# %% FUNCTIONS

def findStrINlist(List,String):
    """
        Find string inside a list: 
            This funciton allows to find the position a string which is contained
            inside a list
    
        Parameters
        ----------
        List : list of strings
            List of the name of wathever you want, need to be a list of strings
            example: List = ['a','b','c']
        String : string
            string that whould be or not be contained inside the list
    
        Returns
        -------
        matched_indexes: np array
            np arry of the position of the string inside the list

    """
    matched_indexes = []
    i = 0
    length = len(List)
    
    while i < length:
        if String in List[i]:
            matched_indexes.append(int(i))
        i += 1        
    return np.array(matched_indexes, dtype = int)

def read_txt(fileName):
    """
    This is to read the text file generated for the case of Panama

    Parameters
    ----------
    fileName : str
        path/to/the/file.txt.

    Returns
    -------
    df : dataframe
        pandas dataframe of the file.txt with datetime vector.

    """
    # open the file
    f = open(fileName,'r', encoding="ISO-8859-1")
    # read each line
    l = f.readlines()
    # replace the tab '\t\n' with only '\n'
    l = [i.replace('\t\n','\n') for i in l]
    # find where the 'yyyy' starts to jump the header
    ll = findStrINlist(l,'yyyy')[0]
    # store the lines from 'yyyy' to the end
    l = l[ll:]

    # make this a list, split the strings
    li = [i.split() for i in l]
    # col_names = ['yyyy', 'mm', 'dd', 'hh', 'hs', 'tm02', 'tp', 'dir', 'nivel','marea']
    # the names are the first line
    col_names = li[0]
    # create a dataframe from the list
    df = pd.DataFrame(li[1:], columns = col_names)
    
    # convert from strings to float and integers
    for icol in df.columns:
        if icol in ['yyyy', 'mm', 'dd', 'hh']:
            dtype = 'int'
        else:
            dtype = 'float'
            
        df[icol] = df[icol].astype(dtype)

    # converting the string 'yyyy' 'mm' 'dd' 'hh' in datetime
    df['Datetime'] = pd.to_datetime(df['yyyy'].astype(str) + '/' +\
                                    df['mm'].astype(str) + '/' +\
                                    df['dd'].astype(str) + '/' +\
                                    df['hh'].astype(str),
                                    format='%Y/%m/%d/%H')
        
    return df

def MATLAB_distance(lat1, lon1, lat2, lon2):
    """
    MATLAB distance: function which allows to calculate the geodetic distance as 
    the MATLAB CODE DOES!! This is a matlab pythoned code, thank matlab
    between two nodes as is done in matlab with the function distance.
    This is the same as doing the matlab distance(lat1,lon1,lat2,lon2)
            INPUT:
                -lat1: first node latitude
                -lon1: first node longitude
                -lat2: second node latitude, can be array
                -lon2: second node longitude, can be array
    """
    r = 1 # as the esslipsoid(1) matlab
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    a = np.array([sin((lat2 - lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1)/2)**2])
    # ensure that a fall in the closed interval [0 1]
    a[a < 0] = 0 
    a[a > 1] = 1
    rng = np.rad2deg(2 * r * np.arctan2(np.sqrt(a),np.sqrt(1 - a)))
    return rng

def sel_lonlat(da, lon, lat,LONNAME= 'lon',LATNAME = 'lat',SITENAME = 'nvert', method=None, tolerance = 0.1*1e7):
        """Select site based on longitude and latitude.

        Args:
            - lon (float): longitude of the site.
            - lat (float): latitude of the site.
            - method (string): Method to use for inexact matches (None or 'nearest').

        Returns:
            - Dataset for the site defined by lon and lat.
        """
        if method not in (None, "nearest"):
            raise ValueError(
                "Invalid method. Expecting None or nearest. Got {}".format(method)
            )
        lons = da[LONNAME].values
        lats = da[LATNAME].values
        dist = MATLAB_distance(lat, lon, lats, lons)[0,:]
        isite = [int(dist.argmin())]
        if (method is None) and (dist[isite] > tolerance):
            raise ValueError(
                "lon={:f}, lat={:f} not found. Use method='nearest' to get lon={:f}, lat={:f}".format(
                    lon, lat, lons[isite][0], lats[isite][0]
                )
            )
        indexersdict = {
            k: isite
            for k in {LONNAME, LATNAME, SITENAME}.intersection(
                da.dims
            )
        }
        return da.isel(indexersdict)
    
def list_dirs(dir_to_list):
    """
    List in a distionary all the subdirectories and then store the files in a list

    Parameters
    ----------
    dir_to_list : str
        path to list the subdirectories.

    Returns
    -------
    h : dict
        dict of the subdirectories.

    """
    h={}
    for dirs in os.listdir(dir_to_list):
        if os.path.basename(dirs).split('_')[0] not in h.keys():
            h[os.path.basename(dirs).split('_')[0]] = {}
            
        if os.path.basename(dirs).split('_')[1] not in h[os.path.basename(dirs).split('_')[0]].keys():
            h[os.path.basename(dirs).split('_')[0]][os.path.basename(dirs).split('_')[1]] = []
        
        files = natsort.natsorted(glob.glob(os.path.join(dir_to_list,dirs,'*')))
        for file in files:
            h[os.path.basename(dirs).split('_')[0]][os.path.basename(dirs).split('_')[1]].append(file)
    return h