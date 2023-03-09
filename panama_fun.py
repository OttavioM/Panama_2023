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
    f = open(fileName,'r')
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
    # converting the string 'yyyy' 'mm' 'dd' 'hh' in datetime
    df['Datetime'] = pd.to_datetime(df['yyyy'].astype(str) + '/' +\
                                    df['mm'].astype(str) + '/' +\
                                    df['dd'].astype(str) + '/' +\
                                    df['hh'].astype(str),
                                    format='%Y/%m/%d/%H')
        
    return df