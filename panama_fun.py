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

import geopandas as gpd

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from windrose.windrose import WindroseAxes
from shapely.affinity import affine_transform as T
from shapely.affinity import rotate as R
import shapely.geometry
import gzip
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

def find_file_ID(filesTxt,ID):
    """
    This function allows to find all the ID of the nodes selected by the user
    
    
    Parameters
    ----------
    filesTxt : list
        list of the txt.gz files
    ID : int
        integer of the ID of the node
    
    Returns
    -------
    fileID: str
        string corresponding to the name of the file
    """
    fileID = filesTxt[findStrINlist(filesTxt, '%04.0f' %(ID))[0]]
    return fileID

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
    if fileName.endswith('.txt'):
        # read if it is txt
        f = open(fileName,'r', encoding="ISO-8859-1")
    elif fileName.endswith('.txt.gz'):
        # read with gzip the file
        f = gzip.open(fileName, 'rt', encoding='unicode_escape')
    else:
        raise ValueError('File must end with .txt or .txt.gz')
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
    f.close()    
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
        isite = int(dist.argmin())
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
    # create empty dictionary
    h={}
    # loop over the subdirectories
    for dirs in os.listdir(dir_to_list):
        # check if the subdirecoty first part exists
        if os.path.basename(dirs).split('_')[0] not in h.keys():
            h[os.path.basename(dirs).split('_')[0]] = {}
        # check if the subdirecoty second part exists
        if os.path.basename(dirs).split('_')[1] not in h[os.path.basename(dirs).split('_')[0]].keys():
            h[os.path.basename(dirs).split('_')[0]][os.path.basename(dirs).split('_')[1]] = []

        # Loop over the files in the subdirectory
        files = natsort.natsorted(glob.glob(os.path.join(dir_to_list,dirs,'*')))
        for file in files:
            h[os.path.basename(dirs).split('_')[0]][os.path.basename(dirs).split('_')[1]].append(file)
    return h


def plot_dpt_netcdf(filePath):
    """
    This function allows to plot the dpt for a netcdf file, this is an example on how to use plotly and
    the plot of a variable spatially.
    
    
    Parameters
    ----------
    filePath : string,
        path/to/folder_nc
    
    Returns
    -------
        None
    """
    # create the dictionary of the folders and subfolders
    netCdfFiles = list_dirs(os.path.join(filePath,"ficheros_netcdf"))
    # load the dataset example in pacific and waves
    dauxP = xr.open_dataset(netCdfFiles['OLAS']['PACIFICO'][0])
    # Rename the station coordinate so that you don't overwrite it
    dauxP = dauxP.assign_coords({'nvert':dauxP['nvert'].data})
    dauxP = dauxP.rename_vars({'latitude':'lat','longitude':'lon'})
    # the same waves but in the caribbean
    dauxC = xr.open_dataset(netCdfFiles['OLAS']['CARIBE'][0])
    # Rename the station coordinate so that you don't overwrite it
    oldauxCsP = dauxC.assign_coords({'nvert':dauxC['nvert'].data})
    dauxC = dauxC.rename_vars({'latitude':'lat','longitude':'lon'})
    
    # create the scatter map_box of the dataset already loaded, the range color is 
    # 0-100 due to the fact that some depth is very high and the automatic decision of 
    # the clorbar can be too high
    t = px.scatter_mapbox(lat=dauxP.lat, lon=dauxP.lon, color=dauxP.z, range_color = [0,100], labels = {'color':'depth [m]'}).update_layout(mapbox={"style": "carto-positron", "zoom":6})
    # add the traces of the caribbean part
    t.add_traces(data = px.scatter_mapbox(lat=dauxC.lat, lon=dauxC.lon, color=dauxC.z).data)
    t.show()
    return
 
def get_temporal_series(filePath):
    """
    Funciton that allows to get the temporal series of a all the nodes in the netcdf.
    This concatenate all the netcdf and load them in dask array, masked and then less memorey usage

    Parameters
    ----------
    filePath : list
        list of the files in the desired folder

    Returns
    -------
    da: xarrayDataset
        dataset of all the time of the files
    """
    # open the dataset
    da = xr.open_mfdataset(filePath)
    # Rename the station coordinate so that you don't overwrite it
    da = da.assign_coords({'nvert':da['nvert'].data})
    da = da.rename_vars({'latitude':'lat','longitude':'lon'})
    # the concatenation of the variables that has no the time dimension
    # is increasing the size of this variables
    da['lon'] = (('nvert'),da['lon'][0,:].data)
    da['lat'] = (('nvert'),da['lat'][0,:].data)
    da['z'] = (('nvert'),da['z'][0,:].data)
    # the attributes have accents and also have special characters, then they are deleted due to the fact
    # that sometimes colab cannot decode these
    da.attrs = []

    return da

def quiver_map(da, step = 100, fecha = '2016-11-22T06:00:00',title = ''):
    """
    This is a quiver map, it is an example as a quiver map can be done with plotly, however, is hughe hand made
    
    
    Parameters
    ----------
    da : xarray dataset
        dataset that has at least: lat,lon,direction, hs, tp
    step : int, optional
        step to plot the data, it higher than lower data, step = 1 all the data are plotted, by default 100
    fecha : str, optional
        date in which the quiver will diaplayed, by default '2016-11-22T06:00:00'
    title : str, optional
        title of the figure, by default ''
    
    Returns
    -------
        None
    """

    # create with shapely the arrow
    a = shapely.wkt.loads(
            "POLYGON ((-0.5 0.1, 0.5 0.1, 0.2 0.4, 1 0, 0.2 -0.4, 0.5 -0.1, -0.5 -0.1, -0.5 -0.1, -0.5 -0.1, -0.5 0.1))")
    # creating a geopandas database
    gdf = (
        gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        # .set_crs("EPSG:4326", allow_override=True)
    )

    # quiv = px.scatter_mapbox(lat=da.lat[::step], lon=da.lon[::step]).update_layout(mapbox={"style": "carto-positron", "zoom":6})
    # creating a dataset in pandas that uses the variables that we decided before
    # a dataframe per each gdf geometry
    df_waves = pd.concat(
            [
                pd.DataFrame(
                    {
                        "lat": da.lat.data[::step],
                        "lon": da.lon.data[::step],
                        "d": -(da['dir'].sel(time = fecha)[::step,0] +90)%360,
                        "s": da['tp'].sel(time = fecha)[::step,0]*15*1e-4,
                        "e": da['hs'].sel(time = fecha)[::step,0]
                    }
                )
                for b in [gdf.sample(2)["geometry"].total_bounds for _ in range(5)]
            ]
            ).reset_index(drop=True)
        
    # fill the nans with 0 to have always data, the nans or nulls are not accepted in
    # the px.choropleth_mapbox
    df_waves = df_waves.fillna(0)
    # creating the quiver, In a Mapbox choropleth map, each row of data_frame is represented by a colored region on a Mapbox map
    # (GeoJSON-formatted dict) – Must contain a Polygon feature collection, with IDs, which are references from locations.
    quiv = px.choropleth_mapbox(
        df_waves,
        geojson=gpd.GeoSeries(
            df_waves.loc[:, ["lat", "lon", "d", "s"]].apply(
                lambda r: R(
                    T(shapely.affinity.translate(a, -a.exterior.xy[0][3], -a.exterior.xy[1][3]),
                        [r["s"], 0, 0, r["s"], r["lon"], r["lat"]]
                    ),
                    r["d"],
                    origin=(r["lon"], r["lat"]),
                    use_radians=False,
                ),
                axis=1,
            )
        ).__geo_interface__,
        locations=df_waves.index,
        color="e",
        labels = {'e':'hs [m]'},
    )

    # update alyout defining the map style and the zoom
    quiv.update_layout(title = title,mapbox={"style": "carto-positron", "zoom": 7, "center":{'lat':8.9,'lon':-81}}, margin={"l":0,"r":0,"t":20,"b":0})
    # change the last colorbar to be on the left:
    # quiv.update_layout(coloraxis_colorbar_x=-0.15)
    quiv.update_coloraxes(cmin = 0, cmax = 2.5, colorbar = {'title':'hs [m]'})
    quiv.show()

    return