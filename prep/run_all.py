
# Everything needed to complete statewide analysis, outline for now

import os
import pandas as pd
import geopandas as gpd
import numpy as np
import pygridmet
import requests
import pyproj
from copy import deepcopy
from rasterstats import zonal_stats
from tqdm import tqdm

import sqlite3

import matplotlib.pyplot as plt

from utils.thredds import GridMet
from iwr.iwr_approx import iwr_daily

CLIMATE_COLS = {
    'etr': {
        'nc': 'agg_met_etr_1979_CurrentYear_CONUS',
        'var': 'daily_mean_reference_evapotranspiration_alfalfa',
        'col': 'etr_mm'},
    'pet': {
        'nc': 'agg_met_pet_1979_CurrentYear_CONUS',
        'var': 'daily_mean_reference_evapotranspiration_grass',
        'col': 'eto_mm'},
    'pr': {
        'nc': 'agg_met_pr_1979_CurrentYear_CONUS',
        'var': 'precipitation_amount',
        'col': 'prcp_mm'},
    'sph': {
        'nc': 'agg_met_sph_1979_CurrentYear_CONUS',
        'var': 'daily_mean_specific_humidity',
        'col': 'q_kgkg'},
    'srad': {
        'nc': 'agg_met_srad_1979_CurrentYear_CONUS',
        'var': 'daily_mean_shortwave_radiation_at_surface',
        'col': 'srad_wm2'},
    'vs': {
        'nc': 'agg_met_vs_1979_CurrentYear_CONUS',
        'var': 'daily_mean_wind_speed',
        'col': 'u10_ms'},
    'tmmx': {
        'nc': 'agg_met_tmmx_1979_CurrentYear_CONUS',
        'var': 'daily_maximum_temperature',
        'col': 'tmax_k'},
    'tmmn': {
        'nc': 'agg_met_tmmn_1979_CurrentYear_CONUS',
        'var': 'daily_minimum_temperature',
        'col': 'tmin_k'},
    'th': {
        'nc': 'agg_met_th_1979_CurrentYear_CONUS',
        'var': 'daily_mean_wind_direction',
        'col': 'wdir_deg'},
    # 'vpd': {
    #     'nc': 'agg_met_vpd_1979_CurrentYear_CONUS',
    #     'var': 'daily_mean_vapor_pressure_deficit',
    #     'col': 'vpd_kpa'}
}

COLUMN_ORDER = ['date',
                'year',
                'month',
                'day',
                'centroid_lat',
                'centroid_lon',
                'elev_m',
                # 'u2_ms',
                'tmin_c',
                'tmax_c',
                'srad_wm2',
                # 'ea_kpa',
                # 'pair_kpa',
                'prcp_mm',
                'etr_mm',
                'eto_mm',
                'etr_mm_uncorr',
                'eto_mm_uncorr']  # which of these do we need?


def gridmet_match(fields, gridmet_points, fields_join):
    """This depends on running 'Raster Pixels to Points' on a WGS Gridmet raster,
     attributing GFID, lat, and lon in the attribute table, and saving to project crs: 5071.
     GFID is an arbitrary identifier e.g., @row_number. It further depends on projecting the
     rasters to EPSG:5071, using the project.sh bash script (or gis)

     The reason we're not just doing a zonal stat on correction surface for every object is that
     there may be many fields that only need data from one gridmet cell. This prevents us from downloading
     many redundant data sets. Looks like it often works out to roughly 1 gridmet data pull for every 10 fields."""

    convert_to_wgs84 = lambda x, y: pyproj.Transformer.from_crs('EPSG:5071', 'EPSG:4326').transform(x, y)

    fields = gpd.read_file(fields)
    gridmet_pts = gpd.read_file(gridmet_points)

    fields['GFID'] = np.nan
    fields['STATION_ID'] = np.nan

    print('Find field-gridmet joins')

    gridmet_targets = []
    for i, field in tqdm(fields.iterrows(), total=fields.shape[0]):

        xx, yy = field['geometry'].centroid.x, field['geometry'].centroid.y
        lat, lon = convert_to_wgs84(xx, yy)
        fields.at[i, 'LAT'] = lat
        fields.at[i, 'LON'] = lon

        close_points = gridmet_pts.sindex.nearest(field['geometry'].centroid)
        closest_fid = gridmet_pts.iloc[close_points[1]]['GFID'].iloc[0]

        fields.at[i, 'GFID'] = closest_fid
        fields.at[i, 'STATION_ID'] = closest_fid
        gridmet_targets.append(closest_fid)

        # print('Matched {} to {}'.format(field['fid'], closest_fid))

        g = GridMet('elev', lat=lat, lon=lon)
        elev = g.get_point_elevation()
        fields.at[i, 'ELEV_GM'] = elev

    fields.to_file(fields_join, crs='EPSG:5071')

    len_ = len(gridmet_targets)
    print('Get gridmet for {} target points'.format(len_))
    gridmet_pts.index = gridmet_pts['GFID']


def gridmet_match_db(fields, gridmet_points, fields_join):
    """This depends on running 'Raster Pixels to Points' on a WGS Gridmet raster,
     attributing GFID, lat, and lon in the attribute table, and saving to project crs: 5071.
     GFID is an arbitrary identifier e.g., @row_number. It further depends on projecting the
     rasters to EPSG:5071, using the project.sh bash script (or gis)

     The reason we're not just doing a zonal stat on correction surface for every object is that
     there may be many fields that only need data from one gridmet cell. This prevents us from downloading
     many redundant data sets. Looks like it often works out to roughly 1 gridmet data pull for every 10 fields."""

    convert_to_wgs84 = lambda x, y: pyproj.Transformer.from_crs('EPSG:5071', 'EPSG:4326').transform(x, y)

    # fields = gpd.read_file(fields) # This is now already read in, and thus redundant
    print('Finding field-gridmet joins for {} fields'.format(fields.shape[0]))

    gridmet_pts = gpd.read_file(gridmet_points)

    existing_fields = pd.read_sql("SELECT DISTINCT fid FROM {}".format(fields_join), con)
    # If there are any new fields,
    if ~existing_fields.isin(fields['fid']).any().any():
        # Remove potential duplicate fields
        fields = fields.loc[~fields['fid'].isin(existing_fields)]
        fields['gfid'] = np.nan
        # and run matching algorithm
        for i, field in tqdm(fields.iterrows(), total=fields.shape[0]):
            xx, yy = field['geometry'].centroid.x, field['geometry'].centroid.y
            lat, lon = convert_to_wgs84(xx, yy)
            fields.at[i, 'lat'] = lat
            fields.at[i, 'lon'] = lon

            close_points = gridmet_pts.sindex.nearest(field['geometry'].centroid)
            closest_fid = gridmet_pts.iloc[close_points[1]]['GFID'].iloc[0]
            fields.at[i, 'gfid'] = closest_fid
            # print('Matched {} to {}'.format(field['fid'], closest_fid))

            g = GridMet('elev', lat=lat, lon=lon)
            elev = g.get_point_elevation()
            fields.at[i, 'elev_gm'] = elev
        # Do not store geometry in db
        fields = fields.drop(labels='geometry', axis='columns')
        # Save new entries to db
        fields.to_sql(fields_join, con, if_exists='append', index=False)
        print('Found {} gridmet target points for {} new fields'.format(len(fields['gfid'].unique()), fields.shape[0]))
        print()
    else:
        print("These fields are already in the database")
        print()


# Bad, do not use:
def corrected_gridmet_hyriver(gridmet_points, fields_join, gridmet_csv_dir, gridmet_ras):
    """This depends on running 'Raster Pixels to Points' on a WGS Gridmet raster,
     attributing GFID, lat, and lon in the attribute table, and saving to project crs: 5071.
     GFID is an arbitrary identifier e.g., @row_number. It further depends on projecting the
     rasters to EPSG:5071, usng the project.sh bash script

     The reason we're not just doing a zonal stat on correction surface for every object is that
     there may be many fields that only need data from one gridmet cell. This prevents us from downloading
     many redundant data sets."""

    fields = gpd.read_file(fields_join)
    gridmet_pts = gpd.read_file(gridmet_points)
    gridmet_pts.index = gridmet_pts['GFID']

    rasters = []  # Correction surfaces
    for v in ['eto', 'etr']:
        [rasters.append(os.path.join(gridmet_ras, 'gridmet_corrected_{}_{}.tif'.format(v, m))) for m in range(1, 13)]

    gridmet_targets = {}  # Getting correction factors for the required gridmet stations
    for i in fields['GFID'].unique():
        gridmet_targets[i] = {str(m): {} for m in range(1, 13)}
        geo = gridmet_pts.at[i, 'geometry']
        gdf = gpd.GeoDataFrame({'geometry': [geo]})
        for r in rasters:
            splt = r.split('_')
            _var, month = splt[-2], splt[-1].replace('.tif', '')
            stats = zonal_stats(gdf, r, stats=['mean'])[0]['mean']
            gridmet_targets[i][month].update({_var: stats})

    len_ = len(gridmet_targets)
    print('Get gridmet for {} target points'.format(len_))
    gridmet_pts.index = gridmet_pts['GFID']

    # Entire period of record, to end of last full calendar year.
    start = '1980-01-01'
    end = '2023-12-31'

    for k, v in tqdm(gridmet_targets.items(), total=len_):
        # df, first = pd.DataFrame(), True
        r = gridmet_pts.loc[k]

        lat, lon = r['lat'], r['lon']
        print(lon, lat)
        df = pygridmet.get_bycoords((lon, lat), (start, end), variables=CLIMATE_COLS.keys())  # , crs=5071)
        # Why did it not like the different coordinate system? What coordinate system am I in, anyway?

        df['date'] = [i.strftime('%Y-%m-%d') for i in df.index]
        df['year'] = [i.year for i in df.index]
        df['month'] = [i.month for i in df.index]
        df['day'] = [i.day for i in df.index]
        df['centroid_lat'] = [lat for _ in range(df.shape[0])]
        df['centroid_lon'] = [lon for _ in range(df.shape[0])]
        g = GridMet('elev', lat=lat, lon=lon)
        elev = g.get_point_elevation()
        df['elev_m'] = [elev for _ in range(df.shape[0])]

        for _var in ['etr', 'eto']:
            variable = '{}_mm'.format(_var)
            for month in range(1, 13):
                corr_factor = v[str(month)][_var]
                idx = [i for i in df.index if i.month == month]
                df.loc[idx, '{}_uncorr'.format(variable)] = df.loc[idx, variable]
                df.loc[idx, variable] = df.loc[idx, '{}_uncorr'.format(variable)] * corr_factor

        # zw = 10
        # df['u2_ms'] = wind_height_adjust(
        #     df.u10_ms, zw)
        # df['pair_kpa'] = air_pressure(
        #     df.elev_m, method='asce')
        # df['ea_kpa'] = actual_vapor_pressure(
        #     df.q_kgkg, df.pair_kpa)

        df['tmax_c'] = df.tmax_k - 273.15
        df['tmin_c'] = df.tmin_k - 273.15

        df = df[COLUMN_ORDER]
        _file = os.path.join(gridmet_csv_dir, 'gridmet_historical_{}.csv'.format(r['GFID']))
        df.to_csv(_file, index=False)


def corrected_gridmet(gridmet_points, fields_join, gridmet_csv_dir, gridmet_ras):
    """This depends on running 'Raster Pixels to Points' on a WGS Gridmet raster,
     attributing GFID, lat, and lon in the attribute table, and saving to project crs: 5071.
     GFID is an arbitrary identifier e.g., @row_number. It further depends on projecting the
     rasters to EPSG:5071, usng the project.sh bash script

     The reason we're not just doing a zonal stat on correction surface for every object is that
     there may be many fields that only need data from one gridmet cell. This prevents us from downloading
     many redundant data sets."""

    fields = gpd.read_file(fields_join)
    gridmet_pts = gpd.read_file(gridmet_points)
    gridmet_pts.index = gridmet_pts['GFID']

    rasters = []  # Correction surfaces
    for v in ['eto', 'etr']:
        [rasters.append(os.path.join(gridmet_ras, 'gridmet_corrected_{}_{}.tif'.format(v, m))) for m in range(1, 13)]

    gridmet_targets = {}  # Getting correction factors for the required gridmet stations
    for i in fields['GFID'].unique():
        gridmet_targets[i] = {str(m): {} for m in range(1, 13)}
        geo = gridmet_pts.at[i, 'geometry']
        gdf = gpd.GeoDataFrame({'geometry': [geo]})
        for r in rasters:
            splt = r.split('_')
            _var, month = splt[-2], splt[-1].replace('.tif', '')
            stats = zonal_stats(gdf, r, stats=['mean'])[0]['mean']
            gridmet_targets[i][month].update({_var: stats})

    len_ = len(gridmet_targets)
    print('Get gridmet for {} target points'.format(len_))
    gridmet_pts.index = gridmet_pts['GFID']

    # Entire period of record, to end of last full calendar year.
    start = '1979-01-01'
    end = '2023-12-31'

    for k, v in tqdm(gridmet_targets.items(), total=len_):
        df, first = pd.DataFrame(), True
        r = gridmet_pts.loc[k]
        for thredds_var, cols in CLIMATE_COLS.items():
            variable = cols['col']
            if not thredds_var:
                continue
            lat, lon = r['lat'], r['lon']
            g = GridMet(thredds_var, start=start, end=end, lat=lat, lon=lon)
            s = g.get_point_timeseries()
            df[variable] = s[thredds_var]

            if first:
                df['date'] = [i.strftime('%Y-%m-%d') for i in df.index]
                df['year'] = [i.year for i in df.index]
                df['month'] = [i.month for i in df.index]
                df['day'] = [i.day for i in df.index]
                df['centroid_lat'] = [lat for _ in range(df.shape[0])]
                df['centroid_lon'] = [lon for _ in range(df.shape[0])]
                g = GridMet('elev', lat=lat, lon=lon)
                elev = g.get_point_elevation()
                df['elev_m'] = [elev for _ in range(df.shape[0])]
                first = False

        for _var in ['etr', 'eto']:
            variable = '{}_mm'.format(_var)
            for month in range(1, 13):
                corr_factor = v[str(month)][_var]
                idx = [i for i in df.index if i.month == month]
                df.loc[idx, '{}_uncorr'.format(variable)] = df.loc[idx, variable]
                df.loc[idx, variable] = df.loc[idx, '{}_uncorr'.format(variable)] * corr_factor

        # zw = 10
        # df['u2_ms'] = wind_height_adjust(
        #     df.u10_ms, zw)
        # df['pair_kpa'] = air_pressure(
        #     df.elev_m, method='asce')
        # df['ea_kpa'] = actual_vapor_pressure(
        #     df.q_kgkg, df.pair_kpa)

        df['tmax_c'] = df.tmax_k - 273.15
        df['tmin_c'] = df.tmin_k - 273.15

        df = df[COLUMN_ORDER]
        _file = os.path.join(gridmet_csv_dir, 'gridmet_historical_{}.csv'.format(r['GFID']))
        df.to_csv(_file, index=False)


def corrected_gridmet_db(gridmet_points, fields_join, gridmet_csv_dir, gridmet_ras):
    """This depends on running 'Raster Pixels to Points' on a WGS Gridmet raster,
     attributing GFID, lat, and lon in the attribute table, and saving to project crs: 5071.
     GFID is an arbitrary identifier e.g., @row_number. It further depends on projecting the
     rasters to EPSG:5071, usng the project.sh bash script

     The reason we're not just doing a zonal stat on correction surface for every object is that
     there may be many fields that only need data from one gridmet cell. This prevents us from downloading
     many redundant data sets."""

    # Does this work instead?
    # fields = pd.read_sql("SELECT DISTINCT gfid FROM {} WHERE gfid "
    #                      "NOT IN SELECT DISTINCT gfid FROM {} ".format(fields_join, gridmet_csv_dir), con)

    # Looks for all new gridmet stations
    fields = pd.read_sql("SELECT DISTINCT gfid FROM {}".format(fields_join), con)
    existing_fields = pd.read_sql("SELECT DISTINCT gfid FROM {}".format(gridmet_csv_dir), con)
    # If there are any new gridmet points,
    if ~existing_fields.isin(fields['gfid']).any().any():
        # Remove potential duplicate locations
        fields = fields.loc[~fields['gfid'].isin(existing_fields)]
        # and run data fetching algorithm

        gridmet_pts = gpd.read_file(gridmet_points)
        gridmet_pts.index = gridmet_pts['GFID']

        rasters = []  # Correction surfaces
        for v in ['eto', 'etr']:
            [rasters.append(os.path.join(gridmet_ras, 'gridmet_corrected_{}_{}.tif'.format(v, m))) for m in range(1, 13)]

        gridmet_targets = {}  # Getting correction factors for the required gridmet stations
        for i in fields['gfid'].unique():
            gridmet_targets[i] = {str(m): {} for m in range(1, 13)}
            geo = gridmet_pts.at[i, 'geometry']
            gdf = gpd.GeoDataFrame({'geometry': [geo]})
            for r in rasters:
                splt = r.split('_')
                _var, month = splt[-2], splt[-1].replace('.tif', '')
                stats = zonal_stats(gdf, r, stats=['mean'])[0]['mean']
                gridmet_targets[i][month].update({_var: stats})

        len_ = len(gridmet_targets)
        print('Get gridmet for {} target points'.format(len_))
        gridmet_pts.index = gridmet_pts['GFID']

        # Entire period of record, to end of last full calendar year.
        start = '1979-01-01'
        end = '2023-12-31'

        for k, v in tqdm(gridmet_targets.items(), total=len_):
            df, first = pd.DataFrame(), True
            r = gridmet_pts.loc[k]
            for thredds_var, cols in CLIMATE_COLS.items():
                variable = cols['col']
                if not thredds_var:
                    continue
                lat, lon = r['lat'], r['lon']
                g = GridMet(thredds_var, start=start, end=end, lat=lat, lon=lon)
                s = g.get_point_timeseries()
                df[variable] = s[thredds_var]

                if first:
                    df['date'] = [i.strftime('%Y-%m-%d') for i in df.index]
                    df['year'] = [i.year for i in df.index]
                    df['month'] = [i.month for i in df.index]
                    df['day'] = [i.day for i in df.index]
                    df['centroid_lat'] = [lat for _ in range(df.shape[0])]
                    df['centroid_lon'] = [lon for _ in range(df.shape[0])]
                    g = GridMet('elev', lat=lat, lon=lon)
                    elev = g.get_point_elevation()
                    df['elev_m'] = [elev for _ in range(df.shape[0])]
                    first = False

            for _var in ['etr', 'eto']:
                variable = '{}_mm'.format(_var)
                for month in range(1, 13):
                    corr_factor = v[str(month)][_var]
                    idx = [i for i in df.index if i.month == month]
                    df.loc[idx, '{}_uncorr'.format(variable)] = df.loc[idx, variable]
                    df.loc[idx, variable] = df.loc[idx, '{}_uncorr'.format(variable)] * corr_factor

            # zw = 10
            # df['u2_ms'] = wind_height_adjust(
            #     df.u10_ms, zw)
            # df['pair_kpa'] = air_pressure(
            #     df.elev_m, method='asce')
            # df['ea_kpa'] = actual_vapor_pressure(
            #     df.q_kgkg, df.pair_kpa)

            df['tmax_c'] = df.tmax_k - 273.15
            df['tmin_c'] = df.tmin_k - 273.15

            df = df[COLUMN_ORDER]
            df['gfid'] = i
            df.to_sql(gridmet_csv_dir, con, if_exists='append', index=False)
        print()
    else:
        print("These gridmet data are already in the database")
        print()


# very similar to open_et_get_fields in point_comparison.py
def openet_get_fields_link(fields, start, end, et_too=False,
                           api_key='ZBXxCeBRsSgkeLvsROKVTDS1w9UV0xfOKyEJGTNcEEPT15DQsYfbB0uu1K9w'):
    """ Uses OpenET API timeseries multipolygon endpoint to get etof data given a Google Earth Engine asset.
    Prints the url, click link to download csv file; link lasts 5 minutes.
    Switch to export multipolygon? I might be able to automatically retrieve that.
    :fields: path to gee asset, called 'Table ID' in gee,
    potential form of 'projects/cloud_project/assets/asset_filename'
    :start: beginning of period of study, 'YYYY-MM-DD' format
    :end: end of period of study, 'YYYY-MM-DD' format
    :et_too: if True, also get link for downloading OpenET ensemble ET over same time period and set of fields
    :api_key: from user's OpenET account, Hannah's is default
    """

    # set your API key before making the request
    header = {"Authorization": api_key}

    # endpoint arguments
    args = {
        "date_range": [
            start,
            end
        ],
        "interval": "monthly",
        "asset_id": fields,
        "attributes": [
            "fid"
        ],
        "reducer": "mean",
        "model": "Ensemble",
        "variable": "ETof",  # "ETof" or "ET"
        "reference_et": "gridMET",
        "units": "in"
    }

    # query the api
    resp = requests.post(
        headers=header,
        json=args,
        url="https://openet-api.org/raster/timeseries/multipolygon"
    )
    print(resp.json())

    if et_too:
        # getting et variable too, in separate file.
        args.update({"variable": "ET"})
        resp = requests.post(
            headers=header,
            json=args,
            url="https://openet-api.org/raster/timeseries/multipolygon"
        )
        print(resp.json())

    pass


# This one is not going to work for now... something weird happening:
# "Cannot access OpenET cloud storage while ee-linked."
def openet_get_fields_export(fields, start, end, et_too=False,
                             api_key='ZBXxCeBRsSgkeLvsROKVTDS1w9UV0xfOKyEJGTNcEEPT15DQsYfbB0uu1K9w'):
    """ Uses OpenET API timeseries multipolygon endpoint to get etof data given a Google Earth Engine asset.
    Prints the url, click link to download csv file; link lasts 5 minutes.
    Switch to export multipolygon? I might be able to automatically retrieve that.
    :fields: path to gee asset, form of 'projects/cloud_project/assets/asset_filename'
    :start: beginning of period of study, 'YYYY-MM-DD' format
    :end: end of period of study, 'YYYY-MM-DD' format
    :et_too: if True, also get link for downloading OpenET ensemble ET over same time period and set of fields
    :api_key: from user's OpenET account, Hannah's is default
    """

    # set your API key before making the request
    header = {"Authorization": api_key}

    # endpoint arguments
    args = {
        "date_range": [
            start,
            end
        ],
        "interval": "monthly",
        "asset_id": fields,
        "attributes": [
            "fid"
        ],
        "reducer": "mean",
        "model": "Ensemble",
        "variable": "ETof",  # "ETof" or "ET"
        "reference_et": "gridMET",
        "units": "in"
    }

    # query the api
    resp = requests.post(
        headers=header,
        json=args,
        url="https://openet-api.org/raster/export/multipolygon"
    )
    print(resp.json())

    if et_too:
        # getting et variable too, in separate file.
        args.update({"variable": "ET"})
        resp = requests.post(
            headers=header,
            json=args,
            url="https://openet-api.org/raster/export/multipolygon"
        )
        print(resp.json())

    pass


def cu_analysis(shp, gridmet, start, end, etof, out):
    # Based on field_comparison in point_comparison.py
    # suggestion in memo: use gridmet for ET, then OpenET for crop coefficient (etof).
    gdf = gpd.read_file(shp)

    # loading in OpenET data from files
    file = pd.read_csv(etof, index_col=['fid', 'time'], date_format="%m/%d/%Y").sort_index()
    # file = pd.read_csv(etof, index_col=['fid', 'time'], date_format="%Y-%m-%d").sort_index()  # This was wrong

    summary = deepcopy(gdf)
    summary['etos'] = [-99.99 for _ in summary['fid']]  # gridMET seasonal ET
    summary['etbc'] = [-99.99 for _ in summary['fid']]  # blaney criddle ET based on gridMET weather data
    summary['etof'] = [-99.99 for _ in summary['fid']]  # crop coefficient/etof from OpenET ensemble
    summary['opnt_cu'] = [-99.99 for _ in summary['fid']]  # consumptive use calculated from gridMET ET
    summary['dnrc_cu'] = [-99.99 for _ in summary['fid']]  # consumptive sue calculated from blaney criddle ET

    for i, row in tqdm(gdf.iterrows(), total=len(gdf)):
        # Loading in gridMET
        lon, lat = row['LON'], row['LAT']
        elev = row['ELEV_GM']  # elevation from retrieved gridmet station

        # Are the names correct?
        gm_str = os.path.join(gridmet, 'gridmet_historical_{}.csv'.format(row['GFID']))
        grd = pd.read_csv(gm_str)
        # rename variables needed in blaney_criddle calculations
        grd = grd.rename(columns={'eto_mm': 'ETOS', 'tmin_c': 'MN', 'tmax_c': 'MX', 'prcp_mm': 'PP'})
        grd.index = pd.to_datetime(grd['date'])

        # crop gridmet to time period
        grd = grd[start:end]

        if row['itype'] == 'P':
            pivot = True
        else:
            pivot = False

        mfs = gpd.read_file('C:/Users/CND571/Documents/Data/management_factors/mt_county_management_factors.shp')

        # Get management factor for county that the field is in
        # How to determine the closest time period for management factor?
        # time periods in 'the rule': 1964–1973, 1973–2006, 1997–2006
        mflist = ['1964', '1973', '1997']
        mf = mfs[mfs['FIPS'] == row['fid'][:3]][mflist[2]].values[0]

        # Calculating bc seasonal ET
        grd['MM'] = (grd['MN'] + grd['MX']) / 2  # gridmet units are in mm and degrees celsius.
        bc, start1, end1 = iwr_daily(grd, lat, elev, pivot=pivot)

        bc_cu = mf * bc['cu'].sum()  # add management factor
        bc_pet = bc['u'].sum()
        # eff_precip = bc['ep'].sum()

        # Masking daily gridMET data to growing season
        grd['mday'] = ['{}-{}'.format(x.month, x.day) for x in grd.index]
        target_range = pd.date_range('2000-{}-{}'.format(start1.month, start1.day),
                                     '2000-{}-{}'.format(end1.month, end1.day))
        accept = ['{}-{}'.format(x.month, x.day) for x in target_range]
        grd['mask'] = [1 if d in accept else 0 for d in grd['mday']]
        grd = grd[grd['mask'] == 1]

        grd['ETOS'] = grd['ETOS'] / 25.4  # daily ET, mm to in
        # Sum over the season, get total seasonal consumptive use by year
        et_by_year = grd.groupby(grd.index.year)['ETOS'].sum()

        # Loading in OpenET eotf/kc
        df = file.loc[row['fid']]
        r_index = pd.date_range('2016-01-01', '2022-12-31', freq='D')  # avg over all data
        df = df.reindex(r_index)
        df = df.interpolate()
        df['mday'] = ['{}-{}'.format(x.month, x.day) for x in df.index]
        # target_range = pd.date_range('2016-05-09', '2016-09-19')  # Why was this different from above?
        # accept = ['{}-{}'.format(x.month, x.day) for x in target_range]
        df['mask'] = [1 if d in accept else 0 for d in df['mday']]
        df = df[df['mask'] == 1]

        # calculating consumptive use
        # average seasonal ET times crop coefficient minus effective precip and carryover (?)

        # no multipolygons
        summary.loc[i, 'etos'] = et_by_year.mean()
        summary.loc[i, 'etbc'] = bc_pet
        summary.loc[i, 'etof'] = df['etof'].mean()
        summary.loc[i, 'opnt_cu'] = et_by_year.mean() * df['etof'].mean()  # - eff_precip - carryover  # ???
        summary.loc[i, 'dnrc_cu'] = bc_cu

    summary = gpd.GeoDataFrame(summary)
    summary.to_file(out)
    pass


def cu_analysis_db(shp, gridmet, start, end, etof, out):
    # Based on field_comparison in point_comparison.py
    # suggestion in memo: use gridmet for ET, then OpenET for crop coefficient (etof).

    # # Looks for all new fields, not calculated over the given time frame.
    # fields = pd.read_sql("SELECT DISTINCT fid FROM {}".format(shp), con)
    # existing_fields = pd.read_sql("SELECT DISTINCT fid, start, end FROM {}".format(out), con)
    # # If there are any new fields,
    # if ~existing_fields.isin(fields['fid']).any().any():
    #     # Remove potential duplicate fields
    #     fields = fields.loc[~fields['fid'].isin(existing_fields)]
    #     # and run data cu algorithm

    # complete analysis for all remaining fields...
    gdf = pd.read_sql("SELECT DISTINCT * FROM {}".format(shp), con)

    # loading in OpenET data from files
    file = pd.read_csv(etof, index_col=['fid', 'time'], date_format="%m/%d/%Y").sort_index()
    # file = pd.read_csv(etof, index_col=['fid', 'time'], date_format="%Y-%m-%d").sort_index()  # This was wrong

    # summary = deepcopy(gdf)
    summary = pd.DataFrame(gdf['fid'])
    summary['start'] = start
    summary['end'] = end
    summary['etos'] = [-99.99 for _ in summary['fid']]  # gridMET seasonal ET
    summary['etbc'] = [-99.99 for _ in summary['fid']]  # blaney criddle ET based on gridMET weather data
    summary['etof'] = [-99.99 for _ in summary['fid']]  # crop coefficient/etof from OpenET ensemble
    summary['opnt_cu'] = [-99.99 for _ in summary['fid']]  # consumptive use calculated from gridMET ET
    summary['dnrc_cu'] = [-99.99 for _ in summary['fid']]  # consumptive sue calculated from blaney criddle ET

    for i, row in tqdm(gdf.iterrows(), total=len(gdf)):
        # Loading in gridMET
        lon, lat = row['lon'], row['lat']
        elev = row['elev_gm']  # elevation from retrieved gridmet station
        #
        # grd = pd.read_sql("SELECT date, eto_mm, tmin_c, tmax_c, prcp_mm FROM {} WHERE gfid={} AND "
        #                   "date(date) BETWEEN date('{}') AND date('{}')".format(gridmet, row['gfid'], start, end), con)
        grd = pd.read_sql("SELECT date, eto_mm, tmin_c, tmax_c, prcp_mm FROM ? WHERE gfid=? AND "
                          "date(date) BETWEEN date('?') AND date('?')", con, params=(gridmet, row['gfid'], start, end))
        # rename variables needed in blaney_criddle calculations
        grd = grd.rename(columns={'eto_mm': 'ETOS', 'tmin_c': 'MN', 'tmax_c': 'MX', 'prcp_mm': 'PP'})
        grd.index = pd.to_datetime(grd['date'])

        if row['itype'] == 'P':
            pivot = True
        else:
            pivot = False

        mfs = gpd.read_file('C:/Users/CND571/Documents/Data/management_factors/mt_county_management_factors.shp')

        # Get management factor for county that the field is in
        # How to determine the closest time period for management factor?
        # time periods in 'the rule': 1964–1973, 1973–2006, 1997–2006
        mflist = ['1964', '1973', '1997']
        mf = mfs[mfs['FIPS'] == row['fid'][:3]][mflist[2]].values[0]

        # Calculating bc seasonal ET
        grd['MM'] = (grd['MN'] + grd['MX']) / 2  # gridmet units are in mm and degrees celsius.
        bc, start1, end1 = iwr_daily(grd, lat, elev, pivot=pivot)

        bc_cu = mf * bc['cu'].sum()  # add management factor
        bc_pet = bc['u'].sum()
        # eff_precip = bc['ep'].sum()

        # Masking daily gridMET data to growing season
        grd['mday'] = ['{}-{}'.format(x.month, x.day) for x in grd.index]
        target_range = pd.date_range('2000-{}-{}'.format(start1.month, start1.day),
                                     '2000-{}-{}'.format(end1.month, end1.day))
        accept = ['{}-{}'.format(x.month, x.day) for x in target_range]
        grd['mask'] = [1 if d in accept else 0 for d in grd['mday']]
        grd = grd[grd['mask'] == 1]

        grd['ETOS'] = grd['ETOS'] / 25.4  # daily ET, mm to in
        # Sum over the season, get total seasonal consumptive use by year
        et_by_year = grd.groupby(grd.index.year)['ETOS'].sum()

        # Loading in OpenET eotf/kc
        df = file.loc[row['fid']]
        r_index = pd.date_range('2016-01-01', '2022-12-31', freq='D')  # avg over all data
        df = df.reindex(r_index)
        df = df.interpolate()
        df['mday'] = ['{}-{}'.format(x.month, x.day) for x in df.index]
        # target_range = pd.date_range('2016-05-09', '2016-09-19')  # Why was this different from above?
        # accept = ['{}-{}'.format(x.month, x.day) for x in target_range]
        df['mask'] = [1 if d in accept else 0 for d in df['mday']]
        df = df[df['mask'] == 1]

        # calculating consumptive use
        # average seasonal ET times crop coefficient minus effective precip and carryover (?)

        # no multipolygons
        summary.loc[i, 'etos'] = et_by_year.mean()
        summary.loc[i, 'etbc'] = bc_pet
        summary.loc[i, 'etof'] = df['etof'].mean()
        summary.loc[i, 'opnt_cu'] = et_by_year.mean() * df['etof'].mean()  # - eff_precip - carryover  # ???
        summary.loc[i, 'dnrc_cu'] = bc_cu

    summary.to_sql(out, con, if_exists='append', index=False)
    # summary = gpd.GeoDataFrame(summary)
    # summary.to_file(out)


def create_shapefile(name, field_select, variables, geo_source):
    """ Take list of fields desired, list of variables describing which columns to include,
    and pull data from appropriate databases, then save the resulting dataframe to a shapefile.
    Should I append columns to a shapefile? Otherwise, where am I getting the geomtery from?"""
    pass


if __name__ == '__main__':
    # # need to fetch OpenET data for each county separately (and first...)
    # blargh = ['019_Daniels', '033_Garfield', '051_Liberty', '061_Mineral', '101_Toole']
    # start_por = "2016-01-01"  # What is the period of record for OpenET etof?
    # end_por = "2022-12-31"
    # for i in blargh:
    #     gee_asset = 'projects/ee-hehaugen/assets/{}'.format(i)
    #     openet_get_fields_link(gee_asset, start_por, end_por)

    # sqlite database for gridmet data
    # con = sqlite3.connect("C:/Users/CND571/Documents/Data/tutorial.db")
    con = sqlite3.connect("C:/Users/CND571/Documents/Data/opnt_analysis_02092024.db")

    # sqlite database table names
    gm_ts, fields_db, results = 'gridmet_ts', 'field_data', 'field_cu_results'

    # Initialize tables with correct column names - Issue here with hard-coding?
    # Only run once
    cur = con.cursor()
    # These don't work. Why?
    # cur.execute("CREATE TABLE ?(date, year, month, day, centroid_lat, centroid_lon, elev_m, "
    #             "tmin_c, tmax_c, srad_wm2, prcp_mm, etr_mm, eto_mm, etr_mm_uncorr, eto_mm_uncorr, gfid)", (gm_ts,))
    # cur.execute("CREATE TABLE ?(fid, itype, usage, mapped_by, gfid, lat, lon, elev_gm)", (fields,))
    # cur.execute("CREATE TABLE ?(fid, start, end, etos, etbc, etof, opnt_cu, dnrc_cu)", (results,))
    # These do work.
    cur.execute("CREATE TABLE {}(date, year, month, day, centroid_lat, centroid_lon, elev_m, "
                "tmin_c, tmax_c, srad_wm2, prcp_mm, etr_mm, eto_mm, etr_mm_uncorr, eto_mm_uncorr, gfid)".format(gm_ts))
    cur.execute("CREATE TABLE {}(fid, itype, usage, mapped_by, county, gfid, lat, lon, elev_gm)".format(fields_db))
    cur.execute("CREATE TABLE {}(fid, start, end, etos, etbc, etof, opnt_cu, dnrc_cu)".format(results))

    # gridmet information
    gm_d = 'C:/Users/CND571/Documents/Data/gridmet'  # location of general gridmet files
    gridmet_cent = os.path.join(gm_d, 'gridmet_centroids_MT.shp')
    rasters_ = os.path.join(gm_d, 'correction_surfaces_aea')  # correction surfaces, one for each month and variable.
    # Management factor and effective precip table files are called in functions.

    # Loading in state irrigation dataset
    mt_fields = gpd.read_file("C:/Users/CND571/Documents/Data/sid_30JAN2024_all.shp")
    mt_fields['county'] = mt_fields['fid'].str.slice(0, 3)
    county_count = mt_fields['county'].value_counts(ascending=True)
    # print(county_count)
    # print(county_count.index[:3])

    pos_start = '2016-01-01'
    pos_end = '2022-12-31'

    # Running analysis for smallest county
    i = 0
    county_id = county_count.index[i]
    print("County {}".format(county_id))
    county_fields = mt_fields[mt_fields['county'] == county_id]
    # print(county_fields)
    gridmet_match_db(county_fields, gridmet_cent, fields_db)  # short
    corrected_gridmet_db(gridmet_cent, fields_db, gm_ts, rasters_)  # very long
    etof_file = "C:/Users/CND571/Documents/Data/ensemble_monthly_etof_{}.csv".format(county_id)
    cu_analysis_db(fields_db, gm_ts, pos_start, pos_end, etof_file, results)  # short

    # # # # # # # # # # # # # # # #

    # one county: 061_mineral (59 fields)
    # Step 0: file locations
    # region = '061_mineral'  # name of location/region we are doing analysis for
    # d = 'C:/Users/CND571/Documents/Data/061_mineral_county'  # location of county files
    # gm_d = 'C:/Users/CND571/Documents/Data/gridmet'  # location of general gridmet files

    # Step 1: gridmet lookup table
    # output: shapefile
    # region_fields = os.path.join(d, '{}_5071.shp'.format(region))  # Shapefile of irrigated fields, in CRS 'EPSG:5071'
    # gridmet_cent = os.path.join(gm_d, 'gridmet_centroids_MT.shp')
    # fields_gfid = os.path.join(d, '{}_fields_gfid.shp'.format(region))  # shapefile linking field ids to gridmet ids
    # DO I want this to be a shapefile, or just a table in the database?
    # Find all relevant gridmet points and associate them with the fields
    # gridmet_match(region_fields, gridmet_cent, fields_gfid)
    # fields_gfid_db = 'field_data'
    # met_db = 'gridmet'
    # fields_gfid_db = 'fields_gfid'  # Name of table to save pairing to.
    # met_db = 'gridmet'  # name of table to save gridmet time series to.
    # gridmet_match_db(region_fields, gridmet_cent, fields_gfid_db)

    # Step 2: gridmet data retrieval
    # output: csv files, one for each gridmet point
    # met_dir = os.path.join(d, 'gridmet_timeseries')  # folder in which to save all gridmet time series files
    # rasters_ = os.path.join(gm_d, 'correction_surfaces_aea')  # correction surfaces, one for each month and variable.
    # Fetch all gridmet data for the fields identified in gridmet_match
    # corrected_gridmet(gridmet_cent, fields_gfid, met_dir, rasters_)
    # corrected_gridmet_db(gridmet_cent, fields_gfid_db, met_db, rasters_)

    # Step 3: openet data retrieval
    # output: csv file downloaded with produced link.
    # will need to figure out how to use export endpoint for larger sets.
    # Fetch all OpenET monthly etof data for fields
    # daily frequency would require separate queries for each month.
    # gee_asset = 'projects/ee-hehaugen/assets/{}_fields_gfid'.format(region)
    # start_pos = "2016-01-01"  # What is the period of record for OpenET etof?
    # end_pos = "2022-12-31"
    # openet_get_fields_link(gee_asset, start_pos, end_pos)

    # Step 4: calculate consumptive use (iwr and
    # yields average data over time period specified below, with season determined based on gridmet meteorology
    # independently for each field/gridmet point
    #
    # pos_start = '2016-01-01'
    # pos_end = '2022-12-31'
    # etof_file = os.path.join(d, 'ensemble_monthly_etof.csv')
    # out_file = os.path.join(d, '{}_cu_results.shp'.format(region))  # file/directory to save iwr results to
    # cu_analysis(fields_gfid, met_dir, pos_start, pos_end, etof_file, out_file)
    # out_file = 'field_cu_results'
    # cu_analysis_db(fields_gfid_db, met_db, pos_start, pos_end, etof_file, out_file)

    # # Step 5: visualization
    # # Any plotting? This is all analysis, and I imagine things would mostly be plotted in qgis.
    #
    # data = gpd.read_file(out_file)
    # print(data.columns)
    #
    # plt.figure(figsize=(10, 5))
    #
    # plt.subplot(121)
    # plt.plot([24, 28], [24, 28], 'k', zorder=4)
    # plt.scatter(data['etos'], data['etbc'], c=data['GFID'], zorder=3)
    # plt.xlabel('GridMET average seasonal ET')
    # plt.ylabel('BC average seasonal ET')
    # plt.grid()
    #
    # plt.subplot(122)
    # plt.plot([10, 22], [10, 22], 'k', zorder=4)
    # plt.scatter(data['opnt_cu'], data['dnrc_cu'], c=data['GFID'], zorder=3)
    # plt.xlabel('GridMET average seasonal CU')
    # plt.ylabel('BC average seasonal CU')
    # plt.grid()
    # plt.tight_layout()
    #
    # plt.figure()
    # plt.hist(data['ELEV_GM'])
    #
    # plt.show()

    # Playing with sqlite3


    # gfids = [69532, 69533, 69534, 70920, 72307]
    # gfids = [69532, 69533, 69534, 70920, 72307, 72308, 72309, 72311, 73695,
    #          75084, 76471, 77856, 77857, 77858, 77859, 79248, 80634]
    # gfids = [70920, 72307]
    # for i in gfids:
    #
    #     existing_stations = pd.read_sql("SELECT DISTINCT gfid FROM gridmet", con)
    #     # If that gfid is not in the table, add data from the new station
    #     if ~existing_stations.isin([i]).any().any():
    #         df = pd.read_csv('C:/Users/CND571/Documents/Data/061_mineral_county/gridmet_timeseries/'
    #                          'gridmet_historical_{}.0.csv'.format(i))
    #         df['gfid'] = i
    #         df.to_sql('gridmet', con, if_exists='append', index=False)

    # con

    # df = pd.read_sql("SELECT DISTINCT gfid FROM gridmet", con)
    # print(df)

    # # Looks for all new fields, not calculated over the given time frame.
    # fields = pd.read_sql("SELECT DISTINCT fid FROM field_data", con)
    # fields['start'] = pos_start
    # fields['end'] = pos_end
    # print(fields)
    # print()
    # existing_fields = pd.read_sql("SELECT DISTINCT fid, start, end FROM {}".format(out_file), con)
    # print(existing_fields)
    # print()
    # # If there are any new field/time period combinations,

    # print(existing_fields.isin(fields))

    # print(fields[~fields.isin(existing_fields)])

    # print(existing_fields.isin(fields).any())
    # print(existing_fields.isin(fields).any().any)

    # if ~existing_fields.isin(fields).any():
    #     # Remove potential duplicates
    #     fields = fields.loc[~fields.isin(existing_fields)]
    #     # and run data cu algorithm
    # else:
    #     print("I already have those.")
    # min_fields = gpd.read_file(region_fields)

# ========================= EOF ====================================================================
