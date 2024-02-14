
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
from iwr.iwr_approx import iwr_daily_fm

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

COUNTIES = {'001': 'Beaverhead', '003': 'Big Horn', '005': 'Blaine', '007': 'Broadwater', '009': 'Carbon',
            '011': 'Carter', '013': 'Cascade', '015': 'Chouteau', '017': 'Custer', '019': 'Daniels',
            '021': 'Dawson', '023': 'Deer Lodge', '025': 'Fallon', '027': 'Fergus', '029': 'Flathead',
            '031': 'Gallatin', '033': 'Garfield', '035': 'Glacier', '037': 'Golden Valley', '039': 'Granite',
            '041': 'Hill', '043': 'Jefferson', '045': 'Judith Basin', '047': 'Lake', '049': 'Lewis and Clark',
            '051': 'Liberty', '053': 'Lincoln', '055': 'McCone', '057': 'Madison', '059': 'Meagher',
            '061': 'Mineral', '063': 'Missoula', '065': 'Musselshell', '067': 'Park', '069': 'Petroleum',
            '071': 'Phillips', '073': 'Pondera', '075': 'Powder River', '077': 'Powell', '079': 'Prairie',
            '081': 'Ravalli', '083': 'Richland', '085': 'Roosevelt', '087': 'Rosebud', '089': 'Sanders',
            '091': 'Sheridan', '093': 'Silver Bow', '095': 'Stillwater', '097': 'Sweet Grass', '099': 'Teton',
            '101': 'Toole', '103': 'Treasure', '105': 'Valley', '107': 'Wheatland', '109': 'Wibaux',
            '111': 'Yellowstone'}


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
    # If there are any new fields, (Why does it work here and not for others?)
    if ~existing_fields['fid'].isin(fields['fid']).any().any():
        # Remove potential duplicate fields
        fields = fields.loc[~fields['fid'].isin(existing_fields['fid'])]
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
     rasters to EPSG:5071, using the project.sh bash script

     The reason we're not just doing a zonal stat on correction surface for every object is that
     there may be many fields that only need data from one gridmet cell. This prevents us from downloading
     many redundant data sets."""

    fields = gpd.read_file(fields_join)
    fields = fields.iloc[:10]  # Just pull the first station
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
    # start = '1980-01-01'
    # end = '2023-12-31'

    start = '2000-01-01'
    end = '2001-12-31'

    # gridmet_targets =
    for k, v in tqdm(gridmet_targets.items(), total=len_):
        # df, first = pd.DataFrame(), True
        r = gridmet_pts.loc[k]

        lat, lon = r['lat'], r['lon']
        # print(lon, lat)
        df = pygridmet.get_bycoords((lon, lat), (start, end), variables=CLIMATE_COLS.keys())  # , crs=5071)
        # Why did it not like the different coordinate system? What coordinate system am I in, anyway?
        # Rename columns
        old_col = df.columns
        new_col = [CLIMATE_COLS[i]['col'] for i in CLIMATE_COLS]
        names = dict(zip(old_col, new_col))
        df = df.rename(columns=names)

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


def corrected_gridmet_test(gridmet_points, fields_join, gridmet_csv_dir, gridmet_ras):
    """This depends on running 'Raster Pixels to Points' on a WGS Gridmet raster,
     attributing GFID, lat, and lon in the attribute table, and saving to project crs: 5071.
     GFID is an arbitrary identifier e.g., @row_number. It further depends on projecting the
     rasters to EPSG:5071, usng the project.sh bash script

     The reason we're not just doing a zonal stat on correction surface for every object is that
     there may be many fields that only need data from one gridmet cell. This prevents us from downloading
     many redundant data sets."""

    fields = gpd.read_file(fields_join)
    fields = fields[:5]
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
    # start = '1979-01-01'  # error with day 16435, December 29, 1944
    # end = '2023-12-31'

    start = '2000-01-01'  # Produces error with no description
    end = '2023-12-31'

    start = '2000-01-01'  # date error days 7670 to 15340, December 30, 1920 to December 30 1941???
    end = '2020-12-31'

    for k, v in tqdm(gridmet_targets.items(), total=len_):
        # df, first = pd.DataFrame(), True
        r = gridmet_pts.loc[k]
        for thredds_var, cols in CLIMATE_COLS.items():
            variable = cols['col']
            if not thredds_var:
                continue
            lat, lon = r['lat'], r['lon']
            g = GridMet(thredds_var, start=start, end=end, lat=lat, lon=lon)
            output = os.path.join(gridmet_csv_dir, '{}_{}.nc'.format(int(k), thredds_var))
            g.write_netcdf(output)
        #     s = g.get_point_timeseries()
        #     df[variable] = s[thredds_var]
        #
        #     if first:
        #         df['date'] = [i.strftime('%Y-%m-%d') for i in df.index]
        #         df['year'] = [i.year for i in df.index]
        #         df['month'] = [i.month for i in df.index]
        #         df['day'] = [i.day for i in df.index]
        #         df['centroid_lat'] = [lat for _ in range(df.shape[0])]
        #         df['centroid_lon'] = [lon for _ in range(df.shape[0])]
        #         g = GridMet('elev', lat=lat, lon=lon)
        #         elev = g.get_point_elevation()
        #         df['elev_m'] = [elev for _ in range(df.shape[0])]
        #         first = False
        #
        # for _var in ['etr', 'eto']:
        #     variable = '{}_mm'.format(_var)
        #     for month in range(1, 13):
        #         corr_factor = v[str(month)][_var]
        #         idx = [i for i in df.index if i.month == month]
        #         df.loc[idx, '{}_uncorr'.format(variable)] = df.loc[idx, variable]
        #         df.loc[idx, variable] = df.loc[idx, '{}_uncorr'.format(variable)] * corr_factor
        #
        # # zw = 10
        # # df['u2_ms'] = wind_height_adjust(
        # #     df.u10_ms, zw)
        # # df['pair_kpa'] = air_pressure(
        # #     df.elev_m, method='asce')
        # # df['ea_kpa'] = actual_vapor_pressure(
        # #     df.q_kgkg, df.pair_kpa)
        #
        # df['tmax_c'] = df.tmax_k - 273.15
        # df['tmin_c'] = df.tmin_k - 273.15
        #
        # df = df[COLUMN_ORDER]
        # _file = os.path.join(gridmet_csv_dir, 'gridmet_historical_{}.csv'.format(r['GFID']))
        # df.to_csv(_file, index=False)


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

    # remove duplicate fields
    fields = fields.loc[~fields['gfid'].isin(existing_fields['gfid'])]

    # If there are any new gridmet points,
    if len(fields) > 0:
        # run data fetching algorithm

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
        # start = '1979-01-01'
        start = '2000-01-01'  # shortened to make retrieval faster.
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
            df['gfid'] = k
            df.to_sql(gridmet_csv_dir, con, if_exists='append', index=False)
        print()
    else:
        print("These gridmet data are already in the database")
        print()


# very similar to open_et_get_fields in point_comparison.py
def openet_get_fields_link(fields, start, end, et_too=False,
                           api_key='C:/Users/CND571/Documents/OpenET_API.txt'):
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

    key_file = open(api_key, "r")
    api_key = key_file.readline()

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
        bc, start1, end1 = iwr_daily_fm(grd, lat, elev, pivot=pivot)

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
    """ Takes in database tables for fields ids and gridmet data, calculates average seasonal consumptive use
    over time period using both IWR/DNRC and OpenET methods, and adds results to a third database file."""

    print("Calculating consumptive use for fields")

    # Want to make look for all new fields not calculated over the given time frame.
    # fields = pd.read_sql("SELECT DISTINCT * FROM {}".format(shp), con, index_col='fid')
    # existing_fields = pd.read_sql("SELECT DISTINCT fid, start, end FROM {}".format(out), con, index_col='fid')
    fields = pd.read_sql("SELECT DISTINCT * FROM {}".format(shp), con)
    existing_fields = pd.read_sql("SELECT DISTINCT fid, start, end FROM {}".format(out), con)
    fields['start'] = pd.to_datetime(start)
    fields['end'] = pd.to_datetime(end)
    existing_fields['start'] = pd.to_datetime(existing_fields['start'])
    existing_fields['end'] = pd.to_datetime(existing_fields['end'])
    # print("goal fields:")
    # print(fields)
    # print()
    # print("existing_fields:")
    # print(existing_fields)
    # print()

    # remove duplicate field/time period combos
    print("fields to fetch:")
    gdf = (pd.merge(fields, existing_fields, how='outer', indicator=True)
           .query('_merge == "left_only"')
           .drop(columns=['_merge']))
    # print(gdf)
    # print()

    # remove duplicate fields
    # gdf = fields.loc[~fields['fid'].isin(existing_fields['fid'])]

    if len(gdf) > 0:
        # run data cu algorithm

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
            # print(row['gfid'])
            grd = pd.read_sql("SELECT date, eto_mm, tmin_c, tmax_c, prcp_mm FROM {} WHERE gfid={} AND date(date) "
                              "BETWEEN date('{}') AND date('{}')".format(gridmet, row['gfid'], start, end), con)
            # grd = pd.read_sql("SELECT date, eto_mm, tmin_c, tmax_c, prcp_mm FROM ? WHERE gfid=? AND date(date) "
            #                   "BETWEEN date('?') AND date('?')", con, params=(gridmet, row['gfid'], start, end))
            # rename variables needed in blaney_criddle calculations
            grd = grd.rename(columns={'eto_mm': 'ETOS', 'tmin_c': 'MN', 'tmax_c': 'MX', 'prcp_mm': 'PP'})
            grd.index = pd.to_datetime(grd['date'])

            if row['itype'] == 'P':
                pivot = True
                carryover = 0.5
            else:
                pivot = False
                carryover = 2.0

            mfs = gpd.read_file('C:/Users/CND571/Documents/Data/management_factors/mt_county_management_factors.shp')

            # Get management factor for county that the field is in
            # How to determine the closest time period for management factor?
            # time periods in 'the rule': 1964–1973, 1973–2006, 1997–2006
            mflist = ['1964', '1973', '1997']
            mf = mfs[mfs['FIPS'] == row['fid'][:3]][mflist[2]].values[0]

            # Calculating bc seasonal ET
            grd['MM'] = (grd['MN'] + grd['MX']) / 2  # gridmet units are in mm and degrees celsius.
            bc, start1, end1 = iwr_daily_fm(grd, lat, elev, pivot=pivot)

            bc_cu = mf * bc['cu'].sum()  # add management factor
            bc_pet = bc['u'].sum()
            eff_precip = bc['ep'].sum()

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
            df = pd.read_sql("SELECT time, etof FROM {} WHERE fid='{}' AND date(time) BETWEEN date('{}') "
                             "AND date('{}')".format(etof, row['fid'], start, end), con,
                             index_col='time', parse_dates={'time': '%Y-%m-%d'})
            r_index = pd.date_range('2016-01-01', '2022-12-31', freq='D')  # avg over all data
            df = df.reindex(r_index)
            df = df.interpolate()
            df['mday'] = ['{}-{}'.format(x.month, x.day) for x in df.index]
            df['mask'] = [1 if d in accept else 0 for d in df['mday']]
            df = df[df['mask'] == 1]

            # calculating consumptive use
            # IWR: sum of monthly ET minus effective precip minus carryover, all times management factor
            # sum(u_month - ep_month - co_month) * mf
            # OpenET: average seasonal ET times crop coefficient minus effective precip and carryover
            # (ETo * ETof) - ep - co

            summary.loc[i, 'etos'] = et_by_year.mean()
            summary.loc[i, 'etbc'] = bc_pet
            summary.loc[i, 'etof'] = df['etof'].mean()
            summary.loc[i, 'opnt_cu'] = et_by_year.mean() * df['etof'].mean() - eff_precip - carryover
            summary.loc[i, 'dnrc_cu'] = bc_cu

        summary.to_sql(out, con, if_exists='append', index=False)
    else:
        print('Those fields already have consumptive use data.')
    print()


def create_shapefile(name, field_select, variables, geo_source):
    """ Do I need this?

    Take list of fields desired, list of variables describing which columns to include,
    and pull data from appropriate databases, then save the resulting dataframe to a shapefile.
    Should I append columns to a shapefile? Otherwise, where am I getting the geomtery from?"""
    pass


def plot_results():
    gridmets = pd.read_sql("SELECT DISTINCT gfid FROM gridmet_ts", con)
    print(len(gridmets))

    data = pd.read_sql("SELECT * FROM field_cu_results", con)
    data['county'] = data['fid'].str.slice(0, 3)

    plt.figure(figsize=(10, 5), dpi=200)

    plt.subplot(121)
    plt.title("Average Seasonal ET (in)")
    for i in data['county'].unique():
        plt.scatter(data[data['county'] == i]['etbc'], data[data['county'] == i]['etos'], zorder=5,
                    label="{} ({})".format(COUNTIES[i], i))
    plt.plot(data['etbc'], data['etbc'], 'k', zorder=4, label="1:1")
    plt.grid(zorder=3)
    plt.xlabel('DNRC')
    plt.ylabel('OpenET')
    plt.legend(title='County')

    plt.subplot(122)
    plt.title("Average Seasonal Consumptive Use (in)")
    for i in data['county'].unique():
        plt.scatter(data[data['county'] == i]['dnrc_cu'], data[data['county'] == i]['opnt_cu'], zorder=5,
                    label="{} ({})".format(COUNTIES[i], i))
    plt.plot(data['opnt_cu'], data['opnt_cu'], 'k', zorder=4, label="1:1")
    plt.grid(zorder=3)
    plt.xlabel('DNRC')
    plt.ylabel('OpenET')
    plt.legend(title='County')

    plt.tight_layout()

    plt.show()


def update_etof_db(etof_dir, etof_tb):
    # check by county, not fields...
    # Don't keep loading a ton of data if you're not going to do anything with it.

    print("Updating etof table with new county data")

    # list of counties that we have etof data for
    counties = []
    # Right now, file names should look like "ensemble_monthly_etof_000.csv"
    # We are trying to recover the "000" as the county code to check if these files need to be added.
    for filename in os.listdir(etof_dir):
        county = filename[-7:-4]
        counties.append(county)
    counties = pd.DataFrame(data=counties, columns=['county'])

    # list of counties in db already
    existing_counties = pd.read_sql("SELECT DISTINCT fid FROM {}".format(etof_tb), con)
    existing_counties['county'] = existing_counties['fid'].str.slice(0, 3)

    # remove counties in db
    counties = counties.loc[~counties['county'].isin(existing_counties['county'])]

    # if any are new,
    if len(counties) > 0:
        # load in each new csv file
        for i in counties['county']:
            path = os.path.join(etof_dir, "ensemble_monthly_etof_{}.csv".format(i))
            etof_data = pd.read_csv(path)
            etof_data.to_sql(etof_tb, con, if_exists='append', index=False)
        print("Done!")
    else:
        print('Those counties already have etof data.')
    print()


if __name__ == '__main__':
    # sqlite database (This should stay uncommented, almost all functions below rely on it.)
    # con = sqlite3.connect("C:/Users/CND571/Documents/Data/tutorial.db")
    con = sqlite3.connect("C:/Users/CND571/Documents/Data/opnt_analysis_02132024.db")

    # sqlite database table names
    gm_ts, fields_db, results, etof_db = 'gridmet_ts', 'field_data', 'field_cu_results', 'opnt_etof'

    # # Initialize tables with correct column names - Issue here with hard-coding?
    # # Before building database again, double-check what gridmet variables are actually needed.
    # # Only run once
    # cur = con.cursor()
    # # These don't work due to syntax for placeholders. Why?
    # # cur.execute("CREATE TABLE ?(date, year, month, day, centroid_lat, centroid_lon, elev_m, "
    # #             "tmin_c, tmax_c, srad_wm2, prcp_mm, etr_mm, eto_mm, etr_mm_uncorr, eto_mm_uncorr, gfid)", (gm_ts,))
    # # cur.execute("CREATE TABLE ?(fid, itype, usage, mapped_by, gfid, lat, lon, elev_gm)", (fields,))
    # # cur.execute("CREATE TABLE ?(fid, start, end, etos, etbc, etof, opnt_cu, dnrc_cu)", (results,))
    # # These do work.
    # cur.execute("CREATE TABLE {}(date, year, month, day, centroid_lat, centroid_lon, elev_m, tmin_c, tmax_c, "
    #             "srad_wm2, prcp_mm, etr_mm, eto_mm, etr_mm_uncorr, eto_mm_uncorr, gfid)".format(gm_ts))
    # cur.execute("CREATE TABLE {}(fid, itype, usage, mapped_by, county, gfid, lat, lon, elev_gm)".format(fields_db))
    # cur.execute("CREATE TABLE {}(fid, start, end, etos, etbc, etof, opnt_cu, dnrc_cu)".format(results))
    # cur.execute("CREATE TABLE {}(time, fid, etof, acres)".format(etof_db))

    # # need to fetch OpenET data for each county separately (and first...)
    # small_counties = ['019_Daniels', '033_Garfield', '051_Liberty', '061_Mineral', '101_Toole']
    # start_por = "2016-01-01"  # What is the period of record for OpenET etof?
    # end_por = "2022-12-31"
    # for i in small_counties:
    #     gee_asset = 'projects/ee-hehaugen/assets/{}'.format(i)
    #     openet_get_fields_link(gee_asset, start_por, end_por)
    #
    # After fetching etof csv files, load them into db
    # etof_loc = "C:/Users/CND571/Documents/Data/etof_files"
    # update_etof_db(etof_loc, etof_db)

    # gridmet information
    gm_d = 'C:/Users/CND571/Documents/Data/gridmet'  # location of general gridmet files
    gridmet_cent = os.path.join(gm_d, 'gridmet_centroids_MT.shp')
    rasters_ = os.path.join(gm_d, 'correction_surfaces_aea')  # correction surfaces, one for each month and variable.
    # Management factor and effective precip table files are called in functions.

    # Loading in state irrigation dataset (43k fields, takes a few seconds to load)
    mt_fields = gpd.read_file("C:/Users/CND571/Documents/Data/sid_30JAN2024_all.shp")
    mt_fields['county'] = mt_fields['fid'].str.slice(0, 3)
    county_count = mt_fields['county'].value_counts(ascending=True)
    # print(county_count)

    pos_start = '2016-01-01'
    pos_end = '2020-12-31'

    # Running analysis for 5 smallest counties
    # This will only update with new information. Each step checks for prior inclusion in db.
    for i in range(5):
        county_id = county_count.index[i]
        print("{} County ({})".format(COUNTIES[county_id], county_id))
        county_fields = mt_fields[mt_fields['county'] == county_id]
        # print(county_fields)
        gridmet_match_db(county_fields, gridmet_cent, fields_db)  # short
        corrected_gridmet_db(gridmet_cent, fields_db, gm_ts, rasters_)  # very long
        cu_analysis_db(fields_db, gm_ts, pos_start, pos_end, etof_db, results)  # short

    # plot_results()  # Only dependent on db tables existing

# ========================= EOF ====================================================================
