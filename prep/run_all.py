
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
from datetime import timedelta

from utils.thredds import GridMet
from iwr.iwr_approx import iwr_daily_fm

# Only used in corrected_gridmet functions and variations. When only one is active, put this inside so that it
# pulls these 5 variables, but have another dict with all available daily variables.
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
    'tmmx': {
        'nc': 'agg_met_tmmx_1979_CurrentYear_CONUS',
        'var': 'daily_maximum_temperature',
        'col': 'tmax_k'},
    'tmmn': {
        'nc': 'agg_met_tmmn_1979_CurrentYear_CONUS',
        'var': 'daily_minimum_temperature',
        'col': 'tmin_k'},
}

CLIMATE_COLS_LONG = {
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
    'vpd': {
        'nc': 'agg_met_vpd_1979_CurrentYear_CONUS',
        'var': 'daily_mean_vapor_pressure_deficit',
        'col': 'vpd_kpa'}
}

COLUMN_ORDER = ['date',
                'year',
                'month',
                'day',
                'centroid_lat',
                'centroid_lon',
                'elev_m',
                'tmin_c',
                'tmax_c',
                'prcp_mm',
                'etr_mm',
                'eto_mm',
                'etr_mm_uncorr',
                'eto_mm_uncorr']

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


def openet_get_fields_link(fields, start, end, et_too=False,
                           api_key='C:/Users/CND571/Documents/OpenET_API.txt'):
    """
    Use OpenET API timeseries multipolygon endpoint to get etof data given a Google Earth Engine asset.

    Prints the url, click link to download csv file; link lasts 5 minutes.
    :fields: path to gee asset, called 'Table ID' in gee,
    potential form of 'projects/cloud_project/assets/asset_filename'
    :start: beginning of period of study, string in 'YYYY-MM-DD' format
    :end: end of period of study, string in 'YYYY-MM-DD' format
    :et_too: optional, bool, if True, also get link for downloading OpenET ensemble ET
    :api_key: filepath to .txt file containing only the API key from user's OpenET account
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


def openet_get_fields_export(fields, start, end, et_too=False,
                             api_key='C:/Users/CND571/Documents/OpenET_API.txt'):
    """ Uses OpenET API timeseries multipolygon endpoint to get etof data given a Google Earth Engine asset.
    Prints the url, click link to download csv file; link lasts 5 minutes.
    Switch to export multipolygon? I might be able to automatically retrieve that.
    :fields: path to gee asset, form of 'projects/cloud_project/assets/asset_filename'
    :start: beginning of period of study, 'YYYY-MM-DD' format
    :end: end of period of study, 'YYYY-MM-DD' format
    :et_too: if True, also get link for downloading OpenET ensemble ET over same time period and set of fields
    :api_key: from user's OpenET account
    """

    # Use earth engine and gsutil to upload shapefiles to earth engine and then download resulting files from bucket.

    # This is apparently better, as it makes sure to close the file.
    with open(api_key, 'r') as f:
        api_key = f.readline()

    # key_file = open(api_key, "r")
    # api_key = key_file.readline()

    # set your API key before making the request
    header = {"Authorization": api_key}

    # # temporarily upload a GeoJSON file to use instead of a gee link
    # # limitations: RFC 7946 formatted (?), max 5mb, expires after 15 seconds.
    # # Yellowstone will be too big for this. (6.35mb, 3600 fields) How many fields can fit?
    # args = {
    #     'file': (fields, open(fields, 'rb'), 'application/geo+json')
    # }
    #
    # # query the api
    # resp = requests.get(
    #     headers=header,
    #     files=args,
    #     url="https://openet-api.org/account/upload"
    # )
    # print(resp)  # Currently giving a 400 Bad Request error.
    # # Why is this a bad request? It is exactly the format provided in the documentation...
    # print(resp.json())  # How to call asset_id from this?
    # response = resp.json()
    # temp_id = response['asset_id']
    # print(temp_id)
    # print()

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
        "units": "in",
        "drive_folder": "OpenET Exports"  # It does not like the bucket destination. Will this override it?
    }

    # query the api
    resp = requests.post(
        headers=header,
        json=args,
        url="https://openet-api.org/raster/export/multipolygon"
    )
    print(resp.json())
    response = resp.json()
    tag = response['name']
    print(tag)

    if et_too:
        # getting et variable too, in separate file.
        args.update({"variable": "ET"})
        resp = requests.post(
            headers=header,
            json=args,
            url="https://openet-api.org/raster/export/multipolygon"
        )
        print(resp.json())


def update_etof_db(con, etof_dir, etof_tb):
    """
    Update the sqlite database table that stores OpenET etof data.

    After checking which county codes (of form "000") are already contained in the db table, all the data from files
    in etof_dir with new county codes are appended to the existing table.
    :con: sqlite database connection
    :etof_dir: path to directory where etof files are located; etof files should be like "ensemble_monthly_etof_000.csv"
    :etof_tb: str, name of table in sqlite database containing etof data
    """
    # Do we want it like this? Will I be inputting larger batches?
    # What about having to split up counties into multiple sections?
    # This would be great to integrate into openet_get_fields_export, if I can get that working.

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


def gridmet_match_db(con, fields, gridmet_points, fields_join):
    """
    Match each input field centroid with closest gridmet pixel centroid and update sqlite database table.

    This depends on running 'Raster Pixels to Points' on a WGS Gridmet raster,
     attributing GFID, lat, and lon in the attribute table, and saving to project crs: 5071.
     GFID is an arbitrary identifier e.g., @row_number. It further depends on projecting the
     rasters to EPSG:5071, using the project.sh bash script (or gis)

     The reason we're not just doing a zonal stat on correction surface for every object is that
     there may be many fields that only need data from one gridmet cell. This prevents us from downloading
     many redundant data sets. Looks like it often works out to roughly 1 gridmet data pull for every 10 fields.

     :con: sqlite database connection
     :fields: filepath, shapefile containing field geometries and unique field ids with county identifier
     :gridmet_points: filepath, shapefile of gridmet pixel centroids
     :fields_join: str, name of table in sqlite database containing field/gridmet lookup
     """

    convert_to_wgs84 = lambda x, y: pyproj.Transformer.from_crs('EPSG:5071', 'EPSG:4326').transform(x, y)

    # fields = gpd.read_file(fields) # This is now already read in, and thus redundant
    print('Finding field-gridmet joins for {} fields'.format(fields.shape[0]))

    gridmet_pts = gpd.read_file(gridmet_points)

    existing_fields = pd.read_sql("SELECT DISTINCT fid FROM {}".format(fields_join), con)
    # If there are any new fields,
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


def corrected_gridmet_db(con, gridmet_points, fields_join, gridmet_tb, gridmet_ras,
                         start='1990-01-01', end='2023-12-31', selection=pd.DataFrame()):
    """
    Fetch new gridmet data, apply corrections, and update the sqlite database table of gridmet ts data.

    This depends on running 'Raster Pixels to Points' on a WGS Gridmet raster,
    attributing GFID, lat, and lon in the attribute table, and saving to project crs: 5071.
    GFID is an arbitrary identifier e.g., @row_number. It further depends on projecting the
    rasters to EPSG:5071, using the project.sh bash script

    The reason we're not just doing a zonal stat on correction surface for every object is that
    there may be many fields that only need data from one gridmet cell. This prevents us from downloading
    many redundant data sets.

    Date search function prevents creation of duplicate rows, and ensures continuous record from the earliest start
    date entered to the latest end date entered. No gaps in data are permitted. This may result in downloading "extra"
    data if periods of interest are not overlapping, but greatly simplifies the logic of the algorithm.

    This function does not permit the addition of more meteorological variables to existing rows. See
    more_gridmet_vars() for that functionality.

    :con: sqlite database connection
    :gridmet_points: filepath, shapefile of gridmet pixel centroids
    :fields_join: str, name of table in sqlite database containing field/gridmet lookup
    :gridmet_tb: str, name of table in sqlite database containing gridmet data
    :gridmet_ras: filepath, directory containing .tif files with gridmet correction factors, one for each month of year
    and each variable (etr and eto, 24 total files). Correction is to limit bias between gridmet and agrimet stations.
    :start: optional, beginning of period of study, string in 'YYYY-MM-DD' format
    :end: optional, end of period of study, string in 'YYYY-MM-DD' format
    :selection: optional, pandas dataframe containing a column of gfids for which to load data. If None, function will
    update for all gfids contained in the database's field id/gfid lookup table (fields_join).
    """
    if selection.empty:
        fields = pd.read_sql("SELECT DISTINCT gfid FROM {}".format(fields_join), con)
    else:
        fields = selection
    existing_fields = pd.read_sql("SELECT DISTINCT gfid FROM {}".format(gridmet_tb), con)

    s2 = pd.to_datetime(start)
    e2 = pd.to_datetime(end)

    #  determining what data is missing
    target_gfids = []
    for idx in fields['gfid']:
        if ~existing_fields['gfid'].isin([idx]).any():
            target_gfids.append([idx, start, end])
        else:
            existing_dates = pd.read_sql("SELECT DISTINCT date FROM {} WHERE gfid={}".format(gridmet_tb, idx), con)
            existing_dates = pd.to_datetime(existing_dates['date'])
            s1 = existing_dates.min()
            e1 = existing_dates.max()
            if s2 >= s1 and e2 <= e1:
                continue
            if s2 < s1:
                new_e = s1 - timedelta(days=1)
                new_e = new_e.strftime('%Y-%m-%d')
                target_gfids.append([idx, start, new_e])
            if e2 > e1:
                new_s = e1 + timedelta(days=1)
                new_s = new_s.strftime('%Y-%m-%d')
                target_gfids.append([idx, new_s, end])

    # Makes up to 2 entries for each gridmet point, if requested time range is longer than existing data record
    target_gfids = pd.DataFrame(target_gfids, columns=['gfid', 'start', 'end'])

    # If there is any new data to fetch,
    if not target_gfids.empty:
        # run data fetching algorithm
        gridmet_pts = gpd.read_file(gridmet_points)
        gridmet_pts.index = gridmet_pts['GFID']

        # Loading correction surfaces
        rasters = []
        for v in ['eto', 'etr']:
            [rasters.append(os.path.join(gridmet_ras, 'gridmet_corrected_{}_{}.tif'.format(v, m))) for m in range(1, 13)]

        # Getting correction factors for the required gridmet locations
        gridmet_targets = {}
        for i in target_gfids['gfid'].unique():
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

        # Getting the gridmet data
        for index, row in tqdm(target_gfids.iterrows(), total=len(target_gfids)):
            start = row['start']
            end = row['end']
            df, first = pd.DataFrame(), True
            r = gridmet_pts.loc[row['gfid']]
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

            corr_val = gridmet_targets[row['gfid']]
            for _var in ['etr', 'eto']:
                variable = '{}_mm'.format(_var)
                for month in range(1, 13):
                    corr_factor = corr_val[str(month)][_var]
                    idx = [i for i in df.index if i.month == month]
                    df.loc[idx, '{}_uncorr'.format(variable)] = df.loc[idx, variable]
                    df.loc[idx, variable] = df.loc[idx, '{}_uncorr'.format(variable)] * corr_factor

            df['tmax_c'] = df.tmax_k - 273.15
            df['tmin_c'] = df.tmin_k - 273.15

            df = df[COLUMN_ORDER]  # I don't think order actually matters, as long as names match ones in db
            df['gfid'] = row['gfid']
            df.to_sql(gridmet_tb, con, if_exists='append', index=False)
        print()
    else:
        print("These gridmet data are already in the database")
        print()


def more_gridmet_vars(con, variables, gridmet_points, fields_join, gridmet_tb, gridmet_ras, start, end,
                      selection=pd.DataFrame()):
    """ Update gridmet table with new columns of meteorological variables.

    This should primarily be used when data at the provided locations and in the provided time ranges already exists
    in the database table (no new rows are to be added). However, there is functionality to read in new data to
    prevent the function from breaking. This is clunky.

    Data for variables tmin_c, tmac_c, prcp_mm, etr_mm, eto_mm, etr_mm_uncorr, and eto_mm_uncorr are required to be
    present. They are either already loaded, or will be before additional variables are added.

    :con: sqlite database connection
    :variables: list of variables to be added to the table, each one needs to be identical to a column in gridmet_tb,
    and a 'col' value in CLIMATE_COLS_LONG
    :gridmet_points: filepath, shapefile of gridmet pixel centroids
    :fields_join: str, name of table in sqlite database containing field/gridmet lookup
    :gridmet_tb: str, name of table in sqlite database containing gridmet data
    :gridmet_ras: filepath, directory containing .tif files with gridmet correction factors, one for each month of year
    and each variable (etr and eto, 24 total files). Correction is to limit bias between gridmet and agrimet stations.
    :start: optional, beginning of period of study, string in 'YYYY-MM-DD' format
    :end: optional, end of period of study, string in 'YYYY-MM-DD' format
    :selection: optional, pandas dataframe containing column of gfids for which to load data. If None, function will
    update for all gfids contained in the database's field id/gfid lookup table (fields_join).
    """
    print("Loading additional meteorological data")
    if selection.empty:
        fields = pd.read_sql("SELECT DISTINCT gfid FROM {}".format(fields_join), con)
    else:
        fields = selection
    existing_fields = pd.read_sql("SELECT DISTINCT gfid FROM {}".format(gridmet_tb), con)

    fields_new = fields.loc[~fields['gfid'].isin(existing_fields['gfid'])]

    if not fields_new.empty:
        print("new gfid(s), appending rows first:")
        corrected_gridmet_db(con, gridmet_points, fields_join, gridmet_tb, gridmet_ras,
                               start=start, end=end, selection=fields_new)

    s2 = pd.to_datetime(start)
    e2 = pd.to_datetime(end)

    gridmet_pts = gpd.read_file(gridmet_points)
    gridmet_pts.index = gridmet_pts['GFID']

    cursor = con.cursor()
    for idx in tqdm(fields['gfid'], total=len(fields)):
        # recover start and end times of data
        existing_dates = pd.read_sql("SELECT DISTINCT date FROM {} WHERE gfid={}".format(gridmet_tb, idx), con)
        existing_dates = pd.to_datetime(existing_dates['date'])
        s1 = existing_dates.min()
        e1 = existing_dates.max()
        if s2 < s1 or e2 > e1:
            print("new dates, appending rows first:")
            corrected_gridmet_db(con, gridmet_points, fields_join, gridmet_tb, gridmet_ras,
                                   start=start, end=end, selection=pd.DataFrame([idx], columns=['gfid']))
        # Determine which columns have null values. Any columns with any null values in range will be fully overwritten.
        # This is loading a lot of data for a simple question. I don't understand how to do it in sql though.

        # Investigating alternate ways to determine which rows have null values.

        # nulls3 = pd.read_sql("select count(*) from (select top 1 'There is at least one NULL' AS note from TestTable "
        #                      "where Column_3 is NULL) a")
        # print(nulls3)  # not ready yet

        # nulls2 = pd.read_sql("select sum(case when q_kgkg is null then 1 else 0 end) as q_kgkg, "
        #                      "sum(case when srad_wm2 is null then 1 else 0 end) as srad_wm2, "
        #                      "sum(case when u10_ms is null then 1 else 0 end) as u10_ms,"
        #                      "sum(case when wdir_deg is null then 1 else 0 end) as wdir_deg,"
        #                      "sum(case when vpd_kpa is null then 1 else 0 end) as vpd_kpa from {}"
        #                      .format(gridmet_tb), con)
        # print(nulls2)  # easiest way to get column names where entry is greater than zero?
        # # print(nulls2[nulls2[]])
        #
        # # Why is this one not working? Looks like the most elegant solution...
        # nulls1 = pd.read_sql("SELECT * FROM (SELECT q_kgkg, srad_wm2, u10_ms, wdir_deg, vpd_kpa FROM {} WHERE NULL "
        #                      "IN (q_kgkg, srad_wm2, u10_ms, wdir_deg, vpd_kpa)) AS RESULT".format(gridmet_tb), con)
        # print(nulls1)
        # null_cols1 = set(nulls1.columns)

        nulls = pd.read_sql("SELECT q_kgkg, srad_wm2, u10_ms, wdir_deg, vpd_kpa FROM {} WHERE gfid={} AND date(date) "
                            "BETWEEN date('{}') AND date('{}')".format(gridmet_tb, idx, start, end), con)
        null_cols = nulls.columns[nulls.isnull().any()]
        null_cols = set(null_cols)
        variables = set(variables)
        target_cols = variables.intersection(null_cols)

        if target_cols:
            r = gridmet_pts.loc[idx]
            df = pd.DataFrame()
            for thredds_var, cols in CLIMATE_COLS_LONG.items():
                variable = cols['col']
                if variable in target_cols:
                    lat, lon = r['lat'], r['lon']
                    g = GridMet(thredds_var, start=start, end=end, lat=lat, lon=lon)
                    s = g.get_point_timeseries()
                    # print(s)
                    df[variable] = s[thredds_var]
            # build update query based on new columns
            sql_query = ("UPDATE {} SET ".format(gridmet_tb))
            for i in df.columns:
                sql_query += "{}=?, ".format(i)
            sql_query = sql_query[:-2]  # remove last comma
            sql_query += " WHERE gfid={} AND date=?".format(idx)
            # print(sql_query)

            df['date'] = [i.strftime('%Y-%m-%d') for i in df.index]

            data = df.to_records(index=False).tolist()
            cursor.executemany(sql_query, data)
            con.commit()
    cursor.close()


def cu_analysis_db(con, shp, gridmet, start, end, etof, out):
    """
    Calculate average seasonal consumptive use with both IWR/DNRC and OpenET methods.

    :con: sqlite database connection
    :shp: str, name of table in sqlite database associated with field/gridmet lookup
    :gridmet: str, name of table in sqlite database associated with gridmet data
    :start: str in 'YYYY-MM-DD' format, beginning of period of study
    :end: str in 'YYYY-MM-DD' format, end of period of study
    :etof: str, name of table in sqlite database containing etof data
    :out: str, name of table in sqlite database containing results of consumptive use analysis by field
    """

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

            # Explanation of consumptive use calculations
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


def plot_results(con, met_too=False):
    """Create figure comparing ET and consumptive use from two different methods."""
    gridmets = pd.read_sql("SELECT DISTINCT gfid FROM gridmet_ts", con)
    print(len(gridmets))

    data = pd.read_sql("SELECT * FROM field_cu_results", con)
    data['county'] = data['fid'].str.slice(0, 3)

    # ET comparison
    plt.figure(figsize=(10, 5), dpi=200)

    plt.subplot(121)
    plt.title("Average Seasonal ET (in)")
    for i in data['county'].unique():  # Why did it plot in a different order than last time?
        plt.scatter(data[data['county'] == i]['etbc'], data[data['county'] == i]['etos'], zorder=5,
                    label="{} ({})".format(COUNTIES[i], i))
    plt.plot(data['etbc'], data['etbc'], 'k', zorder=4, label="1:1")
    plt.grid(zorder=3)
    plt.xlabel('DNRC')
    plt.ylabel('Gridmet ETo (grass reference)')
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

    # Looking at other meteorological variables
    if met_too:
        other_climate = pd.DataFrame(columns=['q_kgkg', 'u10_ms', 'srad_wm2', 't'], index=gridmets['gfid'])
        for i in gridmets['gfid']:
            # print("i", i)
            grd = pd.read_sql("SELECT date, q_kgkg, u10_ms, srad_wm2, tmax_c, tmin_c FROM gridmet_ts WHERE gfid={}".format(i), con)
            # grd = pd.read_sql("SELECT date, eto_mm, tmin_c, tmax_c, prcp_mm FROM ? WHERE gfid=? AND date(date) "
            #                   "BETWEEN date('?') AND date('?')", con, params=(gridmet, row['gfid'], start, end))
            # rename variables needed in blaney_criddle calculations
            # grd = grd.rename(columns={'eto_mm': 'ETOS', 'tmin_c': 'MN', 'tmax_c': 'MX', 'prcp_mm': 'PP'})
            grd.index = pd.to_datetime(grd['date'])

            grd['tmean_c'] = (grd['tmax_c'] + grd['tmin_c'])/2

            # Masking daily gridMET data to growing season
            grd['mday'] = ['{}-{}'.format(x.month, x.day) for x in grd.index]
            target_range = pd.date_range('2000-04-01', '2000-09-30')
            accept = ['{}-{}'.format(x.month, x.day) for x in target_range]
            grd['mask'] = [1 if d in accept else 0 for d in grd['mday']]
            grd = grd[grd['mask'] == 1]

            other_climate.at[i, 'q_kgkg'] = grd['q_kgkg'].mean()
            other_climate.at[i, 'u10_ms'] = grd['u10_ms'].mean()
            other_climate.at[i, 'srad_wm2'] = grd['srad_wm2'].mean()
            other_climate.at[i, 't'] = grd['tmean_c'].mean()

        # print(other_climate)

        data['gfid'] = pd.read_sql("SELECT gfid FROM field_data", con)
        # print(lookup)

        data['q_kgkg'] = [other_climate['q_kgkg'].loc[i] for i in data['gfid']]
        data['u10_ms'] = [other_climate['u10_ms'].loc[i] for i in data['gfid']]
        data['srad_wm2'] = [other_climate['srad_wm2'].loc[i] for i in data['gfid']]
        data['t'] = [other_climate['t'].loc[i] for i in data['gfid']]

        # print(data)

        # Investigating reasons for bias
        plt.figure(figsize=(8, 8), dpi=150)

        plt.subplot(221)
        # plt.title("Average Seasonal ET (in)")
        for i in data['county'].unique():  # Why did it plot in a different order than last time?
            plt.scatter(data[data['county'] == i]['q_kgkg'], data[data['county'] == i]['etos'] - data[data['county'] == i]['etbc'], zorder=5,
                        label="{} ({})".format(COUNTIES[i], i))
        # plt.plot(data['etbc'], data['etbc'], 'k', zorder=4, label="1:1")
        plt.grid(zorder=3)
        plt.xlabel('seasonal average daily humidity (kg/kg)')
        plt.ylabel('Bias in ET (OpenET - DNRC)')
        plt.legend(title='County')

        plt.subplot(222)
        # plt.title("Average Seasonal ET (in)")
        for i in data['county'].unique():  # Why did it plot in a different order than last time?
            plt.scatter(data[data['county'] == i]['u10_ms'],
                        data[data['county'] == i]['etos'] - data[data['county'] == i]['etbc'], zorder=5,
                        label="{} ({})".format(COUNTIES[i], i))
        # plt.plot(data['etbc'], data['etbc'], 'k', zorder=4, label="1:1")
        plt.grid(zorder=3)
        plt.xlabel('seasonal average daily wind speed (m/s)')
        plt.ylabel('Bias in ET (OpenET - DNRC)')
        plt.legend(title='County')

        plt.subplot(223)
        # plt.title("Average Seasonal ET (in)")
        for i in data['county'].unique():  # Why did it plot in a different order than last time?
            plt.scatter(data[data['county'] == i]['srad_wm2'],
                        data[data['county'] == i]['etos'] - data[data['county'] == i]['etbc'], zorder=5,
                        label="{} ({})".format(COUNTIES[i], i))
        # plt.plot(data['etbc'], data['etbc'], 'k', zorder=4, label="1:1")
        plt.grid(zorder=3)
        plt.xlabel('seasonal average daily surface radiation (w/m^2)')
        plt.ylabel('Bias in ET (OpenET - DNRC)')
        plt.legend(title='County')

        plt.subplot(224)
        # plt.title("Average Seasonal ET (in)")
        for i in data['county'].unique():  # Why did it plot in a different order than last time?
            plt.scatter(data[data['county'] == i]['t'],
                        data[data['county'] == i]['etos'] - data[data['county'] == i]['etbc'], zorder=5,
                        label="{} ({})".format(COUNTIES[i], i))
        # plt.plot(data['etbc'], data['etbc'], 'k', zorder=4, label="1:1")
        plt.grid(zorder=3)
        plt.xlabel('seasonal average daily temperature (C)')
        plt.ylabel('Bias in ET (OpenET - DNRC)')
        plt.legend(title='County')

        plt.tight_layout()

    plt.show()


def create_shapefile(name, field_select, variables, geo_source):
    """ Do I need this?

    Take list of fields desired, list of variables describing which columns to include,
    and pull data from appropriate databases, then save the resulting dataframe to a shapefile.
    Should I append columns to a shapefile? Otherwise, where am I getting the geomtery from?"""
    pass


if __name__ == '__main__':
    # sqlite database connection
    conec = sqlite3.connect("C:/Users/CND571/Documents/Data/tutorial_gm.db")
    # conec = sqlite3.connect("C:/Users/CND571/Documents/Data/opnt_analysis_02232024.db")  # has additional met data
    # conec = sqlite3.connect("C:/Users/CND571/Documents/Data/opnt_analysis_02132024.db")  # has 6 counties

    # sqlite database table names
    gm_ts, fields_db, results, etof_db = 'gridmet_ts', 'field_data', 'field_cu_results', 'opnt_etof'

    # # Initialize tables with correct column names
    # # Only run once
    # cur = conec.cursor()
    # cur.execute("CREATE TABLE {}(gfid, date, year, month, day, centroid_lat, centroid_lon, elev_m, tmin_c, tmax_c, "
    #             "prcp_mm, etr_mm, eto_mm, etr_mm_uncorr, eto_mm_uncorr, q_kgkg, srad_wm2, u10_ms, wdir_deg, vpd_kpa)"
    #             .format(gm_ts))
    # cur.execute("CREATE TABLE {}(fid, itype, usage, mapped_by, county, gfid, lat, lon, elev_gm)".format(fields_db))
    # cur.execute("CREATE TABLE {}(fid, start, end, etos, etbc, etof, opnt_cu, dnrc_cu)".format(results))
    # cur.execute("CREATE TABLE {}(time, fid, etof, acres)".format(etof_db))
    # cur.close()

    # # Testing mulitpolygon export endpoint (again)
    # # None of these alternative paths to the asset are working, all give response 400.
    # # asset = 'C:/Users/CND571/Documents/Data/091_sheridan.geojson'
    # # asset = '091_sheridan.geojson'
    # asset = 'C:/Users/CND571/PycharmProjects/openet/prep/091_sheridan.geojson'
    # gee_asset = 'projects/ee-hehaugen/assets/091_Sheridan'
    # start_por = "2016-01-01"  # What is the period of record for OpenET etof?
    # end_por = "2022-12-31"
    # openet_get_fields_export(gee_asset, start_por, end_por)

    # # need to fetch OpenET data for each county separately (and first...)
    # small_counties = ['019_Daniels', '033_Garfield', '051_Liberty', '061_Mineral', '101_Toole']
    # # small_counties = ['015_Chouteau', '055_McCone', '075_Powder_River', '093_Silver_Bow']
    # start_por = "2016-01-01"  # What is the period of record for OpenET etof?
    # end_por = "2022-12-31"
    # for i in small_counties:
    #     gee_asset = 'projects/ee-hehaugen/assets/{}'.format(i)
    #     openet_get_fields_link(gee_asset, start_por, end_por)

    # # After fetching etof csv files, load them into db
    # etof_loc = "C:/Users/CND571/Documents/Data/etof_files"
    # update_etof_db(conec, etof_loc, etof_db)

    # # gridmet information
    # gm_d = 'C:/Users/CND571/Documents/Data/gridmet'  # location of general gridmet files
    # gridmet_cent = os.path.join(gm_d, 'gridmet_centroids_MT.shp')
    # rasters_ = os.path.join(gm_d, 'correction_surfaces_aea')  # correction surfaces, one for each month and variable.
    # # Management factor and effective precip table files are called in functions.
    #
    # # Loading in state irrigation dataset (43k fields, takes a few seconds to load)
    # mt_fields = gpd.read_file("C:/Users/CND571/Documents/Data/sid_30JAN2024_all.shp")
    # mt_fields['county'] = mt_fields['fid'].str.slice(0, 3)
    # county_count = mt_fields['county'].value_counts(ascending=True)
    # # print(county_count[:10])
    #
    # pos_start = '2016-01-01'
    # pos_end = '2022-12-31'
    #
    # # Running analysis for 6 smallest counties
    # # This will only update with new information. Each step checks for prior inclusion in db.
    # # i = 0
    # # for i in range(5):  # maybe this is better, not doing
    #
    # # counties = pd.read_sql("SELECT DISTINCT county FROM {}".format(fields_db), conec)
    #
    # for i in tqdm(range(5), total=5):
    #     county_id = county_count.index[i]
    #     print()
    #     print("{} County ({})".format(COUNTIES[county_id], county_id))
    #     county_fields = mt_fields[mt_fields['county'] == county_id]
    #     # print(county_fields)
    #     gridmet_match_db(conec, county_fields, gridmet_cent, fields_db)  # short
    #     corrected_gridmet_db(conec, gridmet_cent, fields_db, gm_ts, rasters_, pos_start, pos_end)  # very long
    #     cu_analysis_db(conec, fields_db, gm_ts, pos_start, pos_end, etof_db, results)  # short
    #     # county_gfids = pd.read_sql("SELECT DISTINCT gfid FROM {} WHERE county='{}'".format(fields_db, county_id), conec)
    #     # more_gridmet_vars(conec, ['u10_ms'], gridmet_cent, fields_db, gm_ts, rasters_,
    #     #                   pos_start, pos_end, selection=county_gfids)  # very long, should always use selection
    #     # more_gridmet_vars(conec, ['q_kgkg', 'u10_ms', 'srad_wm2'], gridmet_cent, fields_db, gm_ts, rasters_,
    #     #                   pos_start, pos_end, selection=county_gfids)  # very long, should always use selection

    # plot_results(conec)  # Only dependent on db tables existing

    # Testing gridmet loading functions/ selective update functionality.
    # pos_start = '2022-02-10'
    # pos_end = '2022-02-20'
    # pos_start = '2022-02-01'
    # pos_end = '2022-02-28'
    # corrected_gridmet_db(conec, gridmet_cent, fields_db, gm_ts, rasters_, pos_start, pos_end)
    # more_gridmet_vars(conec, ['q_kgkg', 'u10_ms', 'srad_wm2'], gridmet_cent, fields_db, gm_ts, rasters_,
    #                   pos_start, pos_end)

    # Save and close database connection (Is this necessary? Everything appeared to be working fine without it.)
    conec.commit()
    conec.close()

# ========================= EOF ====================================================================
