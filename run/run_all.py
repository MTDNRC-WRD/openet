
import os

import pandas as pd
import geopandas as gpd
import numpy as np
import pyproj
from rasterstats import zonal_stats
from tqdm import tqdm
import sqlite3
import matplotlib.pyplot as plt
from datetime import timedelta

from utils.thredds import GridMet
from iwr.iwr_approx import iwr_daily_fm, iwr_database

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

# '011', '025', and '109' have no fields in the 15FEB24 Statewide Irrigation Dataset.
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

# From 'the rule', 3 time periods: 1964–1973, 1973–2006, 1997–2006
# 109 (Wibaux County) is a special case, where you are asked to use nearby data.
# Current SID does not include any fields in Wibaux County.
MANAGEMENT_FACTORS = {'001': [0.64, 0.83, 0.88], '003': [0.55, 0.79, 0.88], '005': [0.59, 0.64, 0.66],
                      '007': [0.69, 0.80, 0.87], '009': [0.58, 0.67, 0.71], '011': [0.38, 0.55, 0.54],
                      '013': [0.57, 0.70, 0.79], '015': [0.53, 0.65, 0.78], '017': [0.55, 0.72, 0.81],
                      '019': [0.47, 0.65, 0.75], '021': [0.57, 0.64, 0.72], '023': [0.78, 0.90, 1.00],
                      '025': [0.48, 0.48, 0.48], '027': [0.49, 0.66, 0.68], '029': [0.88, 0.95, 0.97],
                      '031': [0.74, 0.92, 0.99], '033': [0.43, 0.51, 0.46], '035': [0.60, 0.74, 0.74],
                      '037': [0.63, 0.66, 0.65], '039': [0.87, 0.87, 0.97], '041': [0.54, 0.60, 0.60],
                      '043': [0.61, 0.78, 0.81], '045': [0.49, 0.68, 0.69], '047': [0.55, 0.69, 0.69],
                      '049': [0.60, 0.79, 0.80], '051': [0.55, 0.66, 0.64], '053': [0.47, 0.56, 0.59],
                      '055': [0.44, 0.55, 0.61], '057': [0.65, 0.79, 0.83], '059': [0.57, 0.70, 0.78],
                      '061': [0.56, 0.63, 0.64], '063': [0.70, 0.68, 0.69], '065': [0.50, 0.59, 0.56],
                      '067': [0.57, 0.66, 0.68], '069': [0.44, 0.50, 0.43], '071': [0.55, 0.55, 0.55],
                      '073': [0.71, 0.81, 0.84], '075': [0.39, 0.49, 0.53], '077': [0.78, 0.90, 1.00],
                      '079': [0.60, 0.74, 0.84], '081': [0.80, 0.89, 0.96], '083': [0.56, 0.73, 0.88],
                      '085': [0.47, 0.65, 0.75], '087': [0.48, 0.68, 0.73], '089': [0.59, 0.69, 0.63],
                      '091': [0.45, 0.69, 0.81], '093': [0.69, 0.90, 0.94], '095': [0.47, 0.63, 0.73],
                      '097': [0.45, 0.54, 0.49], '099': [0.69, 0.80, 0.88], '101': [0.52, 0.67, 0.71],
                      '103': [0.53, 0.75, 0.92], '105': [0.58, 0.67, 0.75], '107': [0.47, 0.59, 0.54],
                      '109': [0, 0, 0], '111': [0.60, 0.71, 0.78]}


def init_db_tables(con):
    cur = con.cursor()

    # # Old table formats
    # cur.execute("CREATE TABLE IF NOT EXISTS {}(gfid, date, year, month, day, centroid_lat, centroid_lon, elev_m, "
    #             "tmin_c, tmax_c, prcp_mm, etr_mm, eto_mm, etr_mm_uncorr, eto_mm_uncorr, q_kgkg, srad_wm2, u10_ms, "
    #             "wdir_deg, vpd_kpa)".format(gm_ts))
    # cur.execute("CREATE TABLE IF NOT EXISTS {}(fid, itype, usage, mapped_by, county, gfid, lat, lon, elev_gm)"
    #             .format(fields_db))
    # cur.execute("CREATE TABLE IF NOT EXISTS {}(fid, start, end, etos, etbc, etof, opnt_cu, dnrc_cu)".format(results))
    # cur.execute("CREATE TABLE IF NOT EXISTS {}(time, fid, etof, acres)".format(etof_db))

    cur.execute("""
                CREATE TABLE IF NOT EXISTS opnt_etof
                (time DATE NOT NULL,
                fid TEXT NOT NULL,
                etof REAL,
                acres REAL,
                PRIMARY KEY (fid, time) ON CONFLICT IGNORE
                );
                """)  # Not dependent on other tables, need to download right data at beginning
    # How to update the conflict clauses when making tables? I want things to ignore, not fail.
    cur.execute("""
                CREATE TABLE IF NOT EXISTS field_data
                (fid TEXT PRIMARY KEY ON CONFLICT IGNORE,
                itype TEXT,
                usage INTEGER,
                mapped_by TEXT,
                county TEXT,
                gfid INTEGER,
                lat REAL,
                lon REAL,
                elev_gm REAL);
                """)  # Not dependent on other tables, need to download right data, this controls others
    cur.execute("""
                CREATE TABLE IF NOT EXISTS gridmet_ts
                (gfid INTEGER NOT NULL,
                date DATE NOT NULL,
                year INTEGER,
                month INTEGER,
                day INTEGER,
                centroid_lat REAL,
                centroid_lon REAL,
                elev_m REAL,
                tmin_c REAL,
                tmax_c REAL,
                prcp_mm REAL,
                etr_mm REAL,
                eto_mm REAL,
                etr_mm_uncorr REAL,
                eto_mm_uncorr REAL,
                q_kgkg REAL,
                srad_wm2 REAL,
                u10_ms REAL,
                wdir_deg REAL,
                vpd_kpa REAL,
                PRIMARY KEY (gfid, date) ON CONFLICT IGNORE
                );
                """)  # Check gfids vs field_data, also need to do date checks within this table.
    cur.execute("""
                CREATE TABLE IF NOT EXISTS field_cu_results
                (fid TEXT NOT NULL,
                year DATE NOT NULL,
                irrmapper INTEGER NOT NULL,
                frac_irr REAL,
                mf_periods TEXT NOT NULL,
                mfs REAL,
                etos REAL,
                etbc REAL,
                etof REAL,
                opnt_cu REAL, 
                dnrc_cu REAL,
                PRIMARY KEY (fid, year, irrmapper, mf_periods) ON CONFLICT IGNORE
                );
                """)  # Check fid against field_data, check years against available gridmet?
    cur.execute("""
                CREATE TABLE IF NOT EXISTS static_iwr_results
                (fid TEXT NOT NULL,
                frac_irr REAL,
                mf_periods TEXT,
                mfs REAL,
                etos REAL,
                etbc REAL,
                dnrc_cu REAL,
                PRIMARY KEY (fid, mf_periods) ON CONFLICT IGNORE
                );
                """)  # Are there other fields I need/ones to delete? Is mf_periods or mfs a better index?
    cur.execute("""
                CREATE TABLE IF NOT EXISTS irrmapper
                (fid TEXT NOT NULL,
                year DATE NOT NULL,
                frac_irr REAL,
                PRIMARY KEY (fid, year) ON CONFLICT IGNORE
                );
                """)  # Not dependent on other tables, download correct data to begin

    # This is to help with the old gridmet formatting, since I will not take the time to reload those tables.
    cur.execute("CREATE INDEX IF NOT EXISTS idx1 ON gridmet_ts(gfid, date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx2 ON field_data(fid)")  # does this do much?

    cur.close()


def update_irrmapper_table(con, file):
    """ Load local irrmapper csv file (from gee) to db table. """
    cur = con.cursor()
    if len(pd.read_sql("SELECT DISTINCT fid FROM irrmapper", con)) == 0:
        irrmapper_ref = pd.read_csv(file, index_col='FID')
        irrmapper_ref = irrmapper_ref.drop(columns=['system:index', 'COUNTYNAME', 'COUNTY_NO', 'ITYPE', 'MAPPEDBY',
                                                    'SOURCECODE', 'USAGE', '.geo'])
        data = []
        for i in irrmapper_ref.index:
            for j in irrmapper_ref.columns:
                data.append((i, j, irrmapper_ref.at[i, j]))
        cur.executemany("INSERT INTO irrmapper VALUES(?, ?, ?)", data)
    cur.close()


# Decide which update_etof to keep.
def update_etof_db(con, etof_dir, etof_tb):
    """
    Update the sqlite database table that stores OpenET etof data.

    After checking which FIPS county codes (of form "000") are already contained in the db table, all the data from
    files in etof_dir with new county codes are appended to the existing table.
    Params:
    :con: sqlite database connection
    :etof_dir: path to directory where etof files are located; etof files should be like "ensemble_monthly_etof_000.csv"
    :etof_tb: str, name of table in sqlite database containing etof data
    """
    # Do we want it like this? Will I be inputting larger batches?
    # What about having to split up counties into multiple sections? That's taken care of, I think.
    # This would be great to integrate into openet_get_fields_export, if I can get that working. Don't do that.

    print("Updating etof table with new county data")

    # list of counties that we have etof data for
    counties = []
    # Right now, file names should look like "ensemble_etof_000_XXXX_XXXX.csv"
    # We are trying to recover the "000" as the county code to check if these files need to be added.
    for filename in os.listdir(etof_dir):
        # county = filename[-7:-4]  # old, no years at end.
        county = filename[-17:-14]
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


def update_etof_db_1(con, etof_dir, etof_tb):
    """
    Update the sqlite database table that stores OpenET etof data.

    This one just loads in all the csv files, but the new primary keys in the db table should prevent
    any duplicates from being loaded. Am I wasting a lot of time not doing more checks beforehand?
    Well, this should probably only be run once, as I won't be downloading files over and over.
    Params:
    :con: sqlite database connection
    :etof_dir: path to directory where etof files are located; etof files should be like "ensemble_monthly_etof_000.csv"
    :etof_tb: str, name of table in sqlite database containing etof data
    """
    print("Updating etof table with new county data")

    # load in each csv file
    rows = 0
    # for i in os.listdir(etof_dir):
    for i in tqdm(os.listdir(etof_dir), total=len(os.listdir(etof_dir))):
        path = os.path.join(etof_dir, i)
        etof_data = pd.read_csv(path)
        # rows += etof_data.to_sql(etof_tb, con, if_exists='append', index=False, method=ignore_on_conflict)
        rows += etof_data.to_sql(etof_tb, con, if_exists='append', index=False)
    # print("Done! {} rows updated".format(rows))
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
    if ~existing_fields['fid'].isin(fields['FID']).any().any():
        # Remove potential duplicate fields
        fields = fields.loc[~fields['FID'].isin(existing_fields['fid'])]
        fields['gfid'] = np.nan
        # and run matching algorithm
        # for i, field in tqdm(fields.iterrows(), total=fields.shape[0]):
        for i, field in fields.iterrows():
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
        # fields = fields.drop(labels='geometry', axis='columns')
        fields = fields.drop(labels=['geometry', 'SOURCECODE', 'COUNTY_NO', 'COUNTYNAME'], axis='columns')
        fields = fields.rename(columns={'MAPPEDBY': 'mapped_by'})
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
    print("Looking for missing data...")
    target_gfids = []
    for idx in fields['gfid'].unique():  # makes sure repeated gfids in selection don't create redundant records.
        if ~existing_fields['gfid'].isin([idx]).any():
            target_gfids.append([idx, start, end])
        else:
            # This can take a long time, because it's loading and checking the dates for each point individually.
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

    # print(target_gfids)

    # If there is any new data to fetch,
    if not target_gfids.empty:
        # run data fetching algorithm
        gridmet_pts = gpd.read_file(gridmet_points)
        gridmet_pts.index = gridmet_pts['GFID']

        # Loading correction surfaces
        print('Loading correction factors...')
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


# Update inclusion checking here!
def corrected_gridmet_db_1(con, gridmet_points, fields_join, gridmet_tb, gridmet_ras,
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
    # Looking for missing gfid/date combos.
    print("Looking for missing data...")
    cur = con.cursor()

    s2 = pd.to_datetime(start)
    e2 = pd.to_datetime(end)

    target_gfids = []
    new_gfids = cur.execute("SELECT gfid FROM {} EXCEPT SELECT gfid FROM {}".format(fields_join, gridmet_tb))
    for idx in new_gfids:
        target_gfids.append((idx[0], start, end))

    old_gfids = cur.execute("SELECT gfid FROM {} INTERSECT SELECT gfid FROM {}".format(fields_join, gridmet_tb))
    for idx in old_gfids:
        s1 = pd.to_datetime(cur.execute("SELECT min(date) FROM {} WHERE gfid={}".format(gridmet_tb, idx[0])).fetchone())
        e1 = pd.to_datetime(cur.execute("SELECT max(date) FROM {} WHERE gfid={}".format(gridmet_tb, idx[0])).fetchone())
        if s2 >= s1 and e2 <= e1:
            continue
        if s2 < s1:
            new_e = s1 - timedelta(days=1)
            new_e = new_e.strftime('%Y-%m-%d')
            target_gfids.append((idx[0], start, new_e))
        if e2 > e1:
            new_s = e1 + timedelta(days=1)
            new_s = new_s.strftime('%Y-%m-%d')
            target_gfids.append((idx[0], new_s, end))

    target_gfids = pd.DataFrame(target_gfids, columns=['gfid', 'start', 'end'])

    # If there is any new data to fetch,
    if len(target_gfids) > 0:
        # run data fetching algorithm
        gridmet_pts = gpd.read_file(gridmet_points)
        gridmet_pts.index = gridmet_pts['GFID']

        # Loading correction surfaces
        print('Loading correction factors...')
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

    cur = con.cursor()
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
            cur.executemany(sql_query, data)
            con.commit()
    cur.close()


# Update this one w/ checking.
def iwr_static_cu_analysis_db(con, shp, gridmet, start, end, etof, out, irrmapper=False):
    """ This only needs to be done once (twice for irrmapper?) whereas the other IWR calcs need
    to be done once for each year. So it's its own function."""
    print("Calculating consumptive use for fields")

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    # Want to make look for all new fields not calculated over the given time frame.
    # fields = pd.read_sql("SELECT DISTINCT * FROM {}".format(shp), con, index_col='fid')
    # existing_fields = pd.read_sql("SELECT DISTINCT fid, start, end FROM {}".format(out), con, index_col='fid')
    fields = pd.read_sql("SELECT DISTINCT * FROM {}".format(shp), con)
    existing_fields = pd.read_sql("SELECT DISTINCT fid, year FROM {}".format(out), con)
    # fields['start'] = pd.to_datetime(start)
    # fields['end'] = pd.to_datetime(end)
    # existing_fields['start'] = pd.to_datetime(existing_fields['start'])
    # existing_fields['end'] = pd.to_datetime(existing_fields['end'])
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

    # TODO: Fix inclusion checking.
    # It checks for new fields, not that every year/field combo has been satisfied.

    if len(gdf) > 0:
        # run data cu algorithm
        # for y in range(1985, 2024):  # Fails when those years are not represented in gridmet db table.
        # cur = con.cursor()
        # years = cur.execute("SELECT DISTINCT year FROM {}".format(gridmet)).fetchall()
        # cur.close()
        etbcs = []
        iwrc_cus = []
        frac_irrs = []
        mf_periods = []
        mfs = []

        cur = con.cursor()

        for i, row in tqdm(gdf.iterrows(), total=len(gdf)):
            # Loading in gridMET
            lon, lat = row['lon'], row['lat']
            elev = row['elev_gm']  # elevation from retrieved gridmet station
            # print(row['gfid'])

            if row['itype'] == 'P':
                pivot = True
                carryover = 0.5
            else:
                pivot = False
                carryover = 2.0

            # Get management factor for county that the field is in
            # TODO: How to determine the closest time period for management factor?
            # time periods in 'the rule': 1964–1973 (9), 1973–2006 (33), 1997–2006 (9)
            # HCU, recent HCU, and proposed use
            # Overlay w/ rs data: none (0), 1985-2006 (21), 1997-2023 (26) ?
            # Idea of running averages, por of 38 years of rs data, 1985-1995, 1990-2000, 1995-2005, 2000-2010,
            mf_timeperiod = 2
            mf_list = ['1964-1973', '1973-2006', '1997-2006']
            mf = MANAGEMENT_FACTORS[row['fid'][:3]][mf_timeperiod]
            mf_periods.append(mf_list[mf_timeperiod])
            mfs.append(mf)

            # Calculating bc seasonal ET
            # TODO: Find the closest station to field, complete static IWR computations.
            iwr_db_loc = ''
            station = '0000'
            bc, start1, end1 = iwr_database(iwr_db_loc, station, fullmonth=True, pivot=pivot)

            bc_cu = mf * bc['cu'].sum()
            bc_pet = bc['u'].sum()
            eff_precip = bc['ep'].sum()
            # Do I want to record bc_pet and eff_precip for this one too? Probably

            # Explanation of consumptive use calculations
            # IWR: sum of monthly ET minus effective precip minus carryover, all times management factor
            # sum(u_month - ep_month - co_month) * mf
            # OpenET: average seasonal ET times crop coefficient minus effective precip and carryover
            # (ETo * ETof) - ep - co

            if irrmapper:
                # Call year and field value from the irrmapper table... how?
                frac_irr = cur.execute("SELECT frac_irr from irrmapper WHERE fid=? and ", (gdf['']))
                # Also, how to apply it? Is this right?
                etbcs.append(bc_pet * frac_irr)
                iwrc_cus.append(bc_cu * frac_irr)
                frac_irrs.append(frac_irr)
            else:
                etbcs.append(bc_pet)
                iwrc_cus.append(bc_cu)
                frac_irrs.append(-1)

        fids = gdf['fid'].tolist()
        merged_data = [(fids[n], frac_irrs[n], mf_periods[n], mfs[n], etbcs[n], iwrc_cus[n]) for n in range(len(fids))]
        cur.executemany("INSERT INTO {} VALUES(?, ?, ?, ?, ?, ?)".format(out), merged_data)
        cur.close()
    else:
        print('Those fields already have consumptive use data.')
    print()


def cu_analysis_db(con, shp, gridmet, etof, out, start=1985, end=2024,
                   irrmapper=False, mf_timeperiod=2, selection=gpd.GeoDataFrame()):
    """
    Calculate average seasonal consumptive use with both IWR/DNRC and OpenET methods.

    :con: sqlite database connection
    :shp: str, name of table in sqlite database associated with field/gridmet lookup
    :gridmet: str, name of table in sqlite database associated with gridmet data
    :start: int, year of beginning of period of study
    :end: int, year after end of period of study
    :etof: str, name of table in sqlite database containing etof data
    :out: str, name of table in sqlite database containing results of consumptive use analysis by field
    :mf_timeperiod: int, which set of management factors to apply to dnrc_cu calculations, acceptable values of
    0, 1, or 2 to correspond to the three time periods/set of management factors described in the rule.
    """
    mf_list = ['1964-1973', '1973-2006', '1997-2006']

    print("Calculating consumptive use for fields")
    cur = con.cursor()

    # # Checking for inclusion, any new combos of field, year, irrmapper, and mf period
    # how to do that? Do I need to add a boolean column?

    field_year_tuples = []
    if selection.empty:
        in_fids = cur.execute("SELECT DISTINCT fid FROM {}".format(shp))
        for i in in_fids:
            for j in range(start, end):
                field_year_tuples.append((i[0], j, irrmapper, mf_list[mf_timeperiod]))
    else:
        in_fids = selection['FID']
        for i in in_fids:
            for j in range(start, end):
                field_year_tuples.append((i, j, irrmapper, mf_list[mf_timeperiod]))

    # temp table is closed and deleted at end of script
    cur.execute("DROP TABLE IF EXISTS temp.temp1")
    cur.execute("CREATE TEMP TABLE temp1(fid TEXT NOT NULL, year DATE NOT NULL, im INTEGER NOT NULL, mf_per TEXT NOT NULL)")
    cur.executemany("INSERT INTO temp1 VALUES(?, ?, ?, ?)", field_year_tuples)
    cur.execute("SELECT * FROM temp1 EXCEPT SELECT fid, year, irrmapper, mf_periods FROM {}".format(out))
    gdf = cur.fetchall()

    if len(gdf) > 0:
        print("{} new entries".format(len(gdf)))
        fids = []
        ys = []
        etoss = []
        etbcs = []
        etofs = []
        opnt_cus = []
        dnrc_cus = []
        frac_irrs = []
        mf_periods = []
        mfs = []
        if irrmapper:
            im = np.ones(len(gdf))
        else:
            im = np.zeros(len(gdf))

        # There are now more than 2 values to unpack...
        dictionary = {k: [] for k, v, w, x in gdf}
        for k, v, w, x in gdf:
            dictionary[k].append(v)
        # Is that fast enough? Seems inefficient... At least I have already removed the old stuff.
        # Is the dictionary creation faster than the time saved by not retrieving the row data 39 times?

        for fid, yr_values in tqdm(dictionary.items(), total=len(dictionary)):
            # Do all time-independent stuff first
            row = cur.execute("SELECT * FROM {} WHERE fid=?".format(shp), (fid,)).fetchone()
            # Loading in gridMET
            lon, lat = row[7], row[6]
            elev = row[8]  # elevation from retrieved gridmet station
            # print(row['gfid'])

            if row[1] == 'P':  # Corresponds to 'itype'
                pivot = True
                carryover = 0.5
            else:
                pivot = False
                carryover = 2.0

            # Get management factor for county that the field is in
            # How to determine the closest time period for management factor? No, just don't do that.
            # This is static for now...
            # time periods in 'the rule': 1964–1973 (9), 1973–2006 (33), 1997–2006 (9)
            # HCU, recent HCU, and proposed use
            # Overlay w/ rs data: none (0), 1985-2006 (21), 1997-2023 (26) ?
            # Idea of running averages, por of 38 years of rs data, 1985-1995, 1990-2000, 1995-2005, 2000-2010,
            mf = MANAGEMENT_FACTORS[fid[:3]][mf_timeperiod]

            for y in yr_values:
                mf_periods.append(mf_list[mf_timeperiod])
                mfs.append(mf)
                fids.append(fid)
                ys.append(y)
                grd = pd.read_sql("SELECT date, eto_mm, tmin_c, tmax_c, prcp_mm FROM {} WHERE gfid={} AND year={}"
                                  .format(gridmet, row[5], y), con)
                # grd = pd.read_sql("SELECT date, eto_mm, tmin_c, tmax_c, prcp_mm FROM ? WHERE gfid=? AND date(date) "
                #                   "BETWEEN date('?') AND date('?')", con, params=(gridmet, row['gfid'], start, end))
                # rename variables needed in blaney_criddle calculations
                grd = grd.rename(columns={'eto_mm': 'ETOS', 'tmin_c': 'MN', 'tmax_c': 'MX', 'prcp_mm': 'PP'})
                grd.index = pd.to_datetime(grd['date'])

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
                # et_by_year = grd.groupby(grd.index.year)['ETOS'].sum()
                # single year, can just sum it all down below.

                # Loading in OpenET eotf/kc
                df = pd.read_sql("SELECT time, etof FROM {} WHERE fid='{}' AND date(time) BETWEEN date('{}-01-01') "
                                 "AND date('{}-12-31')".format(etof, fid, y, y), con,
                                 index_col='time', parse_dates={'time': '%Y-%m-%d'})
                r_index = pd.date_range('{}-01-01'.format(y), '{}-12-31'.format(y), freq='D')  # avg over all data
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

                if irrmapper:
                    # Call year and field value from the irrmapper table... how?
                    frac_irr = cur.execute("SELECT frac_irr FROM irrmapper WHERE fid=? AND year=?", (fid, y)).fetchone()
                    # Also, how to apply it? Is this right?
                    etoss.append(grd['ETOS'].sum() * frac_irr[0])
                    etbcs.append(bc_pet * frac_irr[0])
                    etofs.append(df['etof'].mean() * frac_irr[0])
                    opnt_cus.append((grd['ETOS'].sum() * df['etof'].mean() - eff_precip - carryover) * frac_irr[0])
                    dnrc_cus.append(bc_cu * frac_irr[0])
                    frac_irrs.append(frac_irr[0])
                else:
                    etoss.append(grd['ETOS'].sum())
                    etbcs.append(bc_pet)
                    etofs.append(df['etof'].mean())
                    opnt_cus.append(grd['ETOS'].sum() * df['etof'].mean() - eff_precip - carryover)
                    dnrc_cus.append(bc_cu)
                    frac_irrs.append(-1)

        merged_data = [(fids[n], ys[n], im[n], frac_irrs[n], mf_periods[n], mfs[n], etoss[n], etbcs[n], etofs[n],
                        opnt_cus[n], dnrc_cus[n]) for n in range(len(fids))]
        cur.executemany("INSERT INTO {} VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)".format(out), merged_data)
    else:
        print('That consumptive use data is already saved.')
    cur.close()
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


# Might still be needed for gridmet? Not needed if db tables are created properly.
def pd_to_sql_ignore_on_conflict(table, conn, keys, data_iter):
    """ For use as argument in pandas.DataFrame.to_sql 'method' parameter. """
    data = [row for row in data_iter]
    # stmt = "INSERT OR IGNORE INTO {} VALUES(?, ?, ?, ?)".format(table.name)  # generalized to variable len below.
    stmt = "INSERT OR IGNORE INTO {} VALUES(".format(table.name)
    for k in range(len(keys)):
        stmt += "?, "
    stmt = stmt[:-2] + ")"
    result = conn.executemany(stmt, data)
    return result.rowcount


if __name__ == '__main__':

    if os.path.exists('F:/FileShare'):
        main_dir = 'F:/FileShare/openet_pilot'
    else:
        main_dir = 'F:/openet_pilot'

    # # STEP ZERO: CREATE DATABASE TABLES
    conec = sqlite3.connect(os.path.join(main_dir, "opnt_analysis_03042024_Copy.db"))
    # has gm data for all counties, do not open table in gui!
    # conec = sqlite3.connect("C:/Users/CND571/Documents/Data/random_05082024.db")  # test
    # sqlite database table names
    gm_ts, fields_db, results, etof_db, irr_db = 'gridmet_ts', 'field_data', 'field_cu_results', 'opnt_etof', 'irrmapper'
    # Initialize tables with correct column names/types/primary keys
    init_db_tables(conec)

    # # STEP ONE: UPDATE IRRMAPPER DB TABLE
    im_file = os.path.join(main_dir, 'irrmapper_ref_SID.csv')
    # update_irrmapper_table(conec, im_file)

    # # STEP TWO: UPDATE ETOF DB TABLE
    etof_loc = os.path.join(main_dir, 'etof_files')  # loads all data
    # update_etof_db_1(conec, etof_loc, etof_db)

    # # # STEP THREE: RUN OTHER 3 TABLES TO GET RESULTS
    # # gridmet information
    # gm_d = os.path.join(main_dir, 'gridmet')  # location of general gridmet files
    # gridmet_cent = os.path.join(gm_d, 'gridmet_centroids_MT.shp')
    # rasters_ = os.path.join(gm_d, 'correction_surfaces_aea')  # correction surfaces, one for each month and variable.
    # # fields subset
    mt_file = os.path.join(main_dir, "statewide_irrigation_dataset_15FEB2024_5071.shp")
    mt_fields = gpd.read_file(mt_file)  # takes a bit (8.3s)
    mt_fields['county'] = mt_fields['FID'].str.slice(0, 3)
    # mt_fields = mt_fields[mt_fields['county'] == '019']
    # # recent dates
    # pos_start = '2016-01-01'
    # pos_end = '2023-12-31'
    # now run db stuff
    # gridmet_match_db(conec, mt_fields, gridmet_cent, fields_db)
    # corrected_gridmet_db_1(conec, gridmet_cent, fields_db, gm_ts, rasters_, pos_start, pos_end)
    # cu_analysis_db(conec, fields_db, gm_ts, etof_db, results, 2016, 2023, selection=mt_fields)

    for k in ['011', '025', '109']:
        COUNTIES.pop(k, None)
    cntys = list(COUNTIES.keys())
    for i in cntys[1:]:  # '001' complete
        print(i)
        fields = mt_fields[mt_fields['county'] == i]
        cu_analysis_db(conec, fields_db, gm_ts, etof_db, results, 1987, 2024, selection=fields)

    # # STEP FOUR: FINISH/CLOSE THINGS
    cursor = conec.cursor()
    cursor.execute("PRAGMA analysis_limit=500")
    cursor.execute("PRAGMA optimize")
    cursor.close()

    conec.commit()
    conec.close()

# ========================= EOF ====================================================================
