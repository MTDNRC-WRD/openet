
import os

import pandas as pd
import geopandas as gpd
import numpy as np
import pyproj
from rasterstats import zonal_stats
from tqdm import tqdm
import sqlite3
from datetime import timedelta
from geopy import distance
from chmdata.thredds import GridMet

# from utils.thredds import GridMet
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
    """ Initialize the tables in the sqlite database.

    Parameters
    ----------
    con: sqlite database connection object
    """
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
                """)
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
                """)
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
                """)
    cur.execute("""
                CREATE TABLE IF NOT EXISTS static_iwr_results
                (fid TEXT NOT NULL,
                mf_periods TEXT,
                mfs REAL,
                etbc REAL,
                dnrc_cu REAL,
                PRIMARY KEY (fid, mf_periods) ON CONFLICT IGNORE
                );
                """)
    cur.execute("""
                CREATE TABLE IF NOT EXISTS irrmapper
                (fid TEXT NOT NULL,
                year DATE NOT NULL,
                frac_irr REAL,
                PRIMARY KEY (fid, year) ON CONFLICT IGNORE
                );
                """)

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
    def convert_to_wgs84(x, y):
        return pyproj.Transformer.from_crs('EPSG:5071', 'EPSG:4326').transform(x, y)

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
            [rasters.append(os.path.join(gridmet_ras, 'gridmet_corrected_{}_{}.tif'
                                         .format(v, m))) for m in range(1, 13)]

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
            [rasters.append(os.path.join(gridmet_ras, 'gridmet_corrected_{}_{}.tif'
                                         .format(v, m))) for m in range(1, 13)]

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


def cu_analysis_db(con, shp, gridmet, etof, out, start=1985, end=2024,
                   irrmapper=False, mf_timeperiod=2, selection=gpd.GeoDataFrame()):
    """
    Calculate average seasonal consumptive use with both IWR/DNRC and OpenET methods.

    Parameters
    ----------
    con: sqlite database connection
    shp: str, name of table in sqlite database associated with field/gridmet lookup
    gridmet: str, name of table in sqlite database associated with gridmet data
    etof: str, name of table in sqlite database containing etof data
    out: str, name of table in sqlite database containing results of consumptive use analysis by field
    start: int, year of beginning of period of study
    end: int, year after end of period of study
    irrmapper: bool, optional
    mf_timeperiod: int, which set of management factors to apply to dnrc_cu calculations, acceptable values of
    0, 1, or 2 to correspond to the three time periods/set of management factors described in the rule.
    selection: GeoDataFrame, optional
    """
    mf_list = ['1964-1973', '1973-2006', '1997-2006']

    print("Calculating consumptive use for fields")
    cur = con.cursor()

    # Checking for inclusion, any new combos of field, year, irrmapper, and mf period
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
    cur.execute("CREATE TEMP TABLE temp1(fid TEXT NOT NULL, year DATE NOT NULL, "
                "im INTEGER NOT NULL, mf_per TEXT NOT NULL)")
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
                r_index = pd.date_range('{}-01-01'.format(y), '{}-12-31'.format(y), freq='D')
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
        con.commit()
    else:
        print('That consumptive use data is already saved.')
    cur.close()
    print()


def iwr_static_cu_analysis_db(con, shp, out, clim_db_loc, mf_timeperiod=2, selection=gpd.GeoDataFrame(), iwr_coord=""):
    """ Calculate average seasonal consumptive use with IWR climate database files.

    Parameters
    ----------
    con: sqlite database connection
    shp: str, name of table in sqlite database associated with field/gridmet lookup
    out: str, name of table in sqlite database containing results of consumptive use analysis by field
    clim_db_loc:
    mf_timeperiod: int, which set of management factors to apply to dnrc_cu calculations, acceptable values of
    0, 1, or 2 to correspond to the three time periods/set of management factors described in the rule.
    selection: GeoDataFrame, optional
    iwr_coord:
    """
    mf_list = ['1964-1973', '1973-2006', '1997-2006']

    print("Calculating consumptive use for fields")
    cur = con.cursor()

    # Checking for inclusion, any new combos of field, irrmapper, and mf period
    field_year_tuples = []
    if selection.empty:
        in_fids = cur.execute("SELECT DISTINCT fid FROM {}".format(shp))
        for i in in_fids:
            field_year_tuples.append((i[0], mf_list[mf_timeperiod]))
    else:
        in_fids = selection['FID']
        for i in in_fids:
            field_year_tuples.append((i, mf_list[mf_timeperiod]))

    # temp table is closed and deleted at end of script
    cur.execute("DROP TABLE IF EXISTS temp.temp2")
    cur.execute("CREATE TEMP TABLE temp2(fid TEXT NOT NULL, mf_per TEXT NOT NULL)")
    cur.executemany("INSERT INTO temp2 VALUES(?, ?)", field_year_tuples)
    cur.execute("SELECT * FROM temp2 EXCEPT SELECT fid, mf_periods FROM {}".format(out))
    gdf = cur.fetchall()
    # gdf = gdf[:int(len(gdf)/2)]  # to get some stuff loaded before it breaks (memory access error on larger chunks?)

    if len(gdf) > 0:
        print("{} new entries".format(len(gdf)))
        fids = []
        etbcs = []
        dnrc_cus = []
        mf_periods = []
        mfs = []

        if len(iwr_coord) > 0:
            iwr_stations = iwr_station_data(iwr_coord)
        else:
            iwr_stations = iwr_station_data()

        for item in tqdm(gdf, total=len(gdf)):
            fid = item[0]
            # Do all time-independent stuff first
            row = cur.execute("SELECT * FROM {} WHERE fid=?".format(shp), (fid,)).fetchone()

            if row[1] == 'P':  # Corresponds to 'itype'
                pivot = True
            else:
                pivot = False

            # Get management factor for county that the field is in
            # How to determine the closest time period for management factor? No, just don't do that.
            # This is static for now...
            # time periods in 'the rule': 1964–1973 (9), 1973–2006 (33), 1997–2006 (9)
            # HCU, recent HCU, and proposed use
            # Overlay w/ rs data: none (0), 1985-2006 (21), 1997-2023 (26) ?
            # Idea of running averages, por of 38 years of rs data, 1985-1995, 1990-2000, 1995-2005, 2000-2010,
            mf = MANAGEMENT_FACTORS[fid[:3]][mf_timeperiod]

            mf_periods.append(mf_list[mf_timeperiod])
            mfs.append(mf)
            fids.append(fid)

            station = closest_iwr_station_best(row, iwr_stations)

            # Calculating bc seasonal ET
            bc, start1, end1 = iwr_database(clim_db_loc, station, fullmonth=True, pivot=pivot)

            bc_cu = mf * bc['cu'].sum()  # add management factor
            bc_pet = bc['u'].sum()

            # Explanation of consumptive use calculations
            # IWR: sum of monthly ET minus effective precip minus carryover, all times management factor
            # sum(u_month - ep_month - co_month) * mf
            # OpenET: average seasonal ET times crop coefficient minus effective precip and carryover
            # (ETo * ETof) - ep - co

            etbcs.append(bc_pet)
            dnrc_cus.append(bc_cu)

        merged_data = [(fids[n], mf_periods[n], mfs[n], etbcs[n], dnrc_cus[n]) for n in range(len(fids))]
        cur.executemany("INSERT INTO {} VALUES(?, ?, ?, ?, ?)".format(out), merged_data)
    else:
        print('That consumptive use data is already saved.')
    cur.close()
    print()


# Might still be needed for gridmet? Not needed if db tables are created properly.
def pd_to_sql_ignore_on_conflict(table, conn, keys, data_iter):
    """ For use as argument in pandas.DataFrame.to_sql 'method' parameter.

    Parameters
    ----------
    table:
    conn:
    keys:
    data_iter:
    """
    data = [row for row in data_iter]
    # stmt = "INSERT OR IGNORE INTO {} VALUES(?, ?, ?, ?)".format(table.name)  # generalized to variable len below.
    stmt = "INSERT OR IGNORE INTO {} VALUES(".format(table.name)
    for k in range(len(keys)):
        stmt += "?, "
    stmt = stmt[:-2] + ")"
    result = conn.executemany(stmt, data)
    return result.rowcount


def iwr_station_data(file_loc="C:/Users/CND571/Documents/iwr_stations.geojson"):
    """ Load and process file with IWR station coordinates. """
    inv_counties = {v: k for k, v in COUNTIES.items()}

    jack = gpd.read_file(file_loc)
    jack['station_no'] = [i[2:] for i in jack['station_no']]
    jack['county'] = [i[:-4] for i in jack['county']]
    jack = jack.drop(columns=['geometry'])
    jack['county_no'] = [inv_counties[i] for i in jack['county']]

    return jack


def closest_iwr_station(row, jack):
    """ Find IWR station closest to given field. Original option.

    If there are IWR stations in the same county as the field, the closest of those will be chosen.
    Otherwise, all 180 stations will be searched for the closest station.

    Parameters
    ----------
    row: field data
    jack: pandas dataframe containing IWR station info.

    returns best match qualities: station number, distance from field, and elevation
    """
    field_county = row[4]
    field_lat = row[6]
    field_lon = row[7]

    options = jack[jack['county_no'] == field_county]
    dists = []
    if options.empty:  # search all stations
        for i, stn in jack.iterrows():
            this_dist = distance.geodesic((field_lat, field_lon), (stn['lat'], stn['lon'])).km
            dists.append(this_dist)
        best_stn = dists.index(min(dists))
        best_stn_no = jack.iloc[best_stn]['station_no']
        best_stn_elev = jack.iloc[best_stn]['elev']/3.281
    else:  # search stations within county
        for i, stn in options.iterrows():
            this_dist = distance.geodesic((field_lat, field_lon), (stn['lat'], stn['lon'])).km
            dists.append(this_dist)
        best_stn = dists.index(min(dists))
        best_stn_no = options.iloc[best_stn]['station_no']
        best_stn_elev = options.iloc[best_stn]['elev']/3.281

    return best_stn_no, min(dists), best_stn_elev


def closest_iwr_station_1(row, jack):
    """ Find IWR station closest to given field.

    All 180 stations will be searched for the closest station.

    Parameters
    ----------
    row: field data
    jack: pandas dataframe containing IWR station info.

    returns best match qualities: station number, distance from field, and elevation
    """
    field_county = row[4]
    field_lat = row[6]
    field_lon = row[7]

    dists = []

    for i, stn in jack.iterrows():
        this_dist = distance.geodesic((field_lat, field_lon), (stn['lat'], stn['lon'])).km
        dists.append(this_dist)
    best_stn = dists.index(min(dists))
    best_stn_no = jack.iloc[best_stn]['station_no']
    best_stn_elev = jack.iloc[best_stn]['elev']/3.281

    return best_stn_no, min(dists), best_stn_elev


def closest_iwr_station_2(row, jack):
    """ Find representative IWR station for given field.

    If there are no IWR stations in the same county as the field, the closest station will be chosen.
    Otherwise, both the closest station and the closest station within the county will be identified.
    Of the two potential stations, the one that is closer in elevation to the field will be chosen.

    Parameters
    ----------
    row: field data
    jack: pandas dataframe containing IWR station info.

    returns best match qualities: station number, distance from field, and elevation
    """
    field_county = row[4]
    field_lat = row[6]
    field_lon = row[7]
    field_elev = row[8]  # m

    options = jack[jack['county_no'] == field_county]
    dists = []
    dists1 = []
    if options.empty:  # search all stations, choose closest
        for i, stn in jack.iterrows():
            this_dist = distance.geodesic((field_lat, field_lon), (stn['lat'], stn['lon'])).km
            dists.append(this_dist)
        best_stn = dists.index(min(dists))
        best_stn_no = jack.iloc[best_stn]['station_no']
        best_stn_elev = jack.iloc[best_stn]['elev']/3.281
        return best_stn_no, min(dists), best_stn_elev
    else:  # search stations within county, compare elevation with closest overall station
        # stations within county
        for i, stn in options.iterrows():
            this_dist = distance.geodesic((field_lat, field_lon), (stn['lat'], stn['lon'])).km
            dists1.append(this_dist)
        best_stn1 = dists1.index(min(dists1))
        best_stn_no1 = options.iloc[best_stn1]['station_no']
        best_stn_elev1 = options.iloc[best_stn1]['elev']/3.281
        # all stations
        for i, stn in jack.iterrows():
            this_dist = distance.geodesic((field_lat, field_lon), (stn['lat'], stn['lon'])).km
            dists.append(this_dist)
        best_stn = dists.index(min(dists))
        best_stn_no = jack.iloc[best_stn]['station_no']
        best_stn_elev = jack.iloc[best_stn]['elev']/3.281
        # prioritize elevation accuracy
        if np.abs(field_elev - best_stn_elev) < np.abs(field_elev - best_stn_elev1):  # Sign direction?
            return best_stn_no, min(dists), best_stn_elev  # overall closest
        else:
            return best_stn_no1, min(dists1), best_stn_elev1  # closest in county


# Old algorthim (closest_iwr_station) was used for current results as of 6/18/2024.
# New table must be made to use this algorithm (same as closest_iwr_station_2).
def closest_iwr_station_best(row, jack):
    """ Find representative IWR station for given field.

    Guidance from rule: "most representative station" is not necessarily within the same county,
    and must be at a similar elevation and climactic conditions.

    This algorithm: If there are no IWR stations in the same county as the field, the closest station will be chosen.
    Otherwise, both the closest station and the closest station within the county will be identified.
    Of the two potential stations, the one that is closer in elevation to the field will be chosen.

    Parameters
    ----------
    row: field data
    jack: pandas dataframe containing IWR station info.
    """
    field_county = row[4]
    field_lat = row[6]
    field_lon = row[7]
    field_elev = row[8]  # m

    options = jack[jack['county_no'] == field_county]
    dists = []
    dists1 = []

    # search all stations, choose closest
    for i, stn in jack.iterrows():
        this_dist = distance.geodesic((field_lat, field_lon), (stn['lat'], stn['lon'])).km
        dists.append(this_dist)
    best_stn = dists.index(min(dists))
    best_stn_no = jack.iloc[best_stn]['station_no']
    best_stn_elev = jack.iloc[best_stn]['elev'] / 3.281

    if options.empty:
        return best_stn_no
    else:  # search stations within county, compare elevation with closest overall station
        # search stations within county
        for i, stn in options.iterrows():
            this_dist = distance.geodesic((field_lat, field_lon), (stn['lat'], stn['lon'])).km
            dists1.append(this_dist)
        best_stn1 = dists1.index(min(dists1))
        best_stn_no1 = options.iloc[best_stn1]['station_no']
        best_stn_elev1 = options.iloc[best_stn1]['elev'] / 3.281
        # prioritize elevation accuracy
        if np.abs(field_elev - best_stn_elev) < np.abs(field_elev - best_stn_elev1):
            return best_stn_no  # overall closest
        else:
            return best_stn_no1  # closest in county


def test_iwr_station_search(con, shp, iwr_coord=""):
    """ Compare different search schemes for IWR stations. """

    cur = con.cursor()

    num = 1000

    fids = pd.read_sql("SELECT fid FROM field_data", con)
    fids = fids.sample(n=num, random_state=23)  # Constant seed to make reproducible

    # # Not reproducible, but might be much faster?
    # fids = pd.read_sql("SELECT fid FROM field_data WHERE fid IN "
    #                    "(SELECT fid FROM field_data ORDER BY RANDOM() LIMIT 100)", con)

    if len(iwr_coord) > 0:
        iwr_stations = iwr_station_data(iwr_coord)
    else:
        iwr_stations = iwr_station_data()

    same = []
    elev_bet = []
    mean_dist = np.zeros(num)
    mean_dist1 = np.zeros(num)
    mean_dist2 = np.zeros(num)
    mean_elev = np.zeros(num)
    mean_elev1 = np.zeros(num)
    mean_elev2 = np.zeros(num)
    for i in range(num):
        fid = fids.values[i][0]
        row = cur.execute("SELECT * FROM {} WHERE fid=?".format(shp), (fid,)).fetchone()
        station, dist, elev = closest_iwr_station(row, iwr_stations)
        station1, dist1, elev1 = closest_iwr_station_1(row, iwr_stations)
        station2, dist2, elev2 = closest_iwr_station_2(row, iwr_stations)
        mean_dist[i] = dist
        mean_dist1[i] = dist1
        mean_dist2[i] = dist2
        mean_elev[i] = elev - row[-1]
        mean_elev1[i] = elev1 - row[-1]
        mean_elev2[i] = elev2 - row[-1]
        if station == station1:
            same.append(i)
            # print("{} {:.1f} m".format(fid, row[-1]))
            # print("{} {:.1f} m {:.2f} km".format(station, elev, dist))
            # print()
        else:
            if np.abs(mean_elev[i]) > np.abs(mean_elev1[i]):
                elev_bet.append(i)
            # print(fid, "{:.1f} m".format(row[-1]))
            # print("{} {:.1f} m {:.2f} km".format(station, elev, dist))
            # print("{} {:.1f} m {:.2f} km".format(station1, elev1, dist1))
            # print()
    mean_elev = np.abs(mean_elev)
    mean_elev1 = np.abs(mean_elev1)
    mean_elev2 = np.abs(mean_elev2)
    dif = list(range(num))
    for i in same:
        dif.remove(i)

    print("{}/{} stations same result ({:.1f}%)".format(len(same), num, 100*len(same)/num))
    print("{}/{} changed stations had elevation match improve ({:.1f}%)"
          .format(len(elev_bet), num-len(same), 100*len(elev_bet)/(num-len(same))))
    print("Options: closest in county, closest, choose btwn in-county and closest based on elevation match")
    print("Average difference in elevation: {:.1f} {:.1f} {:.1f} m"
          .format(mean_elev.mean(), mean_elev1.mean(), mean_elev2.mean()))
    print("Average distance from field: {:.1f} {:.1f} {:.1f} km"
          .format(mean_dist.mean(), mean_dist1.mean(), mean_dist2.mean()))
    print()
    print("Average difference in elevation: {:.1f} {:.1f} {:.1f} m"
          .format(mean_elev[dif].mean(), mean_elev1[dif].mean(), mean_elev2[dif].mean()))
    print("Average distance from field: {:.1f} {:.1f} {:.1f} km"
          .format(mean_dist[dif].mean(), mean_dist1[dif].mean(), mean_dist2[dif].mean()))

    cur.close()


if __name__ == '__main__':

    if os.path.exists('F:/FileShare'):
        main_dir = 'F:/FileShare/openet_pilot'
    else:
        main_dir = 'F:/openet_pilot'

    # # Required Filepaths/Variable Names
    # sqlite database table names (Do these really need to be passed as variables?)
    gm_ts, fields_db, results, etof_db = 'gridmet_ts', 'field_data', 'field_cu_results', 'opnt_etof'
    irr_db, iwr_cu_db = 'irrmapper', 'static_iwr_results'
    # irrmapper data
    im_file = os.path.join(main_dir, 'irrmapper_ref_SID.csv')
    # all openet etof data
    etof_loc = os.path.join(main_dir, 'etof_files')  # loads all data
    # Gridmet information
    gm_d = os.path.join(main_dir, 'gridmet')  # location of general gridmet files
    gridmet_cent = os.path.join(gm_d, 'gridmet_centroids_MT.shp')
    rasters_ = os.path.join(gm_d, 'correction_surfaces_aea')  # correction surfaces, one for each month and variable.
    # define period of study (for gridmet fetching)
    pos_start = '1987-01-01'
    pos_end = '2023-12-31'
    # location of IWR climate database
    iwr_clim_loc = os.path.join(main_dir, 'IWR', 'climate.db')
    iwr_coord_loc = os.path.join(main_dir, 'IWR', 'iwr_stations.geojson')
    # iwr_clim_loc = 'C:/Users/CND571/Documents/IWR/Database/climate.db'

    # Load Statewide Irrigation Dataset (about 10 seconds)
    mt_file = os.path.join(main_dir, "statewide_irrigation_dataset_15FEB2024_5071.shp")
    mt_fields = gpd.read_file(mt_file)  # takes a bit (8.3s)
    mt_fields['county'] = mt_fields['FID'].str.slice(0, 3)
    # remove tiny fields, no etof data for them.
    mt_fields['area_m2'] = mt_fields['geometry'].area
    tiny_fields = mt_fields[mt_fields['area_m2'] < 100].index
    clean_sid = mt_fields.drop(tiny_fields)

    # # Looking at distribution of field sizes
    # print(clean_sid['area_m2'].sort_values() / 1e6)
    # print(clean_sid[clean_sid['area_m2'] < 1e6].count())
    # print(clean_sid[clean_sid['area_m2'] > 2e6].count())
    # import matplotlib.pyplot as plt
    # plt.figure()
    # (clean_sid['area_m2']/1e6).hist(bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], log=True)
    # plt.show()

    mt_fields = clean_sid.drop(columns=['area_m2'])
    # optional county subset
    # mt_fields = mt_fields[mt_fields['county'] == '019']

    # # Database stuff
    # # --------------

    # conec = sqlite3.connect(os.path.join(main_dir, "opnt_analysis_03042024_Copy.db"))  # full project
    # conec = sqlite3.connect("C:/Users/CND571/Documents/Data/random_05082024.db")  # test

    # # Initialize tables with correct column names/types/primary keys
    # init_db_tables(conec)

    # # Populate irrmapper db table (a few seconds)
    # update_irrmapper_table(conec, im_file)

    # # Populate etof db table (about 15 minutes?)
    # update_etof_db_1(conec, etof_loc, etof_db)

    # # Populate fid/gridmet lookup db table (about 7 hours?)
    # gridmet_match_db(conec, mt_fields, gridmet_cent, fields_db)

    # # Populate gridmet db table (takes forever)
    # corrected_gridmet_db_1(conec, gridmet_cent, fields_db, gm_ts, rasters_, pos_start, pos_end)

    # # Populate consumptive use result db table (about 30 hours?)
    # for k in ['011', '025', '101', '109']:  # issue w/ 101 etof data
    #     COUNTIES.pop(k, None)
    # cntys = list(COUNTIES.keys())
    # for i in cntys:
    #     print(i)
    #     fields = mt_fields[mt_fields['county'] == i]
    #     cu_analysis_db(conec, fields_db, gm_ts, etof_db, results, 1987, 2024, selection=fields)
    #
    # # # Populate IWR static climate consumptive use result db table (about 30 mins?)
    # # iwr_static_cu_analysis_db(conec, fields_db, iwr_cu_db, iwr_clim_loc, iwr_coord=iwr_coord_loc)

    # # SCRATCH WORK

    # fields = mt_fields[mt_fields['county'] == '101']['FID']
    # for fid in fields:
    #     for y in range(1987, 2024):
    #         df = pd.read_sql("SELECT time, etof FROM {} WHERE fid='{}' AND date(time) BETWEEN date('{}-01-01') "
    #                          "AND date('{}-12-31')".format(etof_db, fid, y, y), conec,
    #                          index_col='time', parse_dates={'time': '%Y-%m-%d'})
    #         print(fid, y, len(df))

    # # Finish/Close database things
    # cursor = conec.cursor()
    # cursor.execute("PRAGMA analysis_limit=500")
    # cursor.execute("PRAGMA optimize")
    # cursor.close()

    # conec.commit()
    # conec.close()

# ========================= EOF ====================================================================
