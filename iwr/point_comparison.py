import os
import warnings
from copy import deepcopy

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.geometry import Point, MultiPolygon

from reference_et.combination import pm_fao56, get_rn
from reference_et.modified_bcriddle import modified_blaney_criddle
from reference_et.modified_bcriddle import modified_blaney_criddle_1
from reference_et.modified_bcriddle import modified_blaney_criddle_neh_ex
from reference_et.rad_utils import extraterrestrial_r, calc_rso
from utils.agrimet import load_stations
from utils.agrimet import Agrimet
from utils.agrimet import MT_STATIONS
from utils.elevation import elevation_from_coordinate_ee
from utils.thredds import GridMet

import requests

warnings.filterwarnings(action='once')

large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large,
          'xtick.color': 'black',
          'ytick.color': 'black',
          'xtick.direction': 'out',
          'ytick.direction': 'out',
          'xtick.bottom': True,
          'xtick.top': False,
          'ytick.left': True,
          'ytick.right': False,
          }

plt.rcParams.update(params)
# print(plt.style.available)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("white", {'axes.linewidth': 0.5})


def openet_get_fields():
    # import requests

    # set your API key before making the request
    header = {"Authorization": 'ZBXxCeBRsSgkeLvsROKVTDS1w9UV0xfOKyEJGTNcEEPT15DQsYfbB0uu1K9w'}

    # endpoint arguments
    args = {
        "date_range": [
            "2018-01-01",
            "2021-12-31"
        ],
        "interval": "monthly",
        "asset_id": "projects/ee-hehaugen/assets/sweetgrass_fields_sample",  ## ?
        "attributes": [
            "FID"
        ],
        "reducer": "mean",
        "model": "Ensemble",
        "variable": "ETof",
        "reference_et": "gridMET",
    }

    # # endpoint arguments
    # args = {
    #     "date_range": [
    #         "2019-01-01",
    #         "2019-12-31"
    #     ],
    #     "interval": "monthly",
    #     "asset_id": "projects/openet/api_demo_features",
    #     "attributes": [
    #         "id"
    #     ],
    #     "reducer": "mean",
    #     "model": "ptJPL",
    #     "variable": "ET",
    #     "reference_et": "gridMET",
    #     "units": "mm"
    # }

    # query the api
    resp = requests.post(
        headers=header,
        json=args,
        url="https://openet-api.org/raster/timeseries/multipolygon"
    )

    print(resp.json())

def field_comparison(shp, etof, out):
    gdf = gpd.read_file(shp)
    start, end = '2016-01-01', '2021-12-31'

    summary = deepcopy(gdf)
    summary['etos'] = [-99.99 for _ in summary['FID']]
    summary['etbc'] = [-99.99 for _ in summary['FID']]
    summary['geo'] = [None for _ in summary['FID']]
    idx = max(summary.index.values) + 1

    ct = 0
    for i, row in gdf.iterrows():
        file_ = os.path.join(etof, str(row['FID']).rjust(3, '0'))
        lon, lat = row.geometry.centroid.x, row.geometry.centroid.y
        elev = elevation_from_coordinate_ee(lat, lon)
        gridmet = GridMet('pet', start=start, end=end,
                          lat=lat, lon=lon)
        grd = gridmet.get_point_timeseries()  ## initialize df with first time series
        grd = grd.rename(columns={'pet': 'ETOS'})
        for var, _name in zip(['tmmn', 'tmmx', 'pr'], ['MN', 'MX', 'PP']):  ## collect additional time series
            ts = GridMet(var, start=start, end=end,
                         lat=lat, lon=lon).get_point_timeseries()
            if 'tm' in var:
                ts -= 273.15

            grd[_name] = ts

        grd['MM'] = (grd['MN'] + grd['MX']) / 2
        bc, start1, end1, kc = modified_blaney_criddle(grd, lat, elev)
        ## start and end were being overwritten by the return of the mbc function. They should not be.
        ## renamed to start1 and end1

        bc_pet = bc['u'].sum()
        summary.loc[i, 'etbc'] = bc_pet

        grd['mday'] = ['{}-{}'.format(x.month, x.day) for x in grd.index]
        target_range = pd.date_range('2000-{}-{}'.format(start1.month, start1.day),
                                     '2000-{}-{}'.format(end1.month, end1.day))
        accept = ['{}-{}'.format(x.month, x.day) for x in target_range]
        grd['mask'] = [1 if d in accept else 0 for d in grd['mday']]
        grd = grd[grd['mask'] == 1]

        df = pd.read_csv(file_, index_col=0, parse_dates=True) ## monthly ET, from OpenET?
        df = df.rename(columns={list(df.columns)[0]: 'etof'})
        r_index = pd.date_range('2016-01-01', '2021-12-31', freq='D')
        df = df.reindex(r_index)
        df = df.interpolate()
        df['mday'] = ['{}-{}'.format(x.month, x.day) for x in df.index]
        target_range = pd.date_range('2016-05-09', '2016-09-19')

        accept = ['{}-{}'.format(x.month, x.day) for x in target_range]
        df['mask'] = [1 if d in accept else 0 for d in df['mday']]
        df = df[df['mask'] == 1]
        if isinstance(row['geometry'], MultiPolygon): ## multi-part fields
            first = True
            for g in row['geometry'].geoms:
                if first:
                    summary.loc[i, 'geo'] = g
                    summary.loc[i, 'etof'] = df['etof'].mean()
                    summary.loc[i, 'etos'] = grd['ETOS'].mean() ## gridmet et

                    first = False
                else: ## adding additional parts of fields to end of file?
                    summary.loc[idx] = row
                    summary.loc[idx, 'geo'] = g
                    summary.loc[idx, 'etof'] = df['etof'].mean()
                    summary.loc[idx, 'etos'] = grd['ETOS'].mean() ## gridmet et
                    idx += 1
        else:
            summary.loc[i, 'geo'] = row['geometry']
            summary.loc[i, 'etof'] = df['etof'].mean()
            summary.loc[i, 'etos'] = grd['ETOS'].mean() ## gridmet et
        print(summary)
    # return summary
        pass


def point_comparison_iwr_stations(_dir, meta_csv, out_summary):
    meta_df = pd.read_csv(meta_csv)
    summary = deepcopy(meta_df)

    summary['etos'] = [-99.99 for _ in summary['LON']]
    summary['etbc'] = [-99.99 for _ in summary['LON']]
    summary['geo'] = [None for _ in summary['LON']]

    for i, row in meta_df.iterrows():
        _file = os.path.join(_dir, '{}.csv'.format(row['STAID']))
        df = pd.read_csv(_file)
        lat = df.iloc[0]['LATITUDE']
        lon = df.iloc[0]['LONGITUDE']
        elev = row['ELEV']
        geo = Point(lon, lat)
        summary.loc[i, 'geo'] = geo
        start, end = '1997-01-01', '2006-12-31'
        gridmet = GridMet('pet', start=start, end=end,
                          lat=lat, lon=lon)
        grd = gridmet.get_point_timeseries() / 25.4

        dt_index = pd.date_range(start, end)

        df.index = pd.to_datetime(df['DATE'])

        df = df.reindex(dt_index)

        try:
            df = df[['TMAX', 'TMIN', 'PRCP']]

            df['MX'] = df['TMAX'] / 10.
            df['MN'] = df['TMIN'] / 10.
            df['PP'] = df['PRCP'] / 10.
            df = df[['MX', 'MN', 'PP']]
            df['MM'] = (df['MX'] + df['MN']) / 2

            bc, start, end, kc = modified_blaney_criddle(df, lat, elev, mid_month=True)

            print('IWR est', bc['u'].sum())
            bc_pet = bc['ref_u'].sum()
            summary.loc[i, 'etbc'] = bc_pet

            grd['mday'] = ['{}-{}'.format(x.month, x.day) for x in grd.index]
            target_range = pd.date_range('2000-{}-{}'.format(start.month, start.day),
                                         '2000-{}-{}'.format(end.month, end.day))
            accept = ['{}-{}'.format(x.month, x.day) for x in target_range]
            grd['mask'] = [1 if d in accept else 0 for d in grd['mday']]
            grd = grd[grd['mask'] == 1]

            grd_etos = grd['pet'].resample('A').sum().mean()
            summary.loc[i, 'etos'] = grd_etos
        except Exception as e:
            print(row, e)

        print('{}: bc {:.3f} grd {:.3f}'.format(row['NAME'], bc_pet, grd_etos))

    gdf = gpd.GeoDataFrame(summary)
    gdf.geometry = summary['geo']
    gdf.drop(columns=['geo'], inplace=True)
    gdf = gdf.set_crs('epsg:4326')
    gdf.to_file(out_summary)
    gdf.to_csv(out_summary.replace('.shp', '.csv'))


def point_comparison_agrimet(station_dir, out_figs, out_shp):
    stations = load_stations()
    # print(stations['bfam'])
    station_files = [os.path.join(station_dir, x) for x in os.listdir(station_dir)]
    etbc, etos = [], []
    station_comp = {}
    # for f in station_files:
    #     sid = os.path.basename(f).split('.')[0]
    for sid in MT_STATIONS:
        # print(sid)
        meta = stations[sid]
        print(meta['properties'])
        coords = meta['geometry']['coordinates']
        geo = Point(coords)
        coord_rads = np.array(coords) * np.pi / 180
        elev = elevation_from_coordinate_ee(coords[1], coords[0])
        # df = pd.read_csv(f, index_col=0, parse_dates=True,
        #                  header=0, skiprows=[1, 2, 3]) ## removed infer_datetime_format
        ## load from website instead:
        df = Agrimet(station=sid, region=stations[sid]['properties']['region'],
                   start_date='2000-01-01', end_date='2023-12-31').fetch_met_data() ## correct daterange?
        df.columns = df.columns.droplevel([1, 2]) ## remove multiindex

        tmean, tmax, tmin, wind, rs, rh = df['MM'], df['MX'], df['MN'], df['UA'], df['SR'], df['TA']
        ra = extraterrestrial_r(df.index, lat=coord_rads[1], shape=[df.shape[0]])
        rso = calc_rso(ra, elev)
        rn = get_rn(tmean, rs=rs, lat=coord_rads[1], tmax=tmax, tmin=tmin, rh=rh, elevation=elev, rso=rso)
        df['ETOS'] = pm_fao56(tmean, wind, rs=rs, tmax=tmax, tmin=tmin, rh=rh, elevation=elev, rn=rn)
        df['ETRS'] = df['ETOS'] * 1.2

        try:
            bc, start, end, kc = modified_blaney_criddle(df, coords[1], elev)
        except IndexError:
            print(sid, 'failed')
            continue

        etbc_ = bc['ref_u'].sum()
        df['mday'] = ['{}-{}'.format(x.month, x.day) for x in df.index]
        target_range = pd.date_range('2000-{}-{}'.format(start.month, start.day),
                                     '2000-{}-{}'.format(end.month, end.day))
        accept = ['{}-{}'.format(x.month, x.day) for x in target_range]
        df.dropna(subset=['ETOS'], inplace=True)
        df['mask'] = [1 if d in accept else 0 for d in df['mday']]
        df = df[df['mask'] == 1]
        size = df.groupby(df.index.year).size()
        for i, s in size.items():
            if s < len(target_range) - 1:
                df = df.loc[[x for x in df.index if x.year != i]]

        years = len(list(set([d.year for d in df.index])))

        etos_ = df['ETOS'].resample('A').sum().mean() / 25.4
        etbc.append(etbc_)
        etos.append(etos_)
        diff_ = (etbc_ - etos_) / etos_
        station_comp[sid] = {'STAID': sid, 'etbc': etbc_, 'etos': etos_,
                             'no_yrs': years, 'diff': diff_ * 100., 'geo': geo}
        print(sid)

    df = pd.DataFrame(station_comp, columns=station_comp.keys()).T
    df = df.astype({'etbc': float,
                    'etos': float,
                    'diff': float})
    df = df.astype({'no_yrs': int})

    gdf = gpd.GeoDataFrame(df)

    gdf.geometry = gdf['geo']
    gdf.drop(columns=['geo'], inplace=True)
    gdf = gdf.set_crs('epsg:4326')
    gdf.to_file(out_shp)
    gdf.to_csv(out_shp.replace('.shp', '.csv'))

    xmin, xmax = min([min(etos), min(etbc)]) - 2, max([max(etos), max(etbc)]) + 2
    line = np.linspace(xmin, xmax)
    plt.scatter(etos, etbc)
    plt.plot(line, line)
    plt.xlim([xmin, xmax])
    plt.ylim([xmin, xmax])
    plt.xlabel('Penman-Monteith (FAO 56)')
    plt.ylabel('Blaney Criddle (SCS TR 21)')
    plt.suptitle('Seasonal Crop Reference Evapotranspiration Comparison [inches/season]')
    _filename = os.path.join(out_figs, 'FAO_56_SCSTR21_MBC_comparison1.png')
    plt.show()
    # plt.savefig(_filename)


def check_implementation(clim_db_loc, station='USC00242409', data_dir=None,
                         start='1970-01-01', end='2000-12-31'):

    _file = os.path.join(data_dir, '{}.csv'.format(station))
    df = pd.read_csv(_file)
    dt_index = pd.date_range(start, end)
    df.index = pd.to_datetime(df['DATE'])
    df = df.reindex(dt_index)

    lat = df.iloc[0]['LATITUDE']
    lon = df.iloc[0]['LONGITUDE']
    elev = elevation_from_coordinate_ee(lat, lon)
    # print()
    # print(elev)

    df = df[['TMAX', 'TMIN', 'PRCP']]

    df['MX'] = df['TMAX'] / 10.
    df['MN'] = df['TMIN'] / 10.
    df['PP'] = df['PRCP'] / 10.
    df = df[['MX', 'MN', 'PP']]
    df['MM'] = (df['MX'] + df['MN']) / 2

    bc, start, end, kc = modified_blaney_criddle_1(clim_db_loc, station[-4:],
                                                   lat_degrees=lat, elev=elev, fullmonth=False)
    # print(bc)
    print('Season: ', start, ' to ', end)

    print(bc['u'])
    bc_pet = bc['u'].sum()
    # print('Should match with first entry under "totals" in IWR: ', bc_pet)

    pass


def check_implementation_neh_ex(station='USC00242409', data_dir=None, iwr_table=None,
                         start='1970-01-01', end='2000-12-31'):

    _file = os.path.join(data_dir, '{}.csv'.format(station))
    df = pd.read_csv(_file)
    dt_index = pd.date_range(start, end)
    df.index = pd.to_datetime(df['DATE'])
    df = df.reindex(dt_index)

    lat = 39.7
    elev = 0

    df = df[['TMAX', 'TMIN', 'PRCP']]

    df['MX'] = df['TMAX'] / 10.
    df['MN'] = df['TMIN'] / 10.
    df['PP'] = df['PRCP'] / 10.
    df = df[['MX', 'MN', 'PP']]
    df['MM'] = (df['MX'] + df['MN']) / 2

    bc, start, end, kc = modified_blaney_criddle_neh_ex(df, lat, elev,
                                                 season_start='2000-04-24',
                                                 season_end='2000-10-25',
                                                 mid_month=True)
    # print(bc)

    print(bc['u'])
    bc_pet = bc['u'].sum()
    print('Should match with first column of IWR file: ',bc_pet)

    pass


def location_analysis():
    #### Investigating the geojson file Jack sent me.

    stations_davd = pd.read_csv('C:/Users/CND571/Documents/Data/mt_arm_iwr_stations.csv')
    stations_jack = gpd.read_file('C:/Users/CND571/Downloads/iwr_stations.geojson')

    print(stations_jack['station_no'])

    # print(stations_jack.iloc[41]) ## 41 is Dillon
    ## This line doesn't work, nuances make station names different.
    # stations_jack = stations_jack.loc[stations_jack['station_name'].isin(stations_davd['NAME'])]

    ## calling last four digits of station id
    stations_jack['ids'] = stations_jack['station_no']
    stations_davd['ids'] = stations_davd['STAID']
    for i in range(len(stations_jack)):
        stations_jack['ids'][i] = stations_jack['station_no'].iloc[i][2:]
    for i in range(len(stations_davd)):
        stations_davd['ids'][i] = stations_davd['STAID'].iloc[i][7:]

    stations_davd.sort_values(by=['ids'], inplace=True)
    stations_jack.sort_values(by=['ids'], inplace=True)

    stations_jack_bth = stations_jack.loc[stations_jack['ids'].isin(stations_davd['ids'])]
    stations_davd_bth = stations_davd.loc[stations_davd['ids'].isin(stations_jack['ids'])]

    stations_jack_dif = stations_jack.loc[~stations_jack['ids'].isin(stations_davd['ids'])]
    stations_davd_dif = stations_davd.loc[~stations_davd['ids'].isin(stations_jack['ids'])]

    # for i in range(len(stations_davd_bth)): ## THese are indeed the same.
    #     print(stations_jack_bth.ids.iloc[i], stations_jack_bth.station_name.iloc[i], ',',
    #           stations_davd_bth.ids.iloc[i], stations_davd_bth.NAME.iloc[i])

    # for i in range(len(stations_jack_dif)):  ## checking how different things are
    #     if i < len(stations_davd_dif):
    #         print(stations_jack_dif.ids.iloc[i], stations_jack_dif.station_name.iloc[i], ',',
    #               stations_davd_dif.ids.iloc[i], stations_davd_dif.NAME.iloc[i])
    #     else:
    #         print(stations_jack_dif.ids.iloc[i], stations_jack_dif.station_name.iloc[i])

    # print('stations_davd')
    # print(len(stations_davd))
    # print(stations_davd.PACKAGED.unique())
    # print(stations_davd.loc[stations_davd['PACKAGED']=='exists in db']) ## only 51 in database?
    # print(stations_davd.geometry)

    # print('stations_jack')
    # print(stations_jack.geometry) ## overlap of 119 with the same ID.

    jack_lon = np.asarray(stations_jack_bth.geometry.x)  ## dtype == geometry
    jack_lat = np.asarray(stations_jack_bth.geometry.y)

    davd_lon = np.asarray(stations_davd_bth.LON)
    davd_lat = np.asarray(stations_davd_bth.LAT)

    jack_lon_d = stations_jack_dif.geometry.x  ## dtype == geometry
    jack_lat_d = stations_jack_dif.geometry.y

    davd_lon_d = stations_davd_dif.LON
    davd_lat_d = stations_davd_dif.LAT

    # print(len(jack_lon), len(davd_lon))
    # print(len(jack_lon_d), len(davd_lon_d))

    # print(jack_lat[:5], davd_lat[:5])

    plt.figure()
    plt.title('132 stations: overlap between stations in David\'s dataset (162) and IWR database/MT ARM (180)')
    plt.scatter(davd_lon, davd_lat, label='David overlap')
    plt.scatter(jack_lon, jack_lat, label='Jack overlap')
    plt.scatter(davd_lon_d, davd_lat_d, marker='^', label='David unique')
    plt.scatter(jack_lon_d, jack_lat_d, marker='^', label='Jack unique')
    plt.legend()
    plt.grid()
    plt.show()

    # plt.figure()
    # plt.title("difference in longitude value")
    # plt.hist(davd_lon - jack_lon)
    # plt.show()
    #
    # plt.figure()
    # plt.title("difference in latitude value")
    # plt.hist(davd_lat - jack_lat)
    # plt.show()


if __name__ == '__main__':
    d = 'C:/Users/CND571/Documents/Data'
    if not os.path.exists(d):
        d = 'C:/Users/CND571/Documents/Data'

    r = os.path.join(d, 'comparison_data')
    shp_ = os.path.join(r, 'sweetgrass_fields_sample.shp')
    etof_ = os.path.join(r, 'sweetgrass_fields_etof')
    out_summary = os.path.join(r, 'sweetgrass_fields_comparison.shp')
    # field_comparison(shp_, etof_, out_summary)
    ## out_summary function not present. Nothing is done with resulting geopandas file.

    iwr_data_dir = os.path.join(d, 'from_ghcn')
    stations = os.path.join(d, 'mt_arm_iwr_stations.csv')
    comp = os.path.join(d, 'iwr_gridmet_comparison.shp')
    # point_comparison_iwr_stations(iwr_data_dir, stations, comp) ## this appears to have worked.
    ## about 23 stations have exceptions. how big of a problem is this?

    _dir = os.path.join(d, 'agrimet/mt_stations')
    fig_dir = os.path.join(d, 'agrimet/comparison_figures')
    out_shp = os.path.join(d, 'agrimet/shapefiles/comparison.shp')
    # point_comparison_agrimet(station_dir=_dir, out_figs=fig_dir, out_shp=out_shp) ## need to switch files.

    iwr_table = None
    iwr_clim_db_loc = 'C:/Users/CND571/Documents/IWR/Database/climate.db'
    start, end = '1971-01-01', '2000-12-31'
    # start, end = '1997-01-01', '2006-12-31'

    # Beaverhead, Dillon WMCE
    # IWR elev: 5230
    # IWR: 23.92, this gives 24.17
    # print(5230/3.28)
    check_implementation(iwr_clim_db_loc, 'USC00242409', iwr_data_dir, start=start, end=end)
    # Bighorn, Busby
    # IWR elev: 3430
    # IWR: 26.55, this gives 29.26
    # print(3430/3.28)
    check_implementation(iwr_clim_db_loc, 'USC00241297',  iwr_data_dir, start=start, end=end)
    # Blaine, Chinook
    # IWR elev: 2340
    # IWR: 27.86, this gives 26.31
    # print(2340/3.28)
    check_implementation(iwr_clim_db_loc, 'USC00241722', iwr_data_dir, start=start, end=end)
    # Broadwater, Townsend
    check_implementation(iwr_clim_db_loc, 'USC00248324', iwr_data_dir, start=start, end=end)


    ## This line replicates the Denver, CO example from NEH ch 2, appendix A
    # check_implementation_neh_ex('USC00242409', iwr_data_dir, iwr_table, start=start, end=end)

    # location_analysis()

    # openet_get_fields()
# ========================= EOF ====================================================================
