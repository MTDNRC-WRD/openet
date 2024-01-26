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
from utils.elevation import elevation_from_coordinate
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


def openet_get_fields(fields, start, end, et_too=False,
                      api_key='ZBXxCeBRsSgkeLvsROKVTDS1w9UV0xfOKyEJGTNcEEPT15DQsYfbB0uu1K9w'):
    """ Uses OpenET API multipolygon timeseries endpoint to get etof data given a Google Earth Engine asset.
    Prints the url, click link to download csv file; link lasts 5 minutes.
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
            "FID"
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


def field_comparison(shp, start, end, etof, out):
    # suggestion in memo: use gridmet for ET, then OpenET for crop coefficient (etof).
    gdf = gpd.read_file(shp)

    # loading in OpenET data from files. - do we not need the et, just the etof?
    file = pd.read_csv(etof, index_col=['FID', 'time'], date_format="%Y-%m-%d").sort_index()

    summary = deepcopy(gdf)
    summary['etos'] = [-99.99 for _ in summary['FID']]  # gridMET seasonal ET
    summary['etbc'] = [-99.99 for _ in summary['FID']]  # blaney criddle ET based on gridMET weather data
    summary['etof'] = [-99.99 for _ in summary['FID']]  # crop coefficient/etof from OpenET ensemble
    summary['etcu'] = [-99.99 for _ in summary['FID']]  # consumptive use calculated from gridMET ET
    summary['geo'] = [None for _ in summary['FID']]
    idx = max(summary.index.values) + 1

    for i, row in gdf.iterrows():
        # Loading in gridMET
        lon, lat = row.geometry.centroid.x, row.geometry.centroid.y
        elev = elevation_from_coordinate(lat, lon)

        gridmet = GridMet('pet', start=start, end=end,
                          lat=lat, lon=lon)  # pet is grass, etr is alfalfa. Alfalfa gives a much higher et estimate.
        grd = gridmet.get_point_timeseries()  # initialize df with first time series
        grd = grd.rename(columns={'pet': 'ETOS'})
        for var, _name in zip(['tmmn', 'tmmx', 'pr'], ['MN', 'MX', 'PP']):  # collect additional time series
            ts = GridMet(var, start=start, end=end,
                         lat=lat, lon=lon).get_point_timeseries()
            if 'tm' in var:
                ts -= 273.15

            grd[_name] = ts

        # Calculating bc seasonal ET
        grd['MM'] = (grd['MN'] + grd['MX']) / 2  # gridmet units are in mm and degrees celsius, I think.
        bc, start1, end1, kc = modified_blaney_criddle(grd, lat, elev, season_start='2000-05-09',
                                                       season_end='2000-09-19', mid_month=True)

        bc_pet = bc['u'].sum()

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
        df = file.loc[row['FID']]
        r_index = pd.date_range('2016-01-01', '2021-12-31', freq='D')
        df = df.reindex(r_index)
        df = df.interpolate()
        df['mday'] = ['{}-{}'.format(x.month, x.day) for x in df.index]
        target_range = pd.date_range('2016-05-09', '2016-09-19')

        accept = ['{}-{}'.format(x.month, x.day) for x in target_range]
        df['mask'] = [1 if d in accept else 0 for d in df['mday']]
        df = df[df['mask'] == 1]

        # getting effective precip and carryover based on county and irrigation type
        if row['mf'] == 0.675:  # Park County
            if row['IType'] == 'P':  # center pivot, low net irrigation
                eff_precip = 3.86
                carryover = 0.5
            else:  # other (ex: flood), high net irrigation
                eff_precip = 5.16
                carryover = 2
        elif row['mf'] == 0.494:  # Sweet Grass County
            if row['IType'] == 'P':
                eff_precip = 4.91
                carryover = 0.5
            else:
                eff_precip = 3.68
                carryover = 2
        else:
            print("Wrong county!")
            eff_precip = 0
            if row['IType'] == 'P':
                carryover = 0.5
            else:
                carryover = 2

        # calculating consumptive use
        # average seasonal ET times crop coefficient minus effective precip and carryover
        cu = (et_by_year.mean() * df['etof'].mean()) - eff_precip - carryover

        if isinstance(row['geometry'], MultiPolygon):  # multi-part fields
            first = True
            for g in row['geometry'].geoms:
                if first:
                    summary.loc[i, 'geo'] = g
                    summary.loc[i, 'etos'] = et_by_year.mean()
                    summary.loc[i, 'etbc'] = bc_pet
                    summary.loc[i, 'etof'] = df['etof'].mean()
                    summary.loc[i, 'etcu'] = cu

                    first = False
                else:  # Breaking up multipolygons; adding additional parts of fields to end of file
                    summary.loc[idx] = row
                    summary.loc[idx, 'geo'] = g
                    summary.loc[idx, 'etos'] = et_by_year.mean()
                    summary.loc[i, 'etbc'] = bc_pet
                    summary.loc[idx, 'etof'] = df['etof'].mean()
                    summary.loc[idx, 'etcu'] = cu

                    idx += 1
        else:
            summary.loc[i, 'geo'] = row['geometry']
            summary.loc[i, 'etos'] = et_by_year.mean()
            summary.loc[i, 'etbc'] = bc_pet
            summary.loc[i, 'etof'] = df['etof'].mean()
            summary.loc[i, 'etcu'] = cu
        # "progress bar"
        if i % 5 == 0:
            print(i, '/', idx)
    summary = gpd.GeoDataFrame(summary.drop(columns=['geometry']), geometry='geo')
    summary.to_file(out)
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
        elev = elevation_from_coordinate(coords[1], coords[0])
        # df = pd.read_csv(f, index_col=0, parse_dates=True,
        #                  header=0, skiprows=[1, 2, 3]) ## removed infer_datetime_format
        # load from website instead:
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
            bc, start, end, kc = modified_blaney_criddle(df, coords[1], elev, mid_month=True)
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
                         start='1970-01-01', end='2000-12-31', management_factor=None):

    _file = os.path.join(data_dir, '{}.csv'.format(station))
    df = pd.read_csv(_file)
    dt_index = pd.date_range(start, end)
    df.index = pd.to_datetime(df['DATE'])
    df = df.reindex(dt_index)

    lat = df.iloc[0]['LATITUDE']
    lon = df.iloc[0]['LONGITUDE']
    elev = elevation_from_coordinate(lat, lon)
    # print()
    # print(elev)

    df = df[['TMAX', 'TMIN', 'PRCP']]

    df['MX'] = df['TMAX'] / 10.
    df['MN'] = df['TMIN'] / 10.
    df['PP'] = df['PRCP'] / 10.
    df = df[['MX', 'MN', 'PP']]
    df['MM'] = (df['MX'] + df['MN']) / 2

    # print(bc)
    print()
    print('Using IWR database:')
    bc1, start1, end1, kc1 = modified_blaney_criddle_1(clim_db_loc, station[-4:],
                                                       lat_degrees=lat, elev=elev, fullmonth=False)

    print('Season: ', start1.date(), ' to ', end1.date())
    if management_factor:
        print('kc = ', bc1['kc'].mean() * management_factor)
    # print(bc1['u'])
    bc_pet1 = bc1['u'].sum()
    print(bc_pet1)
    # print('Should match with first entry under "totals" in IWR: ', bc_pet)

    print()
    print('Using daily data:')
    bc, start, end, kc = modified_blaney_criddle(df, lat_degrees=lat, elev=elev,
                                                 season_start=start1, season_end=end1, mid_month=True)
    # bc, start, end, kc = modified_blaney_criddle(df, lat_degrees=lat, elev=elev, mid_month=True)

    print('Season: ', start.date(), ' to ', end.date())
    if management_factor:
        print('kc = ', bc['kc'].mean() * management_factor)
    # print(bc['u'])
    print(bc['u'].sum())

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

    bc, start, end, kc, ep = modified_blaney_criddle_neh_ex(df, lat, elev,
                                                 season_start='2000-04-24',
                                                 season_end='2000-10-25',
                                                 mid_month=True)
    # print(bc)

    print(bc['u'])
    bc_pet = bc['u'].sum()
    print('Should match with first column of IWR file: ',bc_pet)

    pass


def plot_field_comparison(field_comp_result,data_dir):
    # Loading shapefile for field_comparison results
    gdf = gpd.read_file(field_comp_result)
    print("gdf columns ", gdf.columns)

    # Determine crop coefficient based on management factor
    kc = {0.675: 0.729, 0.494: 0.534}
    gdf['dnrc_kc'] = [kc[x] for x in gdf['mf']]

    # print(len(gdf['etos'].unique()))  # 18, for the number of gridMET stations in study area.
    # print(gdf['etos'].unique())

    # Color bar limits for plotting crop coefficient and seasonal ET
    vals_kc = [0.33, 1]  # possible range: 0-1
    vals_et = [0, 25]  # units: in

    # Loading in Ketchcum 2022 memo data
    r = os.path.join(data_dir, 'comparison_data')
    file1 = os.path.join(r, 'sweetgrass_fields_comparison_et_cu.shp')
    gdf1 = gpd.read_file(file1)
    print("gdf1 columns ", gdf1.columns)
    file2 = os.path.join(r, 'sweetgrass_fields_comparison_etof.shp')
    gdf2 = gpd.read_file(file2)
    print("gdf2 columns ", gdf2.columns)
    # recovering gridMET seasonal ET from David's file
    etos_gdf1 = gdf1['et'] / gdf2['etof']

    print()
    print('Min and Max gridMET ET:')
    print(gdf['etos'].min(), gdf['etos'].max())
    print(etos_gdf1.min(), etos_gdf1.max())
    print()
    print('Min and Max OpenET crop coefficients:')
    print(gdf['etof'].min(), gdf['etof'].max())
    print(gdf2['etof'].min(), gdf2['etof'].max())

    # # Field plots, DNRC vs Hannah's new data
    # fig, axes = plt.subplots(2, 2, figsize=(20, 8))
    # axes[0, 0].set_title('(S-3) OpenET Crop Coefficient')
    # gdf.plot(ax=axes[0, 0], column='etof', cmap='viridis_r', legend=True,
    #          legend_kwds={"shrink": 0.8}, vmin=vals_kc[0], vmax=vals_kc[1])
    #
    # axes[1, 0].set_title('(S-4) OpenET Crop Coefficient')
    # gdf.plot(ax=axes[1, 0], column='dnrc_kc', cmap='viridis_r', legend=True,
    #          legend_kwds={"shrink": 0.8}, vmin=vals_kc[0], vmax=vals_kc[1])
    #
    # axes[0, 1].set_title('(S-5) OpenET Consumptive Use [in]')
    # gdf.plot(ax=axes[0, 1], column='etcu', cmap='RdYlBu', legend=True,
    #          legend_kwds={"shrink": 0.8},  vmin=vals_et[0], vmax=vals_et[1])
    #
    # axes[1, 1].set_title('(S-6) DNRC Consumptive Use [in]')
    # gdf.plot(ax=axes[1, 1], column='dnrc_cu', cmap='RdYlBu', legend=True,
    #          legend_kwds={"shrink": 0.8}, vmin=vals_et[0], vmax=vals_et[1])
    #
    # axes[0, 0].xaxis.set_tick_params(labelbottom=False)
    # axes[0, 0].yaxis.set_tick_params(labelleft=False)
    # axes[1, 0].xaxis.set_tick_params(labelbottom=False)
    # axes[1, 0].yaxis.set_tick_params(labelleft=False)
    # axes[0, 1].xaxis.set_tick_params(labelbottom=False)
    # axes[0, 1].yaxis.set_tick_params(labelleft=False)
    # axes[1, 1].xaxis.set_tick_params(labelbottom=False)
    # axes[1, 1].yaxis.set_tick_params(labelleft=False)

    # Histograms, DNRC vs Hannah's new data
    # fig1, axes1 = plt.subplots(2, 2, figsize=(20, 8))
    # axes1[0, 0].set_title('(S-3) OpenET Crop Coefficient')
    # gdf['etof'].plot.hist(ax=axes1[0, 0], range=(0.33, 1.0), bins=20)
    #
    # axes1[1, 0].set_title('(S-4) OpenET Crop Coefficient')
    # gdf['dnrc_kc'].plot.hist(ax=axes1[1, 0], range=(0.33, 1.0), bins=20)
    #
    # axes1[0, 1].set_title('(S-5) OpenET Consumptive Use [in]')
    # gdf['etcu'].plot.hist(ax=axes1[0, 1], range=(1.5, 21.5), bins=20)
    #
    # axes1[1, 1].set_title('(S-6) DNRC Consumptive Use [in]')
    # gdf['dnrc_cu'].plot.hist(ax=axes1[1, 1], range=(1.5, 21.5), bins=20)

    # # Ketchum 2022 memo data vs Hannah's new data, just for consumptive use
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8))
    axes2[0].set_title('(S-5) Hannah OpenET Consumptive Use [in]')
    gdf.plot(ax=axes2[0], column='etcu', cmap='RdYlBu', legend=True,
              legend_kwds={"shrink": 0.8}, vmin=vals_et[0], vmax=vals_et[1])

    axes2[1].set_title('(S-5) David OpenET Consumptive Use [in]')
    gdf1.plot(ax=axes2[1], column='openet_cu', cmap='RdYlBu', legend=True,
              legend_kwds={"shrink": 0.8}, vmin=vals_et[0], vmax=vals_et[1])

    axes2[0].xaxis.set_tick_params(labelbottom=False)
    axes2[0].yaxis.set_tick_params(labelleft=False)
    axes2[1].xaxis.set_tick_params(labelbottom=False)
    axes2[1].yaxis.set_tick_params(labelleft=False)

    # Histograms, Ketchum 2022 memo data vs Hannah's new data
    fig3, axes3 = plt.subplots(2, 3, figsize=(25, 8))
    axes3[0, 0].set_title('Hannah OpenET Consumptive Use [in]')
    gdf['etcu'].plot.hist(ax=axes3[0, 0], range=(1.5, 21.5))

    axes3[1, 0].set_title('David OpenET Consumptive Use [in]')
    gdf1['openet_cu'].plot.hist(ax=axes3[1, 0], range=(1.5, 21.5))

    axes3[0, 1].set_title('Hannah gridMET reference ET [in]')
    gdf['etos'].plot.hist(ax=axes3[0, 1], range=(23, 28))

    axes3[1, 1].set_title('David gridMET reference ET [in]')
    etos_gdf1.plot.hist(ax=axes3[1, 1], range=(23, 28))

    axes3[0, 2].set_title('Hannah OpenET crop coefficient')
    gdf['etof'].plot.hist(ax=axes3[0, 2], range=(0.33, 1.0))

    axes3[1, 2].set_title('David OpenET crop coefficient')
    gdf2['etof'].plot.hist(ax=axes3[1, 2], range=(0.33, 1.0))

    plt.show()

    pass


if __name__ == '__main__':
    d = 'C:/Users/CND571/Documents/Data'
    if not os.path.exists(d):
        d = 'C:/Users/CND571/Documents/Data'

    gee_asset = "projects/ee-hehaugen/assets/sweetgrass_fields_sample"
    start_pos = "2016-01-01"
    end_pos = "2021-12-31"
    # openet_get_fields(gee_asset, start_pos, end_pos)

    r = os.path.join(d, 'comparison_data')
    shp_ = os.path.join(r, 'sweetgrass_fields_sample.shp')
    # etof_ = os.path.join(r, 'sweetgrass_fields_etof')
    # out_summary = os.path.join(r, 'sweetgrass_fields_comparison.shp')
    etof_ = 'C:/Users/CND571/Downloads/ensemble_monthly_etof.csv'
    out_summary = 'C:/Users/CND571/Downloads/fields_comparison_01262024.shp'
    # field_comparison(shp_, start_pos, end_pos, etof_, out_summary)  # Takes about 25 minutes to run.

    plot_field_comparison(out_summary, d)

    iwr_data_dir = os.path.join(d, 'from_ghcn')
    stations = os.path.join(d, 'mt_arm_iwr_stations.csv')
    comp = os.path.join(d, 'iwr_gridmet_comparison.shp')
    # point_comparison_iwr_stations(iwr_data_dir, stations, comp)

    _dir = os.path.join(d, 'agrimet/mt_stations')
    fig_dir = os.path.join(d, 'agrimet/comparison_figures')
    out_shp = os.path.join(d, 'agrimet/shapefiles/comparison.shp')
    # point_comparison_agrimet(station_dir=_dir, out_figs=fig_dir, out_shp=out_shp) ## need to switch files?

    iwr_table = None
    iwr_clim_db_loc = 'C:/Users/CND571/Documents/IWR/Database/climate.db'
    start, end = '1971-01-01', '2000-12-31'  # time period used in IWR db
    # start, end = '1997-01-01', '2006-12-31'  # time period used in memo

    # # Checking individual stations' relationship to IWR results. (County, Station)

    # # Beaverhead, Dillon WMCE
    # # IWR elev: 5230
    # # print(5230/3.28)
    # check_implementation(iwr_clim_db_loc, 'USC00242409', iwr_data_dir, start=start, end=end)
    # # Bighorn, Busby
    # # IWR elev: 3430
    # # print(3430/3.28)
    # check_implementation(iwr_clim_db_loc, 'USC00241297',  iwr_data_dir, start=start, end=end)
    # # Blaine, Chinook
    # # IWR elev: 2340
    # # print(2340/3.28)
    # check_implementation(iwr_clim_db_loc, 'USC00241722', iwr_data_dir, start=start, end=end)
    # # Broadwater, Townsend
    # check_implementation(iwr_clim_db_loc, 'USC00248324', iwr_data_dir, start=start, end=end)

    # # IWR stations used in Ketchum 2022 memo:
    # # Park, Livingston (12S, not FAA?)
    # check_implementation(iwr_clim_db_loc, 'USC00245080', iwr_data_dir, start=start, end=end, management_factor=0.675)
    # # Sweet Grass, Big Timber
    # check_implementation(iwr_clim_db_loc, 'USC00240780', iwr_data_dir, start=start, end=end, management_factor=0.494)

    # This line replicates the Denver, CO example from NEH ch 2, appendix A
    # check_implementation_neh_ex('USC00242409', iwr_data_dir, iwr_table, start=start, end=end)

# ========================= EOF ====================================================================
