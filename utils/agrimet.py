# =============================================================================================
# Copyright 2017 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================================
from __future__ import print_function, absolute_import

import io
import json
from pprint import pprint
from copy import deepcopy

import pandas as pd
import requests
from requests.compat import urlencode, OrderedDict
from datetime import datetime
from fiona import collection
from fiona.crs import from_epsg
from geopy.distance import geodesic
from pandas import read_table, to_datetime, date_range, to_numeric, DataFrame

import matplotlib.pyplot as plt
import datetime as dt

STATION_INFO_URL = 'https://www.usbr.gov/pn/agrimet/agrimetmap/usbr_map.json'
# AGRIMET_MET_REQ_SCRIPT_PN = 'https://www.usbr.gov/pn-bin/agrimet.pl' ## appears to be broken
AGRIMET_MET_REQ_SCRIPT_PN = 'https://www.usbr.gov/pn-bin/agrimet.pl'
AGRIMET_CROP_REQ_SCRIPT_PN = 'https://www.usbr.gov/pn/agrimet/chart/{}{}et.txt'
AGRIMET_MET_REQ_SCRIPT_GP = 'https://www.usbr.gov/gp-bin/agrimet_archives.pl'
AGRIMET_CROP_REQ_SCRIPT_GP = 'https://www.usbr.gov/gp-bin/et_summaries.pl?station={}&year={}&submit2=++Submit++'
# in km
EARTH_RADIUS = 6371.

WEATHER_PARAMETRS_UNCONVERTED = [('DATETIME', 'Date - [YYYY-MM-DD]'),
                                 ('ET', 'Evapotranspiration Kimberly-Penman - [in]'),
                                 ('MM', 'Mean Daily Air Temperature - [F]'),
                                 ('MN', 'Minimum Daily Air Temperature - [F]'),
                                 ('MX', 'Maximum Daily Air Temperature - [F]'),
                                 ('PC', 'Accumulated Precipitation Since Recharge/Reset - [in]'),
                                 ('PP', 'Daily (24 hour) Precipitation - [in]'),
                                 ('PU', 'Accumulated Water Year Precipitation - [in]'),
                                 ('SR', 'Daily Global Solar Radiation - [langleys]'),
                                 ('TA', 'Mean Daily Humidity - [%]'),
                                 ('TG', 'Growing Degree Days - [base 50F]'),
                                 ('YM', 'Mean Daily Dewpoint Temperature - [F]'),
                                 ('UA', 'Daily Average Wind Speed - [mph]'),
                                 ('UD', 'Daily Average Wind Direction - [deg az]'),
                                 ('WG', 'Daily Peak Wind Gust - [mph]'),
                                 ('WR', 'Daily Wind Run - [miles]'),
                                 ]

WEATHER_PARAMETRS = [('DATETIME', 'Date', '[YYYY-MM-DD]'),
                     ('ET', 'Modified Penman ETr', '[mm]'),
                     ('MM', 'Mean Daily Air Temperature', '[C]'),
                     ('MN', 'Minimum Daily Air Temperature', '[C]'),
                     ('MX', 'Maximum Daily Air Temperature', '[C]'),
                     ('PC', 'Accumulated Precipitation Since Recharge/Reset', '[mm]'),
                     ('PP', 'Daily (24 hour) Precipitation', '[mm]'),
                     ('PU', 'Accumulated Water Year Precipitation', '[mm]'),
                     ('SR', 'Daily Global Solar Radiation', '[MJ m-2]'),
                     ('TA', 'Mean Daily Humidity', '[%]'),
                     ('TG', 'Growing Degree Days', '[base 50F]'),
                     ('YM', 'Mean Daily Dewpoint Temperature', '[C]'),
                     ('UA', 'Daily Average Wind Speed', '[m sec-1]'),
                     ('UD', 'Daily Average Wind Direction - [deg az]', '[deg az]'),
                     ('WG', 'Daily Peak Wind Gust', '[m sec-1]'),
                     ('WR', 'Daily Wind Run', '[m]')]

TARGET_COLUMNS = ['{a}_et', '{a}_etos', '{a}_etrs', '{a}_mm', '{a}_mn',
                  '{a}_mx', '{a}_pp', '{a}_pu', '{a}_sr', '{a}_ta', '{a}_tg',
                  '{a}_ua', '{a}_ud', '{a}_wg', '{a}_wr', '{a}_ym']

STANDARD_PARAMS = ['et', 'mm', 'mn',
                   'mx', 'pp', 'pu', 'sr', 'ta', 'tg',
                   'ua', 'ud', 'wg', 'wr', 'ym']

MT_STATIONS = ['covm',
               'crsm',
               'rdbm',
               'bozm',
               'sigm',
               'tosm',
               'hvmt',
               'gfmt',
               'rbym',
               'dlnm',
               'hrlm',
               'matm',
               'glgm',
               'drlm',
               'brtm',
               'brgm',
               'bfam',
               'bftm',
               'lmmm',
               'trfm',
               'umhm',
               'wssm',
               'jvwm',
               'svwm',
               'mwsm',
               'ulmt',
               'comt',
               'vlmt',
               'gidm',
               'shbm',
               'crnm']


class Agrimet(object):
    def __init__(self, start_date=None, end_date=None, station=None,
                 interval=None, lat=None, lon=None, sat_image=None,
                 write_stations=False, region=None):

        self.station_info_url = STATION_INFO_URL
        self.station = station
        self.distance_from_station = None
        self.station_coords = None
        self.distances = None
        self.region = region

        self.empty_df = True

        if not station and not write_stations:
            if not lat and not sat_image:
                raise ValueError('Must initialize agrimet with a station, '
                                 'an Image, or some coordinates.')
            if not sat_image:
                self.station = self.find_closest_station(lat, lon)
            else:

                lat = (sat_image.corner_ll_lat_product + sat_image.corner_ul_lat_product) / 2
                lon = (sat_image.corner_ll_lon_product + sat_image.corner_lr_lon_product) / 2
                self.station = self.find_closest_station(lat, lon)

        if station:
            self.find_station_coords()

        self.interval = interval

        if start_date and end_date:
            self.start = datetime.strptime(start_date, '%Y-%m-%d')
            self.end = datetime.strptime(end_date, '%Y-%m-%d')
            self.today = datetime.now()
            self.start_index = (self.today - self.start).days - 1

        self.rank = 0

    @property
    def params(self):
        return urlencode(OrderedDict([
            ('cbtt', self.station),
            ('interval', self.interval),
            ('format', 2),
            ('back', self.start_index)
        ]))

    def find_station_coords(self):
        station_data = load_stations()
        sta_ = station_data[self.station]
        self.station_coords = sta_['geometry']['coordinates'][1], sta_['geometry']['coordinates'][0]

    def find_closest_station(self, target_lat, target_lon):
        """ The two-argument inverse tangent function.
        :param station_data:
        :param target_lat:
        :param target_lon:
        :return:
        """
        distances = {}
        station_coords = {}
        station_data = load_stations()
        for feat in station_data['features']:
            stn_crds = feat['geometry']['coordinates']
            stn_site_id = feat['properties']['siteid']
            lat_stn, lon_stn = stn_crds[1], stn_crds[0]
            dist = geodesic((target_lat, target_lon), (lat_stn, lon_stn)).km
            distances[stn_site_id] = dist
            station_coords[stn_site_id] = lat_stn, lon_stn
        k = min(distances, key=distances.get)
        self.distances = sorted(list(distances.items()), key=lambda x: x[1])
        self.distance_from_station = distances[k]
        self.station_coords = station_coords
        return k

    def fetch_met_data(self, return_raw=False, out_csv_file=None, long_names=False):

        # if self.region == 'pnro': ## not working anymore
        #     url = '{}?{}'.format(AGRIMET_MET_REQ_SCRIPT_PN, self.params)
        #     print(url)
        #     r = requests.get(url)
        #     txt = r.text.split('\n')
        #     s_idx, e_idx = txt.index('BEGIN DATA\r'), txt.index('END DATA\r')

        if self.region == 'pnro':
            pairs = ','.join(['{} {}'.format(self.station.upper(), x.upper()) for x in STANDARD_PARAMS])
            url = "https://www.usbr.gov/pn-bin/webarccsv.pl?parameter={0}&syer={1}&smnth={2}&sdy={3}&" \
                  "eyer={4}&emnth={5}&edy={6}&format=2".format(pairs,
                                                               self.start.year,
                                                               self.start.month,
                                                               self.start.day,
                                                               self.end.year,
                                                               self.end.month,
                                                               self.end.day)
            r = requests.get(url)
            txt = r.text.split('\n')
            s_idx, e_idx = txt.index('BEGIN DATA'), txt.index('END DATA')

        if self.region == 'great_plains' or self.region == 'gpro':
            pairs = ','.join(['{} {}'.format(self.station.upper(), x.upper()) for x in STANDARD_PARAMS])
            url = "https://www.usbr.gov/gp-bin/webarccsv.pl?parameter={0}&syer={1}&smnth={2}&sdy={3}&" \
                  "eyer={4}&emnth={5}&edy={6}&format=2".format(pairs,
                                                               self.start.year,
                                                               self.start.month,
                                                               self.start.day,
                                                               self.end.year,
                                                               self.end.month,
                                                               self.end.day)
            r = requests.get(url)
            txt = r.text.split('\r\n')
            s_idx, e_idx = txt.index('BEGIN DATA'), txt.index('END DATA')

        content = txt[s_idx + 1: e_idx]
        names = [c.strip() for c in content[0].split(',')]
        data = {name: [x.split(',')[i].strip() for x in content[1:]] for i, name in enumerate(names)}
        df = DataFrame(data)
        rename = dict((c, str(c).split(' ')[1]) if c != 'DATE' else (c, c) for c in df.columns)

        cols = df.columns[df.dtypes.eq('object')]
        df[cols] = df[cols].apply(to_numeric, errors='coerce')
        df.rename(columns=rename, inplace=True)

        df.index = date_range(self.start, periods=df.shape[0], name='DateTime')
        df = df[to_datetime(self.start): to_datetime(self.end)]

        df.drop(columns='DATE', inplace=True)

        df = df[df[[v for k, v in rename.items() if v != 'DATE']].notna()]
        if df.shape[0] > 3:
            self.empty_df = False

        if return_raw:
            return df

        df = self._reformat_dataframe(df)

        if out_csv_file:
            df.to_csv(path_or_buf=out_csv_file)

        return df

    def fetch_crop_data(self, out_csv_file=None):

        if not self.start.year == self.end.year:
            raise ValueError('Must choose one year for crop water use reports.')

        if self.region == 'pn':
            # this may need a recursive scheme to go down list of closest stations
            two_dig_yr = format(int(str(self.start.year)[-2:]), '02d')
            url = AGRIMET_CROP_REQ_SCRIPT_PN.format(self.station, two_dig_yr)
            raw_df = read_table(url, skip_blank_lines=True, skiprows=[3], index_col=[0],
                                header=2, engine='python', delim_whitespace=True)
            raw_df = raw_df.iloc[1:, :]
            try:
                start_str = raw_df.first_valid_index().replace('/', '')
            except AttributeError:
                start_str = format(int(raw_df.first_valid_index()), '03d')

        if self.region == 'gp':
            raw_df, start_str = self.get_gp_crop()

        et_summary_start = datetime.strptime('{}{}'.format(self.start.year, start_str), '%Y%m%d')
        raw_df.index = date_range(et_summary_start, periods=raw_df.shape[0])
        idx = date_range(self.start, end=self.end)

        raw_df.replace('--', '0.0', inplace=True)
        cols = raw_df.columns.values.tolist()
        try:
            raw_df = raw_df.astype(float)
        except ValueError:
            raw_df = (raw_df.drop(cols, axis=1).join(raw_df[cols].apply(to_numeric, errors='coerce')))

        raw_df.interpolate(inplace=True)

        reformed_data = raw_df.reindex(idx, fill_value=0.0)
        cols = reformed_data.columns.values.tolist()
        for c in cols:
            reformed_data[c] *= 25.4

        if out_csv_file:
            reformed_data.to_csv(path_or_buf=out_csv_file)

        return reformed_data

    def get_gp_crop(self):
        url = AGRIMET_CROP_REQ_SCRIPT_GP.format(self.station, self.start.year)
        data = requests.get(url).content
        str_data = str(data, 'utf-8')
        file = open('data.txt', 'w')
        file.write(str_data)
        raw_df = read_table('data.txt', skip_blank_lines=True, skiprows=[0, 1, 2, 3, 5, 6], index_col=[0],
                            engine='python', delim_whitespace=True, error_bad_lines=False)

        raw_df = raw_df.iloc[2:, :]
        start_str = format(int(raw_df.first_valid_index()), '03d')
        return raw_df, start_str

    def _reformat_dataframe(self, df):

        old_cols = df.columns.values.tolist()
        head_1 = []
        head_2 = []
        head_3 = []
        for x in old_cols:
            end = x.replace('{}_'.format(self.station), '')
            for j, k, l in WEATHER_PARAMETRS:
                if end.upper() == j.upper():
                    head_1.append(j.upper())
                    head_2.append(k)
                    head_3.append(l)
                    break
        if len(list(df.columns)) > len(head_1):
            drop = [c for c in df.columns if c not in head_1]
            print('dropping special parameters dataframe columns....')
            pprint(drop)
            df = deepcopy(df[head_1])

        df.columns = [head_1, head_2, head_3]

        for i, col in enumerate(head_1, start=0):
            try:
                # convert to standard units
                if col in ['ET', 'ETRS', 'ETOS', 'PC', 'PP', 'PU']:
                    # in to mm
                    df[col] *= 25.4
                if col in ['MN', 'MX', 'MM', 'YM']:
                    # F to C
                    df[col] = (df[col] - 32) * 5 / 9
                if col in ['UA', 'WG']:
                    # mph to m s-1
                    df[col] *= 0.44704
                if col == 'WR':
                    # mi to m
                    df['WR'] *= 1609.34
                if col == 'SR':
                    # Langleys to W m-2
                    df['SR'] /= 23.900574
            except KeyError:
                head_1.remove(head_1[i])
                head_2.remove(head_2[i])
                head_3.remove(head_3[i])

        df.columns = [head_1, head_2, head_3]

        return df

    @staticmethod
    def write_agrimet_sation_shp(json_data, epsg, out):
        agri_schema = {'geometry': 'Point',
                       'properties': {
                           'program': 'str',
                           'url': 'str',
                           'siteid': 'str',
                           'title': 'str',
                           'state': 'str',
                           'type': 'str',
                           'region': 'str',
                           'install': 'str'}}

        cord_ref = from_epsg(epsg)
        shp_driver = 'ESRI Shapefile'

        with collection(out, mode='w', driver=shp_driver, schema=agri_schema,
                        crs=cord_ref) as output:
            for rec in json_data['features']:
                try:
                    output.write({'geometry': {'type': 'Point',
                                               'coordinates':
                                                   (rec['geometry']['coordinates'][0],
                                                    rec['geometry']['coordinates'][1])},
                                  'properties': {
                                      'program': rec['properties']['program'],
                                      'url': rec['properties']['url'],
                                      'siteid': rec['properties']['siteid'],
                                      'title': rec['properties']['title'],
                                      'state': rec['properties']['state'],
                                      'type': rec['properties']['type'],
                                      'region': rec['properties']['region'],
                                      'install': rec['properties']['install']}})
                except KeyError:
                    pass


def load_stations():
    r = requests.get(STATION_INFO_URL)
    stations = json.loads(r.text)
    stations = stations['features']
    stations = {s['properties']['siteid']: s for s in stations}
    return stations

#
# def load_stations_1():
#     # # Uses JSON file dowloaded from BOR website,
#     stations = json.load(open('C:/Users/CND571/Downloads/AgriMet_station_list.json'))
#     stations = stations['features']
#     stations = {s['properties']['StationID']: s for s in stations}
#     return stations


def agrimet_data(save=False):
    """ Download Agrimet data, with option to save as csv or use as pd dataframe. """
    ams = stn_lists(mn=False)
    am_idx = pd.date_range("1984-01-01", end="2023-12-31", freq="D")
    all_am = pd.DataFrame(index=am_idx, columns=ams)

    stations = load_stations()
    for i in ams:
        # i = i.upper()
        stn = Agrimet(station=i, region=stations[i]['properties']['StationOffice'],
                      start_date='1984-01-01', end_date='2023-12-31')
        data = stn.fetch_met_data()
        data.columns = data.columns.droplevel([1, 2])
        all_am[i] = data['ET'] / 25.4  # Kimberly-Penman ET

    if save:
        all_am.to_csv('C:/Users/CND571/Documents/Data/all_agrimet_daily_etr_in_through2023.csv')
    else:
        return all_am


def mesonet_data(save=False):
    """ Download Mesonet data, with option to save as csv or use as pd dataframe. """
    mns = stn_lists(am=False)

    mn_idx = pd.date_range("2017-01-01", end="2023-12-31", freq="D")
    all_mn = pd.DataFrame(index=mn_idx, columns=mns)

    for i in mns:
        if i not in ['blmglsou', 'wrsround']:  # THese two do not have data available?
            print(i)
            mn_url = (
                'https://mesonet.climate.umt.edu/api/v2/derived/daily/?crop=corn&high=86&low=50&alpha=0.23&'
                'na_info=false&rm_na=false&premade=true&wide=true&keep=false&units=us&type=csv&tz=America%2FDenver&'
                'simple_datetime=false&time=daily&end_time=2024-01-01T00%3A00%3A00&start_time=2017-01-01T00%3A00%3A00&'
                'level=1&stations={}&elements=etr'.format(i))
            mn_station1 = pd.read_csv(mn_url, index_col='datetime')
            mn_station1.index = [j[:10] for j in mn_station1.index]
            mn_station1.index = pd.to_datetime(mn_station1.index)
            all_mn[i] = mn_station1['Reference ET (a=0.23) [in]']  # Grass reference

    if save:
        all_mn.to_csv('C:/Users/CND571/Documents/Data/all_mesonet_daily_etr_in_through2023.csv')
    else:
        return all_mn


def stn_lists(am=True, mn=True, plot=False):
    """ Return lists of station IDs for MT Agrimet and Mesonet. """
    am_list = 0
    mn_list = 0
    ams = 0
    mns = 0

    if am:
        am_list = pd.read_csv('C:/Users/CND571/Documents/Data/AgriMet_station_list.csv')
        am_list = am_list[am_list['StationState'] == 'MT']
        am_list['StationInstall'] = pd.to_datetime(am_list['StationInstall'])
        ams = list(am_list['StationID'])
    if mn:
        mn_list = pd.read_csv('C:/Users/CND571/Documents/Data/mesonet_station_list.csv')
        mn_list['date_installed'] = pd.to_datetime(mn_list['date_installed'])
        mn_list = mn_list[mn_list['date_installed'] < pd.to_datetime('01/01/2021')]  # >3 years of data available
        mns = list(mn_list['station'])
        # Can't access data for these 2 stations?
        mns.remove('blmglsou')
        mns.remove('wrsround')

    if am and mn:
        if plot:  # Plot locations of Agrimet and Mesonet stations
            close_am = ['COVM', 'CRSM', 'MWSM']
            plt.figure()
            plt.scatter(am_list['StationLongitude'], am_list['StationLatitude'], label='Agrimet')
            plt.scatter(mn_list['longitude'], mn_list['latitude'], label='Mesonet')
            for i in close_am:
                plt.scatter(am_list[am_list['StationID'] == i]['StationLongitude'],
                            am_list[am_list['StationID'] == i]['StationLatitude'], color='tab:green')
            plt.legend()
        return ams, mns
    elif am:
        return ams
    elif mn:
        return mns
    else:
        return 0


def am_mn_comp_plots(am_d, mn_d, am_m, mn_m):
    # Time series comparison of 3 pairs of "collocated" Arigmet/Mesonet stations
    plt.figure(figsize=(15, 10))

    plt.subplot(311)
    plt.title('Corvallis')
    plt.plot(am_d.index, am_d['COVM'], label='Agrimet')
    plt.plot(mn_d.index, mn_d['corvalli'], label='Mesonet')
    plt.plot(am_m.index - dt.timedelta(days=15), am_m['COVM'], label='Agrimet (Monthly)')
    plt.plot(mn_m.index - dt.timedelta(days=15), mn_m['corvalli'], label='Mesonet (Monthly)')
    plt.ylabel('ETr (in)')
    # plt.xlabel('Date')
    plt.grid()
    plt.xlim(dt.date(year=2017, month=1, day=1), dt.date(year=2024, month=1, day=1))
    plt.legend()

    plt.subplot(312)
    plt.title('Kalispell')
    plt.plot(am_d.index, am_d['CRSM'], label='Agrimet')
    plt.plot(mn_d.index, mn_d['kalispel'], label='Mesonet')
    plt.plot(am_m.index - dt.timedelta(days=15), am_m['CRSM'], label='Agrimet (Monthly)')
    plt.plot(mn_m.index - dt.timedelta(days=15), mn_m['kalispel'], label='Mesonet (Monthly)')
    plt.ylabel('ETr (in)')
    # plt.xlabel('Date')
    plt.grid()
    plt.xlim(dt.date(year=2017, month=1, day=1), dt.date(year=2024, month=1, day=1))
    # plt.ylim(0, 5)
    plt.legend()

    plt.subplot(313)
    plt.title('Moccasin')
    plt.plot(am_d.index, am_d['MWSM'], label='Agrimet')
    plt.plot(mn_d.index, mn_d['moccasin'], label='Mesonet')
    plt.plot(am_m.index - dt.timedelta(days=15), am_m['MWSM'], label='Agrimet (Monthly)')
    plt.plot(mn_m.index - dt.timedelta(days=15), mn_m['moccasin'], label='Mesonet (Monthly)')
    plt.ylabel('ETr (in)')
    # plt.xlabel('Date')
    plt.grid()
    plt.xlim(dt.date(year=2017, month=1, day=1), dt.date(year=2024, month=1, day=1))
    plt.legend()

    plt.tight_layout()

    # Difference between Agrimet and Mesonet monthly averages.
    plt.figure()

    plt.subplot(211)
    plt.hlines(1, dt.date(year=2019, month=1, day=1), dt.date(year=2024, month=1, day=1), 'k')
    plt.plot(am_m.index - dt.timedelta(days=15), (am_m['COVM']) / mn_m['corvalli'], label='Corvallis')
    plt.plot(am_m.index - dt.timedelta(days=15), (am_m['CRSM']) / mn_m['kalispel'], label='Kalispell')
    plt.plot(am_m.index - dt.timedelta(days=15), (am_m['MWSM']) / mn_m['moccasin'], label='Moccasin')
    plt.xlim(dt.date(year=2019, month=1, day=1), dt.date(year=2024, month=1, day=1))
    plt.ylim(0, 2)
    plt.ylabel('Agrimet/Mesonet')
    plt.grid()

    plt.subplot(212)
    plt.hlines(0, dt.date(year=2019, month=1, day=1), dt.date(year=2024, month=1, day=1), 'k')
    plt.plot(am_m.index - dt.timedelta(days=15), (am_m['COVM']) - mn_m['corvalli'], label='Corvallis')
    plt.plot(am_m.index - dt.timedelta(days=15), (am_m['CRSM']) - mn_m['kalispel'], label='Kalispell')
    plt.plot(am_m.index - dt.timedelta(days=15), (am_m['MWSM']) - mn_m['moccasin'], label='Moccasin')
    plt.xlim(dt.date(year=2019, month=1, day=1), dt.date(year=2024, month=1, day=1))
    plt.ylim(-0.1, 0.2)
    plt.ylabel('Agrimet minus Mesonet (in)')
    plt.grid()

    # plt.show()


def am_quality_plots(am_d, am_m):
    # Time series comparison of 3 pairs of "collocated" Arigmet/Mesonet stations

    for i in range(8):
        plt.figure(figsize=(15, 12))
        for j in range(4):
            idx = 4*i + j
            if idx < len(am_d.columns):
                stn = am_d.columns[idx]
                plt.subplot(4, 1, j+1)
                plt.title(stn)
                plt.plot(am_d.index, am_d[stn], label='Agrimet')
                plt.plot(am_m.index - dt.timedelta(days=15), am_m[stn], label='Agrimet (Monthly)')
                plt.ylabel('ETr (in)')
                plt.grid()
                # plt.xlim(dt.date(year=2017, month=1, day=1), dt.date(year=2024, month=1, day=1))
                if j == 0:
                    plt.legend()
                print("{}: {:.1f}% of {:.1f} years of data".format(stn, 100 * am_d[stn].count() / len(am_d[stn][am_d[stn].first_valid_index():]), len(am_d[stn][am_d[stn].first_valid_index():]) / 365))
                # print(stn, ":", (am_d[stn] > 1).sum(), am_d[stn].count(),
                #       "{:.1f}%".format(100*(am_d[stn] > 1).sum()/am_d[stn].count()))
        plt.tight_layout()


def mn_quality_plots(mn_d, mn_m):
    # Time series comparison of 3 pairs of "collocated" Arigmet/Mesonet stations

    for i in range(9):
        plt.figure(figsize=(15, 12))
        for j in range(8):
            idx = 8*i + j
            if idx < len(mn_d.columns):
                stn = mn_d.columns[idx]
                plt.subplot(4, 2, j+1)
                plt.title(stn)
                plt.plot(mn_d.index, mn_d[stn], label='Mesonet')
                plt.plot(mn_m.index - dt.timedelta(days=15), mn_m[stn], label='Mesonet (Monthly)')
                if j % 2 == 0:
                    plt.ylabel('ETr (in)')
                plt.grid()
                # plt.xlim(dt.date(year=2017, month=1, day=1), dt.date(year=2024, month=1, day=1))
                if j == 1:
                    plt.legend()
                print("{}: {:.1f}% of {:.1f} years of data".format(stn, 100 * mn_d[stn].count() / len(mn_d[stn][mn_d[stn].first_valid_index():]), len(mn_d[stn][mn_d[stn].first_valid_index():]) / 365))
        plt.tight_layout()


if __name__ == '__main__':

    # Investigating differences between different station metadata sources. (PN region vs GP region.)
    stations = load_stations()
    print(len(stations))
    print(stations.keys())
    print(stations['abei'])
    # stations1 = load_stations()
    # print(len(stations1))
    # print(stations1.keys())
    # print(stations1['AFTY'])
    # for k in stations1.keys():
    #     k = k.lower()
    #     if k not in stations.keys():
    #         print("Oh no! {}".format(k))

    mt = 0
    other = 0
    for k in stations.keys():
        if stations[k]['properties']['state'] == 'MT':
            mt += 1
        else:
            other += 1
        # print(stations[k]['properties']['state'])
    print('Old one: {} MT and {} other'.format(mt, other))

    # mt = 0
    # other = 0
    # for k in stations1.keys():
    #     if stations1[k]['properties']['StationState'] == 'MT':
    #         mt += 1
    #     else:
    #         other += 1
    #     # print(stations[k]['properties']['state'])
    # print('New one: {} MT and {} other'.format(mt, other))

    mt_st = {i for i in stations.keys() if stations[i]['properties']['state'] == 'MT'}
    # mt_st1 = {i.lower() for i in stations1.keys() if stations1[i]['properties']['StationState'] == 'MT'}
    print(mt_st)
    # print(mt_st1)
    # print("{} overlapping stations".format(len(mt_st.intersection(mt_st1))))
    # print("Only in PN", mt_st.difference(mt_st1))  # New in GP region in 2019, except bomt (removed in 2000).
    # print("Only in GP", mt_st1.difference(mt_st))  # New in PN region since 2015.

    # Why is there such a big difference between these two sets of stations?
    # PN region and GP region have different station directories!!! WHY?!

    # Finding average length of period of record for Agrimet stations

    # installs = pd.DataFrame(stations.keys(), columns=['ID'])
    # i = 0
    # for k in stations.keys():
    #     install = stations[k]['properties']['StationInstall']
    #     if len(install) > 0:
    #         installs.at[i, 'Install'] = dt.datetime.strptime(install, '%m/%d/%Y')
    #         installs.at[i, 'POR'] = dt.date.today() - installs.at[i, 'Install'].date()
    #     i += 1
    # print(installs)
    # print(installs['POR'].mean())
    # print(8819 / 365)  # 24 years on 7/22/24
    #
    # # # Loading data
    # # agrimet_data(save=True)
    # am_data = pd.read_csv('C:/Users/CND571/Documents/Data/all_agrimet_daily_etr_in_through2023.csv',
    #                       index_col='Unnamed: 0')
    # am_data.index = pd.to_datetime(am_data.index)
    # am_data = am_data.mask(am_data > 1)  # data values greater than 1 assumed to be errors. Not many at most stations.
    # # Highest incident of erroneously high ETr values was VLMT with 3.8% (3.5/9 years affected)
    # # Some stations still afected by noisy/missing data, but these are also often isolated years,
    # # and can be removed later if desired.
    # # print(am_data)
    #
    # # mesonet_data(save=True)
    # mn_data = pd.read_csv('C:/Users/CND571/Documents/Data/all_mesonet_daily_etr_in_through2023.csv',
    #                       index_col='Unnamed: 0')
    # mn_data.index = pd.to_datetime(mn_data.index)
    # # Mesonet has a lot shorter period of record (earliest start date is 2017?), and much more missing data.
    # # print(mn_data)
    #
    # # Monthly averages
    # am_mnth = am_data.resample('ME').mean()
    # mn_mnth = mn_data.resample('ME').mean()
    #
    # # # Comparison of 3 pairs of "collocated" Arigmet/Mesonet stations
    # am_mn_comp_plots(am_data, mn_data, am_mnth, mn_mnth)
    #
    # # Data quality checking
    # # am_quality_plots(am_data, am_mnth)
    # # mn_quality_plots(mn_data, mn_mnth)
    #
    # plt.show()

# ========================= EOF ====================================================================
