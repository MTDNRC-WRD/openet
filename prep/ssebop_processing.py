
import ee
import os
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

from from_swim_rs.etf_export import clustered_field_etf
from from_swim_rs.etf_export import clustered_field_etf_1
from from_swim_rs.landsat_sensing import landsat_time_series_multipolygon


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


def is_authorized():
    try:
        ee.Initialize()
        print('Authorized')
    except Exception as e:
        print('You are not authorized: {}'.format(e))
        exit(1)
    return None


if __name__ == '__main__':

    # PART ZERO: SETUP
    d = '/media/research/IrrigationGIS/swim'  # ?
    if not os.path.exists(d):
        d = 'C:/Users/CND571/Documents/Data'

    # county = '077'  # too big? >600 fields
    county = '019'  # 37 fields? # files made for this one. Is that affecting the test?
    # county = '075'  # 219 fields  # lots of files are still failing...
    # county = '051'  # just one missing here...
    # county = '033'  # next smallest

    # PART ONE: GOOGLE EARTH ENGINE DATA TO BUCKET
    # Check earth engine authorization before starting ee stuff.
    is_authorized()
    # bucket_ = 'wudr'  # David's bucket
    bucket_ = 'mt_cu_2024'  # Hannah's bucket, billing details needed.

    # ee asset
    # fields = 'users/dgketchum/fields/tongue_9MAY2023'
    fields = 'projects/ee-hehaugen/assets/SID_15FEB2024/{}'.format(county)

    mask = 'irr'
    chk = os.path.join(d, 'ssebop', county)
    # clustered_field_etf(fields, bucket_, debug=False, mask_type=mask, check_dir=chk, county=county)
    clustered_field_etf_1(fields, bucket_, mask_type=mask, county=county)

    # now, go download those files using below command in anaconda prompt
    # They take a while to show up. Maybe that's why eedl is preferred.
    # Can use command 'earthengine task list' to check in on progress (also in anaconda prompt)
    print('gcloud storage cp gs://wudr/MT_CU_2024/{}/*.csv C:/Users/CND571/Documents/Data/ssebop/{}'
          .format(county, county))

    # # PART TWO: FIX TIME SERIES DATA
    # # dtype = 'extracts'
    #
    # project_ws = os.path.join(d, 'ssebop')
    # tables = os.path.join(project_ws, 'landsat')
    #
    # yrs = [x for x in range(2000, 2021)]
    # # shp = os.path.join(project_ws, 'gis', '{}_fields.shp'.format(project))
    # shp = 'C:/Users/CND571/Documents/Data/statewide_irrigation_dataset_15FEB2024/{}.shp'.format(county)
    #
    # # directory with exported ee csv files
    # # ee_data = os.path.join(project_ws, 'landsat', dtype, 'etf', 'irr')
    # ee_data = os.path.join(project_ws, county)
    #
    # # Out files
    # src = os.path.join(tables, '{}_{}_{}.csv'.format(county, 'etf', 'irr'))
    # src_ct = os.path.join(tables, '{}_{}_{}_ct.csv'.format(county, 'etf', 'irr'))
    #
    # # landsat_time_series_multipolygon(shp, ee_data, yrs, src, src_ct)
    #
    # # Plotting detour!
    #
    # # Load files
    # data = pd.read_csv(src, index_col='Unnamed: 0', date_format='%Y-%m-%d')
    # data_ct = pd.read_csv(src_ct, index_col='Unnamed: 0', date_format='%Y-%m-%d')
    # # data.index = data.index.to_datetime()
    #
    # print(data)
    # print(data_ct)
    #
    # j = 7
    #
    # capture_dates = data_ct.index[data_ct[data_ct.columns[j]] == 1]
    # capture_vals = data[data_ct.columns[j]][data_ct[data_ct.columns[j]] == 1]
    # print(len(data))
    # print(len(capture_dates))
    #
    # plt.figure()
    #
    # plt.plot(data.index, data[data.columns[j]], label=data.columns[j][-2:])
    # plt.plot(capture_dates, capture_vals, 'o')
    # # for i in data.columns:
    # #     plt.plot(data.index, data[i], label=i[-2:])
    # plt.legend()
    # # plt.xticks(np.arange(2000, 2021))
    # plt.xticks(pd.date_range('2000-01-01', '2021-01-01', freq='YS'))
    # plt.grid()
    # plt.hlines([0.1, 0.2], dt.date(year=2000, month=1, day=1), dt.date(year=2021, month=1, day=1), 'k')
    # plt.xlim(dt.date(year=2000, month=1, day=1), dt.date(year=2021, month=1, day=1))
    #
    # plt.show()

    # # EXTRA STUFF
    # # Creating file folders to store etf data by county
    # for i in COUNTIES.keys():
    #     file_path = 'C:/Users/CND571/Documents/Data/ssebop/{}'.format(i)
    #     if os.path.exists(file_path):
    #         continue
    #     else:
    #         os.mkdir(file_path)

# ========================= EOF ====================================================================
