
import ee
import os

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
import datetime as dt

from from_swim_rs.etf_export import clustered_field_etf
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


def etof_quality_plots(county_select):
    for cnty in county_select:
        area = gpd.read_file('C:/Users/CND571/Documents/Data/statewide_irrigation_dataset_15FEB2024/{}.shp'
                             .format(cnty))
        area = area.to_crs("EPSG:5071")
        area.index = area['FID']
        area = area.drop(['FID'], axis=1)
        area['area_m2'] = area['geometry'].area

        ts = pd.DataFrame(index=area.index)
        for year in range(1987, 2024):
            data = pd.read_csv('C:/Users/CND571/Documents/Data/ssebop/{}/etf_{}_irr_ct.csv'
                               .format(cnty, year), index_col='FID')
            data = data * 900.0  # to account for 30x30m pixel size

            for i in data.columns:
                for j in data.index:
                    data.at[j, i] = data.at[j, i] / area['area_m2'][j]

            data = data.replace(0, np.nan)

            ts[year] = data.mean(axis=1)

            # if year == 2012:
            #     plt.figure()
            #     for i in data.index:
            #         plt.plot(data.loc[i], label=i)
            #     plt.grid()
            #
            #     bins = np.linspace(-0.05, 1.05, 12)
            #     plt.figure()
            #     plt.hist(ts[year], bins=bins)

        plt.figure()
        plt.title('{} County ({}) Average yearly field coverage'.format(COUNTIES[cnty], cnty))
        plt.plot(ts.mean(), label='percent coverage')  # Total
        plt.plot(ts.count()/len(ts), label='percent of fields included')
        # for i in area.index:
        #     plt.plot(ts.loc[i], label=i)  # By field
        plt.legend()
        plt.ylim(0, 1)
        plt.grid()

        # plt.figure()
        # plt.title('{} County ({}) average field coverage and std dev ({}/{} fields)'
        #           .format(COUNTIES[cnty], cnty, ts.mean(axis=1).count(), len(ts)))
        # plt.axis('off')
        # plt.subplot(121)
        # bins = np.linspace(-0.05, 1.05, 12)
        # plt.hist(ts.mean(axis=1), bins=bins)
        # plt.xlabel('field coverage')
        # plt.subplot(122)
        # plt.hist(ts.std(axis=1))
        # plt.xlabel('std dev')

    plt.show()


if __name__ == '__main__':

    # PART ZERO: SETUP
    d = '/media/research/IrrigationGIS/swim'  # ?
    if not os.path.exists(d):
        d = 'C:/Users/CND571/Documents/Data'

    # Counties already ran through part one:
    # county = '019'  # 37 fields, the smallest
    # county = '033'  # next smallest, 56 fields, good.
    # county = '041'  # 107 fields, good.
    # county = '055'  # 211 fields, good.
    # county = '023'  # 307 fields, good.
    county = '021'  # 402 fields

    # PART ONE: GOOGLE EARTH ENGINE DATA TO BUCKET
    # Check earth engine authorization before starting ee stuff.
    is_authorized()
    # bucket_ = 'wudr'  # David's bucket
    bucket_ = 'mt_cu_2024'  # Hannah's bucket, billing details needed.

    # ee asset
    # fields = 'users/dgketchum/fields/tongue_9MAY2023'
    fields = 'projects/ee-hehaugen/assets/SID_15FEB2024/{}'.format(county)

    mask = 'irr'
    chk = os.path.join(d, 'ssebop', county)  # local path to check for finished output files
    clustered_field_etf(fields, bucket_, debug=False, mask_type=mask, county=county)  # includes pixel count.

    # next, go download those files using below command in anaconda prompt
    # They take a while to show up. Maybe that's why eedl is preferred.
    # Can use command 'earthengine task list' to check in on progress (also in anaconda prompt)
    print('gcloud storage cp gs://mt_cu_2024/MT_CU_2024/{}/*.csv C:/Users/CND571/Documents/Data/ssebop/{}'
          .format(county, county))

    # # PART 1.5: LOOK AT DATA QUALITY
    # counties = ['019', '033', '041', '055', '023']
    # # counties = ['033']
    # etof_quality_plots(counties)

    # # PART TWO: FIX TIME SERIES DATA
    # # dtype = 'extracts'
    #
    # project_ws = os.path.join(d, 'ssebop')
    # tables = os.path.join(project_ws, 'landsat')
    #
    # yrs = [x for x in range(1987, 2024)]
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
    # # Plotting detour
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
