
import ee
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import geopandas as gpd
import numpy as np
import datetime as dt

from tqdm import tqdm

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

SPLIT = ['047', '111', '099', '081', '073', '105', '031']


def is_authorized():
    try:
        ee.Initialize()
        print('Authorized')
    except Exception as e:
        print('You are not authorized: {}'.format(e))
        exit(1)
    return None


def county_ee_data_to_bucket(cnty, ee_dir, bucket_, loc_dir=None):
    """ Download ssebop etf data from Google Earth Engine (GEE) to a Google cloud storage bucket.

    Once this code is run, processing in GEE can take a very long time before files show up in the bucket.
    Progress on tasks can be checked by running 'earthengine task list' in anaconda prompt.
    Once files show up in bucket, they can be downloaded using this command (replace brackets w/ FIPS code):
    gcloud storage cp gs://mt_cu_2024/MT_CU_2024/{}/*.csv C:/Users/CND571/Documents/Data/ssebop/{}

    cnty: str, FIPS code for single MT county
    ee_dir: str, path to GEE directory where field geometry assets are stored
    bucket_: str, name of Google cloud storage bucket to export files to
    loc_dir: str, local directory where data is eventually intended to be stored, used for cancelling
    calculations if data is already stored locally.
    """

    # Check earth engine authorization before starting ee stuff.
    is_authorized()  # which project needs to be connected? Same project that bucket lives in?

    # Split processing in half for large counties. These must have two split field files uploaded to GEE,
    # named 'XXXa' and 'XXXb', with XXX representing the county's FIPS code.
    if cnty in SPLIT:
        print("Splitting request")
        for i in ['a', 'b']:
            countyi = cnty + i
            # ee asset
            fields = ee_dir + countyi
            mask = 'irr'
            chk = os.path.join(loc_dir, 'ssebop', cnty)  # local path to check for finished output files

            clustered_field_etf(fields, bucket_, debug=False, mask_type=mask, check_dir=chk, county=countyi)
    else:
        # ee asset
        # fields = 'users/dgketchum/fields/tongue_9MAY2023'
        fields = ee_dir + cnty

        mask = 'irr'
        chk = os.path.join(loc_dir, 'ssebop', cnty)  # local path to check for finished output files

        clustered_field_etf(fields, bucket_, debug=False, mask_type=mask, check_dir=chk, county=cnty)


def check_fields_vertical(cnty):
    print('preparing data/plots...')
    for year in range(1987, 2024):
        ar_data = pd.read_csv('C:/Users/CND571/Documents/Data/ssebop/{}/etf_{}_irr_area.csv'
                              .format(cnty, year), index_col='FID')
        ar_data = ar_data.replace(0, np.nan)  # so zeros won't affect calcs
        if year == 1987:
            ar_ts = pd.DataFrame(index=ar_data.index)
        ar_ts[year] = ar_data.mean(axis=1)  # average fractional area for each field for the year
    ar_ts = ar_ts.sort_index()

    for year in range(1987, 2024):
        ef_data = pd.read_csv('C:/Users/CND571/Documents/Data/ssebop/{}/etf_{}_irr.csv'
                              .format(cnty, year), index_col='FID')
        ef_data = ef_data.replace(0, np.nan)  # so zeros won't affect calcs
        if year == 1987:
            ef_ts = pd.DataFrame(index=ef_data.index)
        ef_ts[year] = ef_data.mean(axis=1)  # average etf for each field for the year
    ef_ts = ef_ts.sort_index()

    fields = [x[-2:] for x in ar_ts.index]

    plt.figure()
    plt.suptitle("{} County ({})".format(COUNTIES[cnty], cnty))
    plt.subplot(121)
    plt.title("Field fraction info")
    plt.imshow(ar_ts, zorder=5, vmin=0, vmax=1)
    plt.xticks(ticks=np.arange(37), labels=ar_ts.columns, rotation='vertical')
    plt.yticks(ticks=np.arange(len(ar_ts)), labels=fields)
    plt.grid(zorder=0)
    plt.xlabel("year")
    plt.ylabel("field")
    plt.colorbar(shrink=0.7)

    plt.subplot(122)
    plt.title("Etof data")
    plt.imshow(ef_ts, zorder=5, vmin=0, vmax=1)
    plt.xticks(ticks=np.arange(37), labels=ar_ts.columns, rotation='vertical')
    plt.yticks(ticks=np.arange(len(ar_ts)), labels=fields)
    plt.grid(zorder=0)
    plt.xlabel("year")
    plt.ylabel("field")
    plt.colorbar(shrink=0.7)


def check_fields_horizontal(cnty):
    print('preparing data/plots...')
    for year in range(1987, 2024):
        ar_data = pd.read_csv('C:/Users/CND571/Documents/Data/ssebop/{}/etf_{}_irr_area.csv'
                              .format(cnty, year), index_col='FID')
        ar_data = ar_data.replace(0, np.nan)  # so zeros won't affect calcs
        if year == 1987:
            ar_ts = pd.DataFrame(index=ar_data.index)
        ar_ts[year] = ar_data.mean(axis=1)  # average fractional area for each field for the year
    ar_ts = ar_ts.sort_index()

    for year in range(1987, 2024):
        ef_data = pd.read_csv('C:/Users/CND571/Documents/Data/ssebop/{}/etf_{}_irr.csv'
                              .format(cnty, year), index_col='FID')
        ef_data = ef_data.replace(0, np.nan)  # so zeros won't affect calcs
        if year == 1987:
            ef_ts = pd.DataFrame(index=ef_data.index)
        ef_ts[year] = ef_data.mean(axis=1)  # average etf for each field for the year
    ef_ts = ef_ts.sort_index()

    fields = [x[-3:] for x in ar_ts.index]

    plt.figure()
    # plt.figure(figsize=(10, 5))
    plt.suptitle("{} County ({})".format(COUNTIES[cnty], cnty))
    plt.subplot(211)
    plt.title("Field fraction info")
    plt.imshow(ar_ts.T, zorder=5, vmin=0, vmax=1)
    plt.yticks(ticks=np.arange(0, 37, 3), labels=ar_ts.columns[::3])
    plt.yticks(ticks=np.arange(0, 37), minor=True)
    plt.xticks(ticks=np.arange(0, len(ar_ts), 3), labels=fields[::3], rotation='vertical')
    plt.xticks(ticks=np.arange(0, len(ar_ts)), minor=True)
    plt.grid(which='both', zorder=0)
    plt.ylabel("year")
    plt.xlabel("field")
    # plt.colorbar()
    # plt.colorbar(shrink=0.7)

    plt.subplot(212)
    plt.title("Etof data")
    plt.imshow(ef_ts.T, zorder=5, vmin=0, vmax=1)
    plt.yticks(ticks=np.arange(0, 37, 3), labels=ar_ts.columns[::3])
    plt.yticks(ticks=np.arange(0, 37), minor=True)
    plt.xticks(ticks=np.arange(0, len(ar_ts), 3), labels=fields[::3], rotation='vertical')
    plt.xticks(ticks=np.arange(0, len(ar_ts)), minor=True)
    plt.grid(which='both', zorder=0)
    plt.ylabel("year")
    plt.xlabel("field")
    # plt.colorbar()
    # plt.colorbar(shrink=0.7)


def area_fraction_file(cnty, sid_dir, etf_dir):
    """ Take a pixel count output file from gee and create a fractional area file in same dir.
    Also combines and relocates multipart files for large counties (>2000 fields)
    params:
    cnty: 3-digit FIPS code for desired MT county
    sid_dir: the directory containing SID shapefiles split by county and named with FIPS code
    etf_dir: the directory containing gee export files. Files generated using function from etf_export.py
    """
    if cnty in SPLIT:
        # Combine files and relocate to main directory
        for year in range(1987, 2024):
            # Build file names
            in_a = os.path.join(etf_dir, '{}a'.format(cnty), 'etf_{}_irr.csv'.format(year))
            in_b = os.path.join(etf_dir, '{}b'.format(cnty), 'etf_{}_irr.csv'.format(year))
            in_a_ct = os.path.join(etf_dir, '{}a'.format(cnty), 'etf_{}_irr_ct.csv'.format(year))
            in_b_ct = os.path.join(etf_dir, '{}b'.format(cnty), 'etf_{}_irr_ct.csv'.format(year))
            out = os.path.join(etf_dir, cnty, 'etf_{}_irr.csv'.format(year))
            out_ct = os.path.join(etf_dir, cnty, 'etf_{}_irr_ct.csv'.format(year))
            # Check that this has not been done yet
            if not os.path.isfile(out):
                # load in data
                in_a = pd.read_csv(in_a, index_col='FID')
                in_b = pd.read_csv(in_b, index_col='FID')
                # combine dataframes
                out_dat = pd.concat([in_a, in_b])
                # export files
                out_dat.to_csv(out)
            if not os.path.isfile(out_ct):
                # Apparently, searching for one and assuming the other is there doesn't work.
                # load in data
                in_a_ct = pd.read_csv(in_a_ct, index_col='FID')
                in_b_ct = pd.read_csv(in_b_ct, index_col='FID')
                # combine dataframes
                out_dat_ct = pd.concat([in_a_ct, in_b_ct])
                # export files
                out_dat_ct.to_csv(out_ct)

    # Now create fractional area file
    sid_file = os.path.join(sid_dir, '{}.shp'.format(cnty))
    area = gpd.read_file(sid_file)
    area = area.to_crs("EPSG:5071")
    area.index = area['FID']
    area = area.drop(['FID'], axis=1)
    area['area_m2'] = area['geometry'].area
    # print("{} Tiny Fields!".format(len(area[area['area_m2'] < 1])))
    # print(area[area['area_m2'] < 1]['area_m2'])

    for year in range(1987, 2024):
        outfile = os.path.join(etf_dir, cnty, 'etf_{}_irr_area.csv'.format(year))
        if not os.path.isfile(outfile):
            print(year)
            infile = os.path.join(etf_dir, cnty, 'etf_{}_irr_ct.csv'.format(year))
            data = pd.read_csv(infile, index_col='FID')
            data = data * 900.0  # to account for 30x30m pixel size
            for i in data.columns:
                for j in data.index:
                    if area['area_m2'][j] > 1:  # Why are there zero values?
                        data.at[j, i] = data.at[j, i] / area['area_m2'][j]
                    else:
                        data.at[j, i] = 0
            data = data.replace(0, np.nan)
            data.to_csv(outfile)


def data_quality_plot(county_select):
    """ Plot some stuff, handles many counties. Assuming county_select is ordered by increasing number of fields
    in each county, lighter colors indicate smaller numbers of fields.
    county_select: list of str specifying FIPS codes of counties to plot data from.
    """
    print('preparing data/plots...')
    alldata = []
    numfields = 0
    missfieldslo = 0
    missfieldshi = 0
    for cnty in county_select:
        for year in range(1987, 2024):
            data = pd.read_csv('C:/Users/CND571/Documents/Data/ssebop/{}/etf_{}_irr_area.csv'
                               .format(cnty, year), index_col='FID')
            data = data.replace(0, np.nan)  # so zeros won't affect calcs
            if year == 1987:
                ts = pd.DataFrame(index=data.index)
            ts[year] = data.mean(axis=1)  # average fractional area for each field for the year
        alldata.append(ts)
        numfields += len(ts)
        missfieldslo += len(ts) - ts.count().max()
        missfieldshi += len(ts) - ts.count().min()

    alletf = []
    missfieldslo_1 = 0
    missfieldshi_1 = 0
    for cnty in county_select:
        for year in range(1987, 2024):
            data = pd.read_csv('C:/Users/CND571/Documents/Data/ssebop/{}/etf_{}_irr.csv'
                               .format(cnty, year), index_col='FID')
            data = data.replace(0, np.nan)  # so zeros won't affect calcs
            if year == 1987:
                ts = pd.DataFrame(index=data.index)
            ts[year] = data.mean(axis=1)  # average etf for each field for the year
        alletf.append(ts)
        missfieldslo_1 += len(ts) - ts.count().max()
        missfieldshi_1 += len(ts) - ts.count().min()

    # Plotting!
    cmap = mpl.colormaps['viridis_r']
    colors = cmap(np.linspace(0, 1, len(county_select)))
    lss = ['solid', 'dashed', 'dotted', 'dashdot']
    ms = ['o', '^', 's', '*']

    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.title("fields with non-zero pixel coverage - count")
    for i in range(len(county_select)):
        cnty = county_select[i]
        plt.plot(alldata[i].count() / len(alldata[i]), c=colors[i], ls=lss[i % 4],
                 label="{}: {} ({} fields)".format(cnty, COUNTIES[cnty], len(alldata[i])))
    plt.grid()
    # plt.legend()
    plt.ylim(0, 1)
    plt.ylabel('fraction of fields with data')

    # plt.subplot(222)
    # plt.title("number of missing fields")
    # for i in range(len(county_select)):
    #     cnty = county_select[i]
    #     plt.plot(len(alldata[i]) - alldata[i].count(),
    #              label="{}: {} ({} fields)".format(cnty, COUNTIES[cnty], len(alldata[i])))
    # plt.grid()
    # plt.legend()
    # plt.ylabel('number of fields with no data')

    # plt.subplot(222)
    # plt.title("fields with non-zero pixel coverage - sum")
    # for i in range(len(county_select)):
    #     cnty = county_select[i]
    #     plt.plot(alldata[i].count() / len(alldata[i]),
    #              label="{}: {} ({} fields)".format(cnty, COUNTIES[cnty], len(alletf[i])))
    # plt.grid()
    # plt.legend()
    # plt.ylim(0, 1)
    # plt.ylabel('fraction of fields with data')

    plt.subplot(222)
    plt.title("Yearly average field etf")
    for i in range(len(county_select)):
        cnty = county_select[i]
        plt.plot(alletf[i].mean(), c=colors[i], ls=lss[i % 4],
                 label="{}: {} ({} fields)".format(cnty, COUNTIES[cnty], len(alletf[i])))
    plt.grid()
    plt.legend()
    plt.ylabel('average etf value of fields in county')
    plt.ylim(0, 1)

    # plot of average field fraction covered by data
    plt.subplot(223)
    plt.title('Average yearly field coverage')
    for i in range(len(county_select)):
        cnty = county_select[i]
        plt.plot(alldata[i].mean(), c=colors[i], ls=lss[i % 4],
                 label="{}: {} ({} fields)".format(cnty, COUNTIES[cnty], len(alldata[i])))
    # plt.legend()
    plt.ylabel('average fraction of field covered by etof data')
    plt.ylim(0, 1)
    plt.grid()

    # Scatterplot of field number vs coverage
    plt.subplot(224)
    plt.title('field coverage vs number of fields')
    plt.plot([0, 1], [0, 1], 'k')
    for i in range(len(county_select)):
        cnty = county_select[i]
        plt.scatter(alldata[i].mean(), alldata[i].count() / len(alldata[i]), color=colors[i], marker=ms[i % 4],
                    label="{}: {} ({} fields)".format(cnty, COUNTIES[cnty], len(alldata[i])), zorder=i+5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('average fraction of field covered by etof data')
    plt.ylabel('fraction of fields with data')
    plt.legend()
    plt.grid(zorder=0)

    # # Scatterplot of area vs etof field number
    # plt.subplot(224)
    # plt.title('number of fields')
    # plt.plot([0, 1], [0, 1], 'k')
    # for i in range(len(county_select)):
    #     cnty = county_select[i]
    #     plt.scatter(alletf[i].count() / len(alletf[i]), alldata[i].count() / len(alldata[i]),
    #                 label="{}: {} ({} fields)".format(cnty, COUNTIES[cnty], len(alldata[i])), zorder=i+5)
    # # plt.xlim(0, 1)
    # # plt.ylim(0, 1)
    # plt.xlim(0.6, 1)
    # plt.ylim(0.6, 1)
    # plt.xlabel('fraction of fields with non-zero etof data')
    # plt.ylabel('fraction of fields with non-zero pixel count data')
    # plt.legend()
    # plt.grid(zorder=0)

    # plt.subplot(224)
    # plt.title("fields with non-zero etof data")
    # for i in range(len(county_select)):
    #     cnty = county_select[i]
    #     plt.plot(alletf[i].count() / len(alletf[i]),
    #              label="{}: {} ({} fields)".format(cnty, COUNTIES[cnty], len(alletf[i])))
    # plt.grid()
    # plt.legend()
    # plt.ylim(0, 1)
    # plt.ylabel('fraction of fields with data')

    print()
    print("number of fields represented: {}".format(numfields))
    print("percent of statewide fields represented: {:.2f}".format((numfields / 51404) * 100))
    print()
    print('pixel count data:')
    print("number of missing fields: {}/{}".format(missfieldslo, missfieldshi))
    print("missing fields as percent of total considered fields: {:.2f}/{:.2f}"
          .format((missfieldslo / numfields) * 100, (missfieldshi / numfields) * 100))
    # print()
    # print('etof data:')
    # print("number of missing fields: {}/{}".format(missfieldslo_1, missfieldshi_1))
    # print("missing fields as percent of total considered fields: {:.2f}/{:.2f}"
    #       .format((missfieldslo_1 / numfields) * 100, (missfieldshi_1 / numfields) * 100))
    # print("number of counties represented: {}".format(len(county_select)))
    # print("percent of counties represented: {:.2f}".format((len(county_select) / 56) * 100))

    # plt.show()


def etf_ts_plot(cnty):
    etf = pd.read_csv("C:/Users/CND571/Documents/Data/ssebop/landsat/{}_etf_irr.csv".format(cnty),
                      index_col='Unnamed: 0')
    obs_dates = pd.read_csv("C:/Users/CND571/Documents/Data/ssebop/landsat/{}_etf_irr_ct.csv".format(cnty),
                            index_col='Unnamed: 0')
    etf.index = pd.to_datetime(etf.index)
    obs_dates.index = pd.to_datetime(obs_dates.index)

    for year in range(2000, 2021):
        data = pd.read_csv('C:/Users/CND571/Documents/Data/ssebop/{}/etf_{}_irr_area.csv'
                           .format(cnty, year), index_col='FID')
        data = data.replace(0, np.nan)  # so zeros won't affect calcs
        if year == 2000:
            ts = pd.DataFrame(index=data.index)
        ts[year] = data.mean(axis=1)  # average fractional area for each field for the year

    # area = pd.DataFrame(index=etf.index)
    area = etf.copy()
    # area_x = pd.date_range(etf.index[0], etf.index[-1], freq='YS')
    # area_y = ts.loc[field]
    # one_ts = ts.loc[field]
    # one_ts.index = area_x
    for i in area.columns:
        one_ts = ts.loc[i]
        for j in one_ts.index:
            area[i].where(area.index.year != j, one_ts.loc[j], inplace=True)
            # area[area.index.year == j][i] = one_ts.loc[j]
    print()

        # etf_1 = pd.read_csv("C:/Users/CND571/Documents/Data/ssebop/landsat/{}_etf_irr_1.csv".format(cnty),
    #                     index_col='Unnamed: 0')
    # obs_dates_1 = pd.read_csv("C:/Users/CND571/Documents/Data/ssebop/landsat/{}_etf_irr_ct_1.csv".format(cnty),
    #                           index_col='Unnamed: 0')
    # etf_1.index = pd.to_datetime(etf_1.index)
    # obs_dates_1.index = pd.to_datetime(obs_dates_1.index)

    # etf_2 = pd.read_csv("C:/Users/CND571/Documents/Data/ssebop/landsat/{}_etf_irr_2.csv".format(cnty),
    #                     index_col='Unnamed: 0')
    # obs_dates_2 = pd.read_csv("C:/Users/CND571/Documents/Data/ssebop/landsat/{}_etf_irr_ct_2.csv".format(cnty),
    #                           index_col='Unnamed: 0')
    # etf_2.index = pd.to_datetime(etf_2.index)
    # obs_dates_2.index = pd.to_datetime(obs_dates_2.index)

    field = etf.columns[5]  # 5 looks good, 0 looks worse

    obs_etf = etf[obs_dates[field] == 1][field]
    print(len(obs_etf), len(obs_etf[obs_etf < 0.1]))

    etf_gs = etf[(etf.index.month >= 4) & (etf.index.month <= 10)]
    # obs_etf_gs = etf_gs[obs_dates[field] == 1][field]

    # obs_etf_1 = etf_1[obs_dates_1[field] == 1][field]
    # print(len(obs_etf_1), len(obs_etf_1[obs_etf_1 < 0.1]))

    # obs_etf_2 = etf_2[obs_dates_2[field] == 1][field]
    # print(len(obs_etf_2), len(obs_etf_2[obs_etf_2 < 0.1]))

    plt.figure(figsize=(30, 5))
    plt.title(field)

    # plt.plot(etf_1[obs_dates_1[field] == 1][field], 'o')
    # plt.plot(etf_1[field])

    # plt.plot(etf_2[obs_dates_2[field] == 1][field], 'o')
    # plt.plot(etf_2[field])

    plt.plot(area[field], label='field coverage')

    plt.plot(obs_etf, 'o', label='observation dates')
    plt.plot(etf[field])

    # plt.plot(obs_etf_gs, 'o')  # just growing season (April through Oct)
    # plt.plot(etf_gs[field])

    plt.xticks(pd.date_range(etf.index[0], etf.index[-1], freq='YS'),
               labels=np.arange(etf.index[0].year, etf.index[-1].year + 1))
    plt.hlines([0, 1], etf.index[0], etf.index[-1], 'k')
    plt.grid()
    plt.legend()


def misc():
    """ Provides assorted additional functions. None required for main processing procedure. """
    print()
    # # EXTRA STUFF
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

    # # Creating file folders to store etf data by county
    # for i in COUNTIES.keys():
    #     file_path = 'C:/Users/CND571/Documents/Data/ssebop/{}'.format(i)
    #     if os.path.exists(file_path):
    #         continue
    #     else:
    #         os.mkdir(file_path)

    # Looking for mistaken tiny fields
    # When defined as area < 1 m^2: 21 identified in 099, 43 across whole dataset.
    # When defined as area < 100 m^2: 72 identified across whole dataset.
    # When defined as area < 500 m^2: 87 identified across whole dataset.
    # When defined as area < 1000 m^2: 98 identified across whole dataset.
    size = 1000
    sid_dir = os.path.join(d, 'statewide_irrigation_dataset_15FEB2024')
    tiny_fields = []
    for i in COUNTIES.keys():
        sid_file = os.path.join(sid_dir, '{}.shp'.format(i))
        if os.path.isfile(sid_file):
            area = gpd.read_file(sid_file)
            area = area.to_crs("EPSG:5071")
            area.index = area['FID']
            area = area.drop(['FID'], axis=1)
            area['area_m2'] = area['geometry'].area
            tiny_fields.append(list(area[area['area_m2'] < size].index))
            # if len(area[area['area_m2'] < size]) > 0:
            #     # print("{} Tiny Fields!".format(len(area[area['area_m2'] < size])))
            #     print(list(area[area['area_m2'] < size].index))
    tiny_fields = [i for row in tiny_fields for i in row]
    # print(tiny_fields)
    print(len(tiny_fields))
    for i in range(len(tiny_fields)):
        print(tiny_fields[i])
    tiny_counties = [x[:3] for x in tiny_fields]
    print(set(tiny_counties))
    print("{} fields < {} m^2 in {} counties".format(len(tiny_fields), size, len(set(tiny_counties))))


def check_files(directory):
    print('Part 1 completion status')
    # Logic for part 1 is not foolproof, but should be close enough to be useful.
    goal_num = 74
    remain = []
    missing = []
    done = []
    for i in COUNTIES.keys():
        folder = os.path.join(directory, i)
        if os.path.exists(folder):
            num_files = len([name for name in os.listdir(folder) if name.endswith('.csv')])
            if i in SPLIT:  # Check a and b folders too
                foldera = os.path.join(directory, i + 'a')
                folderb = os.path.join(directory, i + 'b')
                num_files_a = len([name for name in os.listdir(foldera) if name.endswith('.csv')])
                num_files_b = len([name for name in os.listdir(folderb) if name.endswith('.csv')])
                num_files = num_files + (num_files_a + num_files_b)/2
            if num_files == 0:
                remain.append(i)
            elif num_files < goal_num:
                missing.append(i)
            elif num_files >= goal_num:
                done.append(i)
    print('{} Counties Completed: {}'.format(len(done), done))
    if missing:
        print('{} Counties Incomplete: {}'.format(len(missing), missing))
    print('{} Counties Remaining: {}'.format(len(remain), remain))
    print()

    print('Part 2 completion status')
    goal_num = 111
    remain = []
    missing = []
    done = []
    too_many = []
    for i in COUNTIES.keys():
        folder = os.path.join(directory, i)
        if os.path.exists(folder):
            num_files = len([name for name in os.listdir(folder) if name.endswith('.csv')])
            if num_files == 0:
                remain.append(i)
            elif num_files < goal_num:
                missing.append(i)
            elif num_files >= goal_num:
                done.append(i)
            else:
                too_many.append(i)
    print('{} Counties Completed: {}'.format(len(done), done))
    if missing:
        print('{} Counties Incomplete: {}'.format(len(missing), missing))
    # print('{} Counties Remaining: {}'.format(len(remain), remain))
    if too_many:
        print('{} Counties have too many files: {}'.format(len(too_many), too_many))

    # # Don't know how to do part 3
    # print()
    # print('Part 3 completion status')
    # folder = os.path.join(directory, 'landsat')
    # filenames = [name for name in os.listdir(folder) if name.endswith('.csv')]
    #
    # # file_list = [x for x in os.listdir(csv_dir) if
    # #              x.endswith('.csv') and '_{}'.format(yr) in x]
    # remain = []
    # missing = []
    # done = []
    # too_many = []
    # print(filenames)

    # print('{} Counties Completed: {}'.format(len(done), done))
    # if missing:
    #     print('{} Counties Incomplete: {}'.format(len(missing), missing))
    # print('{} Counties Remaining: {}'.format(len(remain), remain))
    # if too_many:
    #     print('{} Counties have too many files: {}'.format(len(too_many), too_many))


if __name__ == '__main__':
    # PART ZERO: SETUP
    d = 'C:/Users/CND571/Documents/Data'

    # All counties, smallest to largest
    counties = ['019', '033', '061', '101', '051', '041', '091', '053', '015', '093', '055', '037', '075', '023',
                '069', '045', '079', '107', '021', '027', '089', '039', '035', '085', '065', '063', '059', '077',
                '029', '017', '087', '103', '007', '095', '013', '043', '001', '083', '049', '057', '005', '009',
                '003', '097', '067', '071', '031', '105', '073', '081', '099', '111', '047']
    # county = counties[-1]
    # counties larger than 2000 will be split.

    # # PART ONE: GOOGLE EARTH ENGINE DATA TO BUCKET
    # # Completed on 4/15/24
    # ee_loc = 'projects/ee-hehaugen/assets/SID_15FEB2024/'
    # # bucket = 'wudr'  # David's bucket
    # bucket = 'mt_cu_2024'  # Hannah's bucket
    # county_ee_data_to_bucket(county, ee_loc, bucket, d)

    # # PART TWO: Turning pixel counts into fractional area coverage, also combines split counties.
    # # Completed on 4/15/24
    # sid_dir_ = os.path.join(d, 'statewide_irrigation_dataset_15FEB2024')
    # etf_dir_ = os.path.join(d, 'ssebop')
    # # area_fraction_file('047', sid_dir_, etf_dir_)
    # for i in tqdm(counties, total=len(counties)):
    #     area_fraction_file(i, sid_dir_, etf_dir_)

    # # PART 2.5: LOOK AT DATA QUALITY
    # # counties = ['047']
    # data_quality_plot(counties[:27])
    # data_quality_plot(counties[27:])
    # # check_fields_horizontal(counties[3])
    # # for i in range(6, 9):
    # #     check_fields_horizontal(counties[i])
    # # etf_ts_plot('019')
    # plt.show()

    # PART THREE: FIX TIME SERIES DATA
    # dtype = 'extracts'
    # county = '019'

    project_ws = os.path.join(d, 'ssebop')
    tables = os.path.join(project_ws, 'landsat')

    yrs = [x for x in range(1987, 2022)]  # looks like a year lag, so only Jan/Feb 2023 ssebop data available.

    for county in tqdm(counties[::-1], total=len(counties)):
        shp = os.path.join(d, 'statewide_irrigation_dataset_15FEB2024', '{}.shp'.format(county))

        # directory with exported ee csv files
        ee_data = os.path.join(project_ws, county)

        # Out files
        src = os.path.join(tables, '{}_{}_{}.csv'.format(county, 'etf', 'irr'))
        src_ct = os.path.join(tables, '{}_{}_{}_ct.csv'.format(county, 'etf', 'irr'))

        landsat_time_series_multipolygon(shp, ee_data, yrs, src, src_ct)

    # misc()
    # check_files(os.path.join(d, 'ssebop'))

# ========================= EOF ====================================================================
