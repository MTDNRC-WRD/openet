
import os

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import geopandas as gpd
from scipy.stats import linregress
from itertools import islice
# from sortedcontainers import SortedDict

# For density plots
# from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn

from run_all import COUNTIES


def sum_data(con, start=1987, end=2023, irrmapper=0, mf_periods='1997-2006', static_too=False, save=""):
    """ Creates a dataframe summarizing results over a given time period by querying database.
    con: sqplite connection
    start: int, optional; first year in period of interest.
    end: int, optional; last year in period of interest (inclusive).
    irrmapper: int, optional; zero for results w/o irrmapper, 1 to draw results w/ irrmapper from db.
    mf_periods: str, optional; which management factor to apply to results.
    Options: '1997-2006', '1973-2006', or '1964-1973'
    static_too: bool, optional; False to do nothing, True to query the static IWR result table for the same mf period.
    """
    data = pd.read_sql("SELECT * FROM field_cu_results WHERE irrmapper={} AND mf_periods='{}' "
                       "AND year BETWEEN {} AND {}"
                       .format(irrmapper, mf_periods, start, end), con)
    data.drop(columns=['mf_periods'], inplace=True)
    data = data.groupby('fid').mean()  # average for period of record
    data['county'] = data.index.str.slice(0, 3)

    if static_too:
        data1 = pd.read_sql("SELECT * FROM static_iwr_results WHERE mf_periods='{}'".format(mf_periods), con)
        data1.drop(columns=['mf_periods'], inplace=True)
        data1 = data1.groupby('fid').mean()  # average for period of record
        data1['county'] = data1.index.str.slice(0, 3)
        if len(save) > 0:
            data.to_csv(os.path.join(save, "cu_results_{}_{}_im{}_mf{}.csv".format(start, end, irrmapper, mf_periods)))
            data1.to_csv(os.path.join(save, "iwr_cu_results_mf{}.csv".format(mf_periods)))
        else:
            return data, data1
    else:
        if len(save) > 0:
            data.to_csv(os.path.join(save, "cu_results_{}_{}_im{}_mf{}.csv".format(start, end, irrmapper, mf_periods)))
        else:
            return data


def time_series_data(con, save_dir):
    """ Creates a moving-average time series of several result variables, and saves the result to a csv. """
    window = 6
    starts = range(1987, 2018, int(window/2))

    the_index = pd.MultiIndex.from_product([COUNTIES.keys(), starts], names=["county", "st_year"])
    mov_avg = pd.DataFrame(index=the_index, columns=['etos', 'etbc', 'opnt_cu', 'dnrc_cu', 'frac_irr'])
    # for i in range(1987, 2018, 3):
    for i in tqdm(starts, total=len(starts)):
        decade = sum_data(con, start=i, end=i+window)
        grouped = decade.groupby('county').mean()

        im = sum_data(con, start=i, end=i+window, irrmapper=1)
        im_grouped = im.groupby('county').mean()

        for j in COUNTIES.keys():
            mov_avg.at[(j, i), 'etos'] = grouped[grouped.index == j]['etos'].iloc[0]
            mov_avg.at[(j, i), 'etbc'] = grouped[grouped.index == j]['etbc'].iloc[0]
            mov_avg.at[(j, i), 'opnt_cu'] = grouped[grouped.index == j]['opnt_cu'].iloc[0]
            mov_avg.at[(j, i), 'dnrc_cu'] = grouped[grouped.index == j]['dnrc_cu'].iloc[0]
            mov_avg.at[(j, i), 'frac_irr'] = im_grouped[im_grouped.index == j]['frac_irr'].iloc[0]

    # print(mov_avg)
    mov_avg.to_csv(os.path.join(save_dir, "ts_6yrs_im0_mf1997-2006.csv"))


def gridmet_ct(con, pront=False):
    """ Returns the number of distinct gridmet IDs in the db and each county. Optionally prints the results. """
    ct = {}
    gridmets = pd.read_sql("SELECT DISTINCT gfid FROM field_data", con)
    ct['total'] = len(gridmets)
    if pront:
        print("total: gridmet points: {}".format(len(gridmets)))
    for i in COUNTIES.keys():
        gridmets = pd.read_sql("SELECT DISTINCT gfid FROM field_data WHERE county='{}'".format(i), con)
        ct[i] = len(gridmets)
        if pront:
            print("gridmet points in {} County ({}): {}".format(COUNTIES[i], i, len(gridmets)))
    return ct


def plot_results(con, irrmapper=0, mf_periods='1997-2006', met_too=False):
    """Create figure comparing ET and consumptive use from two different methods."""

    all_data = pd.read_sql("SELECT * FROM field_cu_results WHERE irrmapper={} AND mf_periods='{}'"
                           .format(irrmapper, mf_periods), con)
    all_data.drop(columns=['mf_periods'], inplace=True)
    data = all_data.groupby('fid').mean()  # average for period of record

    data['county'] = data.index.str.slice(0, 3)
    all_data['county'] = all_data['fid'].str.slice(0, 3)

    # ET comparison
    plt.figure(figsize=(10, 5), dpi=200)

    plt.subplot(121)
    plt.title("Average Seasonal ET (in)")
    for i in data['county'].unique():  # Why did it plot in a different order than last time?
        plt.scatter(data[data['county'] == i]['etbc'], data[data['county'] == i]['etos'],
                    zorder=5,
                    label="{} ({})".format(COUNTIES[i], i))
        # plt.scatter(all_data[all_data['county'] == i]['etbc'], all_data[all_data['county'] == i]['etos'],
        #             c=all_data[all_data['county'] == i]['year'],
        #             zorder=5,
        #             label="{} ({}) by year".format(COUNTIES[i], i))
    plt.plot(all_data['etbc'], all_data['etbc'], 'k', zorder=4, label="1:1")
    # plt.plot(data['etbc'], data['etbc'], 'k', zorder=4, label="1:1")
    plt.grid(zorder=3)
    plt.xlabel('DNRC')
    plt.ylabel('Gridmet ETo (grass reference)')
    plt.legend(title='County')

    plt.subplot(122)
    plt.title("Average Seasonal Consumptive Use (in)")
    for i in data['county'].unique():
        plt.scatter(data[data['county'] == i]['dnrc_cu'], data[data['county'] == i]['opnt_cu'], zorder=5,
                    label="{} ({})".format(COUNTIES[i], i))
        # plt.scatter(all_data[all_data['county'] == i]['dnrc_cu'], all_data[all_data['county'] == i]['opnt_cu'],
        #             c=all_data[all_data['county'] == i]['year'],
        #             zorder=5,
        #             label="{} ({}) by year".format(COUNTIES[i], i))
    # plt.plot(data['opnt_cu'], data['opnt_cu'], 'k', zorder=4, label="1:1")
    plt.plot(all_data['opnt_cu'], all_data['opnt_cu'], 'k', zorder=4, label="1:1")
    plt.grid(zorder=3)
    plt.xlabel('DNRC')
    plt.ylabel('OpenET')
    plt.legend(title='County')

    plt.tight_layout()

    # Looking at other meteorological variables
    if met_too:
        gridmets = pd.read_sql("SELECT DISTINCT gfid FROM gridmet_ts", con)
        other_climate = pd.DataFrame(columns=['q_kgkg', 'u10_ms', 'srad_wm2', 't'], index=gridmets['gfid'])
        for i in gridmets['gfid']:
            # print("i", i)
            grd = pd.read_sql("SELECT date, q_kgkg, u10_ms, srad_wm2, tmax_c, tmin_c FROM gridmet_ts WHERE gfid={}"
                              .format(i), con)
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
        # plt.title()
        for i in data['county'].unique():  # Why did it plot in a different order than last time?
            plt.scatter(data[data['county'] == i]['q_kgkg'],
                        data[data['county'] == i]['etos'] - data[data['county'] == i]['etbc'],
                        zorder=5,
                        label="{} ({})".format(COUNTIES[i], i))
        # plt.plot(data['etbc'], data['etbc'], 'k', zorder=4, label="1:1")
        plt.grid(zorder=3)
        plt.xlabel('seasonal average daily humidity (kg/kg)')
        plt.ylabel('Bias in ET (OpenET - DNRC)')
        plt.legend(title='County')

        plt.subplot(222)
        # plt.title()
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
        # plt.title()
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
        # plt.title()
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


def plot_results_1(data):
    """Create figure comparing ET and consumptive use from two different methods."""
    # ET comparison
    plt.figure(figsize=(10, 5), dpi=200)

    plt.subplot(121)
    plt.title("Average Seasonal ET (in)")
    for i in data['county'].unique():
        plt.scatter(data[data['county'] == i]['etbc'], data[data['county'] == i]['etos'],
                    zorder=5, alpha=0.2,
                    label="{} ({})".format(COUNTIES[i], i))
    plt.plot(data['etbc'], data['etbc'], 'k', zorder=4, label="1:1")
    plt.grid(zorder=3)
    plt.xlabel('DNRC')
    plt.ylabel('Gridmet ETo (grass reference)')
    plt.legend(title='County')

    # CU comparison
    plt.subplot(122)
    plt.title("Average Seasonal Consumptive Use (in)")
    for i in data['county'].unique():
        plt.scatter(data[data['county'] == i]['dnrc_cu'], data[data['county'] == i]['opnt_cu'], zorder=5,
                    label="{} ({})".format(COUNTIES[i], i), alpha=0.2)
    plt.plot(data['opnt_cu'], data['opnt_cu'], 'k', zorder=4, label="1:1")
    plt.grid(zorder=3)
    plt.xlabel('DNRC')
    plt.ylabel('OpenET')
    plt.legend(title='County')

    plt.tight_layout()


def county_hist(data, iwr, selection=(), ymax=1000):
    """ Makes 2 triple histograms (DNRC, DNRC climate, and OpenET) for ET and CU for each county given. """
    if len(selection) == 0:
        counties = data['county'].unique()
    else:
        counties = selection

    bins_et = np.arange(20, 35)+0.5
    bins_cu = np.arange(0, 25)+0.5

    clrs = mpl.colormaps['tab20'].colors

    for i in counties:
        plt.figure(figsize=(10, 5), dpi=200)
        plt.suptitle("{} County ({})".format(COUNTIES[i], i))

        plt.subplot(121)
        plt.title("Average Seasonal ET (in)")
        # plt.hist(data[data['county'] == i]['etbc'], label='DNRC', zorder=5)
        # plt.hist(data[data['county'] == i]['etos'], label='Gridmet ETo', zorder=5)
        plt.hist([data[data['county'] == i]['etbc'],
                  iwr[iwr['county'] == i]['etbc'], data[data['county'] == i]['etos']],
                 label=['DNRC', 'DNRC (climate)', 'Gridmet ETo'], zorder=5, bins=bins_et,
                 color=[clrs[0], clrs[4], clrs[2]])
        plt.vlines(data[data['county'] == i]['etbc'].mean(), 0, ymax, ls='dashed',
                   label='mean: {:.2f}'.format(data[data['county'] == i]['etbc'].mean()), color=clrs[1])
        plt.vlines(iwr[iwr['county'] == i]['etbc'].mean(), 0, ymax, ls='dashed',
                   label='mean: {:.2f}'.format(iwr[iwr['county'] == i]['etbc'].mean()), color=clrs[5])
        plt.vlines(data[data['county'] == i]['etos'].mean(), 0, ymax, ls='dashed',
                   label='mean: {:.2f}'.format(data[data['county'] == i]['etos'].mean()), color=clrs[3])
        plt.grid(zorder=1)
        # plt.legend(ncols=2)
        plt.legend()
        plt.ylim(0, ymax)

        # CU comparison
        plt.subplot(122)
        plt.title("Average Seasonal Consumptive Use (in)")
        # plt.hist(data[data['county'] == i]['dnrc_cu'], label='DNRC', zorder=5)
        # plt.hist(data[data['county'] == i]['opnt_cu'], label='OpenET', zorder=5)
        plt.hist([data[data['county'] == i]['dnrc_cu'],
                  iwr[iwr['county'] == i]['dnrc_cu'], data[data['county'] == i]['opnt_cu']],
                 label=['DNRC', 'DNRC (climate)', 'OpenET'], zorder=5, bins=bins_cu,
                 color=[clrs[0], clrs[4], clrs[2]])
        plt.vlines(data[data['county'] == i]['dnrc_cu'].mean(), 0, ymax*0.8, ls='dashed',
                   label='mean: {:.2f}'.format(data[data['county'] == i]['dnrc_cu'].mean()), color=clrs[1])
        plt.vlines(iwr[iwr['county'] == i]['dnrc_cu'].mean(), 0, ymax*0.8, ls='dashed',
                   label='mean: {:.2f}'.format(iwr[iwr['county'] == i]['dnrc_cu'].mean()), color=clrs[5])
        plt.vlines(data[data['county'] == i]['opnt_cu'].mean(), 0, ymax*0.8, ls='dashed',
                   label='mean: {:.2f}'.format(data[data['county'] == i]['opnt_cu'].mean()), color=clrs[3])
        plt.grid(zorder=1)
        # plt.legend(ncols=2)
        plt.legend()
        plt.xlim(0, 25)
        plt.ylim(0, ymax*0.8)

    plt.tight_layout()


def county_hist_1(data, iwr):
    """ Plots histograms of CU for each of the 52 counties in the database in one figure. Ugly, but effective. """
    plt.figure(figsize=(30, 10), dpi=200)
    plt.suptitle("Average Seasonal Consumptive Use (in)")
    # plt.suptitle()

    print("total average CU: dnrc: {:.2f}, opnt: {:.2f}".format(data['dnrc_cu'].mean(), data['opnt_cu'].mean()))

    plot = 1
    for i in data['county'].unique():
        plt.subplot(4, 13, plot)
        # plt.hist([data[data['county'] == i]['etbc'], data[data['county'] == i]['etos']],
        #          label=['DNRC', 'Gridmet ETo'], zorder=5)
        plt.title("{} ({})".format(COUNTIES[i], i), size=8)
        bins = np.arange(30)+0.5
        plt.hist([data[data['county'] == i]['dnrc_cu'], data[data['county'] == i]['opnt_cu']], bins=bins,
                 label=['DNRC', 'OpenET'], zorder=5)
        # plt.hist([data[data['county'] == i]['dnrc_cu'], data[data['county'] == i]['opnt_cu'],
        #          iwr[iwr['county'] == i]['dnrc_cu']], bins=bins,
        #          label=['DNRC', 'OpenET', 'IWR climate'], zorder=5)
        # plt.hist([data[data['county'] == i]['dnrc_cu'], iwr[iwr['county'] == i]['dnrc_cu']], bins=bins,
        #          label=['DNRC', 'DNRC (climate)'], zorder=5)
        # plt.vlines(data[data['county'] == i]['dnrc_cu'].mean(), 0, 500, colors='tab:green')
        # plt.vlines(data[data['county'] == i]['opnt_cu'].mean(), 0, 500, colors='tab:pink')
        plt.grid(zorder=1)
        plt.xticks(size=6)
        plt.yticks(size=6)
        if plot == 1:
            plt.legend(fontsize=4)
        plot += 1

        print("{} average CU: dnrc: {:2.2f}, opnt: {:2.2f}, iwr: {:2.2f} ({} County)"
              .format(i, data[data['county'] == i]['dnrc_cu'].mean(),
                      data[data['county'] == i]['opnt_cu'].mean(), iwr[iwr['county'] == i]['dnrc_cu'].mean(),
                      COUNTIES[i]))

    plt.tight_layout()


def county_scatter_1_cu(data, data_im, iwr):
    """ Plots scatterplots of CU for each of the 52 counties in the database in one figure. Ugly, but effective. """
    plt.figure(figsize=(30, 10), dpi=200)
    plt.suptitle("Average Seasonal Consumptive Use (in)")
    # plt.suptitle()

    print("total average CU: dnrc: {:.2f}, opnt: {:.2f}".format(data['dnrc_cu'].mean(), data['opnt_cu'].mean()))

    dif = []

    plot = 1
    for i in data['county'].unique():
        plt.subplot(4, 13, plot)
        plt.title("{} ({})".format(COUNTIES[i], i), size=8)
        # tight vertical cloud sitting under 1:1 line
        plt.scatter(data[data['county'] == i]['dnrc_cu'], data[data['county'] == i]['opnt_cu'],
                    label='no IM', zorder=5, s=3, alpha=0.2)
        # # curves near 1:1 line, but am I just autocorrellating it?
        # plt.scatter(data_im[data_im['county'] == i]['dnrc_cu'], data_im[data_im['county'] == i]['opnt_cu'],
        #             label='w/ IM', zorder=5, s=3, alpha=0.2)
        # # wide cloud, low slope
        # plt.scatter(data_im[data_im['county'] == i]['dnrc_cu'], data[data['county'] == i]['opnt_cu'],
        #             label='DNRC w/ IM', zorder=5, s=3, alpha=0.2)
        # # Vertical lines
        # plt.scatter(iwr[iwr.index.isin(data[data['county'] == i].index)]['dnrc_cu'],
        #             data[data['county'] == i]['opnt_cu'], label='IWR', zorder=5, s=3, alpha=0.2)
        plt.grid(zorder=1)
        plt.xticks(size=6)
        plt.yticks(size=6)
        plt.xlim(0, 30)
        plt.ylim(0, 30)
        plt.plot([0, 30], [0, 30], 'k', zorder=4)
        if plot == 1:
            plt.legend(fontsize=4)
        plot += 1

        print("{} average CU: dnrc: {:2.2f}, opnt: {:2.2f}, iwr: {:2.2f} ({} County)"
              .format(i, data[data['county'] == i]['dnrc_cu'].mean(),
                      data[data['county'] == i]['opnt_cu'].mean(), iwr[iwr['county'] == i]['dnrc_cu'].mean(),
                      COUNTIES[i]))
        dif.append(data[data['county'] == i]['dnrc_cu'].mean() - data[data['county'] == i]['opnt_cu'].mean())

    plt.tight_layout()
    print(dif)


def county_scatter_1_et(data, data_im, iwr):
    """ Plots scatterplots of ET for each of the 52 counties in the database in one figure. Ugly, but effective. """
    plt.figure(figsize=(30, 10), dpi=200)
    plt.suptitle("Average Seasonal Consumptive Use (in)")
    # plt.suptitle()

    print("total average CU: dnrc: {:.2f}, opnt: {:.2f}".format(data['dnrc_cu'].mean(), data['opnt_cu'].mean()))

    plot = 1
    for i in data['county'].unique():
        plt.subplot(4, 13, plot)
        plt.title("{} ({})".format(COUNTIES[i], i), size=8)
        #
        plt.scatter(data[data['county'] == i]['etbc'], data[data['county'] == i]['etos'],
                    label='no IM', zorder=5, s=3, alpha=0.2)
        # Carrots along the 1:1 line (in orange)
        plt.scatter(data_im[data_im['county'] == i]['etbc'], data_im[data_im['county'] == i]['etos'],
                    label='w/ IM', zorder=5, s=3, alpha=0.2)
        #
        plt.scatter(iwr[iwr.index.isin(data[data['county'] == i].index)]['etbc'],
                    data[data['county'] == i]['etos'], label='IWR', zorder=5, s=3, alpha=0.2)
        plt.grid(zorder=1)
        plt.xticks(size=6)
        plt.yticks(size=6)
        plt.xlim(20, 40)
        plt.ylim(20, 40)
        plt.plot([0, 40], [0, 40], 'k', zorder=4)
        if plot == 1:
            plt.legend(fontsize=4)
        plot += 1

        print("{} average CU: dnrc: {:2.2f}, opnt: {:2.2f}, iwr: {:2.2f} ({} County)"
              .format(i, data[data['county'] == i]['dnrc_cu'].mean(),
                      data[data['county'] == i]['opnt_cu'].mean(), iwr[iwr['county'] == i]['dnrc_cu'].mean(),
                      COUNTIES[i]))

    plt.tight_layout()


def dif_hist(data):
    plt.figure(figsize=(10, 5))
    plt.title("Difference in Consumptive Use Estimate (inches, DNRC minus OpenET)")
    bins = np.arange(-10, 25)+0.5
    plt.hist(data['dnrc_cu'] - data['opnt_cu'], bins=bins, zorder=5)
    # Extra stuff
    mean = (data['dnrc_cu'] - data['opnt_cu']).mean()
    std = (data['dnrc_cu'] - data['opnt_cu']).std()
    plt.vlines(mean, 0, 5000, color='tab:pink', zorder=7,
               label="mean: {:.2f}".format(mean))
    plt.vlines([mean - std, mean + std], 0, 5000, color='tab:pink', zorder=7,
               label="std:     {:.2f}".format(std), linestyle='dashed')
    plt.axvspan(mean-std, mean+std, 0, 5000, color='tab:pink', alpha=0.5, zorder=3)
    # plt.vlines(0, 0, 5000, color='k', zorder=2)
    plt.grid(zorder=1)
    plt.legend()
    plt.ylim(0, 5000)
    plt.xlim(-10, 25)

    print("average difference: {:.2f}".format((data['dnrc_cu'] - data['opnt_cu']).mean()))
    print("Fields with higher DNRC value: {:.2f}%"
          .format(100*(data['dnrc_cu'] - data['opnt_cu']).gt(0).sum()/len(data)))


def dif_hist_1(data, iwr):
    plt.figure(figsize=(10, 5))
    plt.title("Difference in Consumptive Use Estimate (inches, DNRC (climate) minus OpenET)")
    bins = np.arange(-10, 25)+0.5
    plt.hist(iwr[iwr['county'] != '101']['dnrc_cu'] - data['opnt_cu'], bins=bins, zorder=5)
    # Extra stuff
    mean = (iwr[iwr['county'] != '101']['dnrc_cu'] - data['opnt_cu']).mean()
    std = (iwr[iwr['county'] != '101']['dnrc_cu'] - data['opnt_cu']).std()
    plt.vlines(mean, 0, 5000, color='tab:pink', zorder=7,
               label="mean: {:.2f}".format(mean))
    plt.vlines([mean - std, mean + std], 0, 5000, color='tab:pink', zorder=7,
               label="std:     {:.2f}".format(std), linestyle='dashed')
    plt.axvspan(mean-std, mean+std, 0, 5000, color='tab:pink', alpha=0.5, zorder=3)
    # plt.vlines(0, 0, 5000, color='k', zorder=2)
    plt.grid(zorder=1)
    plt.legend()
    plt.ylim(0, 5000)
    plt.xlim(-10, 25)

    print("average difference: {:.2f}".format((iwr[iwr['county'] != '101']['dnrc_cu'] - data['opnt_cu']).mean()))
    print("Fields with higher DNRC value: {:.2f}%"
          .format(100*(iwr[iwr['county'] != '101']['dnrc_cu'] - data['opnt_cu']).gt(0).sum()/len(data)))


def dif_hist_2(data, iwr):
    plt.figure(figsize=(10, 5))
    plt.title("Difference in Consumptive Use Estimate (inches, DNRC minus OpenET)")
    bins = np.arange(-10, 25)+0.5
    # Daily difference
    plt.hist(data['dnrc_cu'] - data['opnt_cu'], bins=bins, zorder=5, label='DNRC (daily)',
             color='tab:blue', alpha=0.8)

    mean = (data['dnrc_cu'] - data['opnt_cu']).mean()
    std = (data['dnrc_cu'] - data['opnt_cu']).std()
    plt.vlines(mean, 0, 5000, color='tab:pink', zorder=7,
               label="mean: {:.2f}".format(mean))
    plt.vlines([mean - std, mean + std], 0, 5000, color='tab:pink', zorder=7,
               label="std:     {:.2f}".format(std), linestyle='dashed')
    plt.axvspan(mean - std, mean + std, 0, 5000, color='tab:pink', alpha=0.2, zorder=3)

    # Climate difference
    plt.hist(iwr[iwr['county'] != '101']['dnrc_cu'] - data['opnt_cu'], bins=bins, zorder=5, label='DNRC (climate)',
             color='tab:green', alpha=0.8)

    mean = (iwr[iwr['county'] != '101']['dnrc_cu'] - data['opnt_cu']).mean()
    std = (iwr[iwr['county'] != '101']['dnrc_cu'] - data['opnt_cu']).std()
    plt.vlines(mean, 0, 5000, color='tab:orange', zorder=7,
               label="mean: {:.2f}".format(mean))
    plt.vlines([mean - std, mean + std], 0, 5000, color='tab:orange', zorder=7,
               label="std:     {:.2f}".format(std), linestyle='dashed')
    plt.axvspan(mean - std, mean + std, 0, 5000, color='tab:orange', alpha=0.2, zorder=3)

    # plt.vlines(0, 0, 5000, color='k', zorder=2)
    plt.grid(zorder=1)
    plt.legend()
    plt.ylim(0, 5000)
    plt.xlim(-10, 25)

    print('DNRC (daily)')
    print("average difference: {:.2f}".format((data['dnrc_cu'] - data['opnt_cu']).mean()))
    print("Fields with higher DNRC value: {:.2f}%"
          .format(100 * (data['dnrc_cu'] - data['opnt_cu']).gt(0).sum() / len(data)))
    print('DNRC (climate)')
    print("average difference: {:.2f}".format((iwr[iwr['county'] != '101']['dnrc_cu'] - data['opnt_cu']).mean()))
    print("Fields with higher DNRC value: {:.2f}%"
          .format(100 * (iwr[iwr['county'] != '101']['dnrc_cu'] - data['opnt_cu']).gt(0).sum() / len(data)))


def all_hist_1(data, iwr):
    plt.figure(figsize=(10, 5))
    plt.title("Average Seasonal Consumptive Use Estimate (inches)")
    bins = np.arange(-1, 30)+0.5

    # Option 1
    # plt.hist(data['dnrc_cu'], bins=bins, zorder=5, alpha=0.5, label='DNRC', color='tab:blue')
    plt.hist(iwr['dnrc_cu'], bins=bins, zorder=5, alpha=0.5, label='DNRC (climate)', color='tab:blue')
    plt.hist(data['opnt_cu'], bins=bins, zorder=5, alpha=0.5, label='OpenET', color='tab:orange')

    # plt.vlines(data['dnrc_cu'].mean(), 0, 8000, zorder=7, color='tab:blue',
    #            label='mean: {:.2f} in'.format(data['dnrc_cu'].mean()))
    plt.vlines(iwr['dnrc_cu'].mean(), 0, 8000, zorder=7, color='tab:blue',
               label='mean: {:.2f} in'.format(iwr['dnrc_cu'].mean()))
    plt.vlines(data['opnt_cu'].mean(), 0, 8000, zorder=7, color='tab:orange',
               label='mean: {:.2f} in'.format(data['opnt_cu'].mean()))

    # Option 2
    # plt.hist([data['dnrc_cu'], data['opnt_cu'], iwr['dnrc_cu']],
    #          bins=bins, zorder=5, label=['DNRC', 'OpenET', 'IWR climate'])
    # plt.hist([data['dnrc_cu'], iwr['dnrc_cu']], bins=bins, zorder=5, label=['DNRC', 'IWR climate'])

    # # Extra stuff
    # mean = (data['dnrc_cu'] - data['opnt_cu']).mean()
    # std = (data['dnrc_cu'] - data['opnt_cu']).std()
    # plt.vlines(mean, 0, 5000, color='tab:pink', zorder=7,
    #            label="mean: {:.2f}".format(mean))
    # plt.vlines([mean - std, mean + std], 0, 5000, color='tab:pink', zorder=7,
    #            label="std:     {:.2f}".format(std), linestyle='dashed')
    # plt.axvspan(mean-std, mean+std, 0, 5000, color='tab:pink', alpha=0.5, zorder=3)

    plt.ylim(0, 8000)
    # plt.xlim(5, 30)
    plt.xlim(0, 30)
    plt.grid(zorder=1)
    plt.legend(ncols=2)
    # plt.ylim(0, 5000)

    # print("average difference: {:.2f}".format((data['dnrc_cu'] - data['opnt_cu']).mean()))
    # print("Fields with higher DNRC value: {:.2f}%"
    #       .format(100*(data['dnrc_cu'] - data['opnt_cu']).gt(0).sum()/len(data)))


def all_hist_2(data, data_im):
    plt.figure(figsize=(10, 5))
    plt.title("Average Seasonal Consumptive Use Estimate (inches)")
    bins = np.arange(-1, 30)+0.5

    # Option 1
    # plt.hist(data['dnrc_cu'], bins=bins, zorder=5, alpha=0.5, label='DNRC', color='tab:blue')
    plt.hist(data_im['dnrc_cu'], bins=bins, zorder=5, alpha=0.5, label='DNRC (IrrMapper)', color='tab:blue')
    plt.hist(data['opnt_cu'], bins=bins, zorder=5, alpha=0.5, label='OpenET', color='tab:orange')

    # plt.vlines(data['dnrc_cu'].mean(), 0, 8000, zorder=7, color='tab:blue',
    #            label='mean: {:.2f} in'.format(data['dnrc_cu'].mean()))
    plt.vlines(data_im['dnrc_cu'].mean(), 0, 8000, zorder=7, color='tab:blue',
               label='mean: {:.2f} in'.format(data_im['dnrc_cu'].mean()))
    plt.vlines(data['opnt_cu'].mean(), 0, 8000, zorder=7, color='tab:orange',
               label='mean: {:.2f} in'.format(data['opnt_cu'].mean()))

    # Option 2
    # plt.hist([data['dnrc_cu'], data['opnt_cu'], iwr['dnrc_cu']],
    #          bins=bins, zorder=5, label=['DNRC', 'OpenET', 'IWR climate'])
    # plt.hist([data['dnrc_cu'], iwr['dnrc_cu']], bins=bins, zorder=5, label=['DNRC', 'IWR climate'])

    # # Extra stuff
    # mean = (data['dnrc_cu'] - data['opnt_cu']).mean()
    # std = (data['dnrc_cu'] - data['opnt_cu']).std()
    # plt.vlines(mean, 0, 5000, color='tab:pink', zorder=7,
    #            label="mean: {:.2f}".format(mean))
    # plt.vlines([mean - std, mean + std], 0, 5000, color='tab:pink', zorder=7,
    #            label="std:     {:.2f}".format(std), linestyle='dashed')
    # plt.axvspan(mean-std, mean+std, 0, 5000, color='tab:pink', alpha=0.5, zorder=3)

    plt.ylim(0, 8000)
    # plt.xlim(5, 30)
    plt.xlim(0, 30)
    plt.grid(zorder=1)
    plt.legend(ncols=2)
    # plt.ylim(0, 5000)

    # print("average difference: {:.2f}".format((data['dnrc_cu'] - data['opnt_cu']).mean()))
    # print("Fields with higher DNRC value: {:.2f}%"
    #       .format(100*(data['dnrc_cu'] - data['opnt_cu']).gt(0).sum()/len(data)))


def scatter(data):

    xs = []
    ys = []
    xs_err = []
    ys_err = []

    for i in data['county'].unique():
        xs.append(data[data['county'] == i]['dnrc_cu'].mean())
        ys.append(data[data['county'] == i]['opnt_cu'].mean())
        xs_err.append(data[data['county'] == i]['dnrc_cu'].std())
        ys_err.append(data[data['county'] == i]['opnt_cu'].std())

    # plt.figure(figsize=(8, 8))
    # plt.title("Average Seasonal Consumptive Use (in)")
    # plt.plot(xs, xs, 'k')
    # plt.plot(ys, ys, 'k')
    # plt.errorbar(xs, ys, xerr=xs_err, yerr=ys_err, fmt='none')
    # plt.xlabel('DNRC')
    # plt.ylabel('OpenET')
    # plt.grid()

    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Ellipse

    def make_error_boxes(ax1, xdata, ydata, xerror, yerror, facecolor='tab:blue',
                         edgecolor='none', alpha=0.15):
        # Loop over data points; create box from errors at each point
        # errorboxes = [Rectangle((x - xe, y - ye), xe*2, ye*2)
        #               for x, y, xe, ye in zip(xdata, ydata, xerror, yerror)]
        errorboxes = [Ellipse((x, y), xe*2, ye*2)
                      for x, y, xe, ye in zip(xdata, ydata, xerror, yerror)]

        # Create patch collection with specified colour/alpha
        pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                             edgecolor=edgecolor, zorder=3)

        # Add collection to Axes
        ax1.add_collection(pc)

        # Plot errorbars
        artists = ax1.errorbar(xdata, ydata, xerr=xerror, yerr=yerror,
                               fmt='ko', ecolor='none', zorder=4, label='County Mean Values')

        return artists

    # Create figure and Axes
    fig, ax = plt.subplots(1)
    fig.set_size_inches(6, 6)
    ax.set_title("Average Seasonal Consumptive Use (inches)")
    # ax.plot(xs, xs, 'k', zorder=5)
    # ax.plot(ys, ys, 'k', zorder=5)
    # Call function to create error boxes
    _ = make_error_boxes(ax, xs, ys, xs_err, ys_err)
    ax.grid(zorder=1)
    ax.set_xlabel('DNRC')
    ax.set_ylabel('OpenET')
    ax.set_xlim(5, 25)
    ax.set_ylim(5, 25)
    ax.plot([5, 25], [5, 25], 'k', zorder=5)
    ax.legend()
    # plt.show()


def plot_results_2(data):
    """Create figure comparing ET and consumptive use from two different methods."""

    xs = []
    ys = []
    xs_et = []
    ys_et = []
    for i in data['county'].unique():
        xs.append(data[data['county'] == i]['dnrc_cu'].mean())
        ys.append(data[data['county'] == i]['opnt_cu'].mean())
        xs_et.append(data[data['county'] == i]['etbc'].mean())
        ys_et.append(data[data['county'] == i]['etos'].mean())

    # ET comparison
    plt.figure(figsize=(10, 5), dpi=200)

    plt.subplot(121)
    plt.title("Average Seasonal ET (inches)")
    for i in data['county'].unique():
        plt.scatter(data[data['county'] == i]['etbc'], data[data['county'] == i]['etos'],
                    zorder=5, alpha=0.1, edgecolors='none', color='tab:blue', s=24)
    # plt.plot(data['etbc'], data['etbc'], 'k', zorder=7)
    plt.scatter(xs_et, ys_et, color='k', edgecolors='none', s=12, zorder=6, label='County mean values')
    plt.grid(zorder=3)
    plt.xlim(22, 38)
    plt.ylim(22, 38)
    plt.plot([0, 40], [0, 40], 'k', zorder=7)
    plt.xlabel('DNRC')
    plt.ylabel('Gridmet ETo (grass reference)')
    plt.legend()

    mean = (data['dnrc_cu'] - data['opnt_cu']).mean()
    std = (data['dnrc_cu'] - data['opnt_cu']).std()

    # CU comparison
    plt.subplot(122)
    plt.title("Average Seasonal Consumptive Use (inches)")
    for i in data['county'].unique():
        plt.scatter(data[data['county'] == i]['dnrc_cu'], data[data['county'] == i]['opnt_cu'], zorder=5,
                    alpha=0.1, edgecolors='none', color='tab:blue', s=24)  # label="{} ({})".format(COUNTIES[i], i),
    # plt.plot(data['opnt_cu'], data['opnt_cu'], 'k', zorder=7)
    plt.scatter(xs, ys, color='k', edgecolors='none', s=12, zorder=8, label='County mean values')
    plt.grid(zorder=3)
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.plot([0, 30], [0, 30], 'k', zorder=7)

    # # option 1
    # plt.plot([0, 30], [0-mean, 30-mean], 'tab:pink', zorder=7, label='Mean difference')
    # plt.plot([0, 30], [0-mean+std, 30-mean+std], 'tab:pink', zorder=7, linestyle='dashed')
    # plt.plot([0, 30], [0-mean-std, 30-mean-std], 'tab:pink', zorder=7, linestyle='dashed')
    # plt.fill_between([0, 30], [0-mean+std, 30-mean+std], [0-mean-std, 30-mean-std],
    #                  color='tab:pink', alpha=0.2, ec='none')

    # option 2
    plt.plot([0, 30], [0 - mean, 30 - mean], 'tab:pink', linestyle='dashed', zorder=7, label='Mean difference')
    plt.fill_between([0, 30], [0 - mean + std, 30 - mean + std], [0 - mean - std, 30 - mean - std],
                     color='tab:pink', alpha=0.2, ec='none')

    plt.xlabel('DNRC')
    plt.ylabel('OpenET')
    plt.legend()

    plt.tight_layout()


def plot_results_2_1(data, conv):
    """Create figure comparing ET and consumptive use from two different methods."""

    xs = []
    ys = []
    xs_et = []
    ys_et = []
    for i in data['county'].unique():
        xs.append(data[data['county'] == i]['dnrc_cu'].mean())
        ys.append(data[data['county'] == i]['opnt_cu'].mean())
        xs_et.append(data[data['county'] == i]['etbc'].mean())
        ys_et.append(data[data['county'] == i]['etos'].mean())

    res1 = linregress(data['etbc'], data['etos'])
    res1a = linregress(xs_et, ys_et)

    # ET comparison
    plt.figure(figsize=(15, 5), dpi=200)

    plt.subplot(131)
    plt.title("Average Seasonal ET (inches)")
    plt.scatter(data['etbc'], data['etos'],
                zorder=5, alpha=0.1, edgecolors='none', color='tab:blue', s=24)
    # for i in data['county'].unique():
    #     plt.scatter(data[data['county'] == i]['etbc'], data[data['county'] == i]['etos'],
    #                 zorder=5, alpha=0.1, edgecolors='none', color='tab:blue', s=24)
    # plt.plot(data['etbc'], data['etbc'], 'k', zorder=7)
    plt.scatter(xs_et, ys_et, color='k', edgecolors='none', s=12, zorder=6, label='County mean values')
    plt.grid(zorder=3)
    plt.xlim(22, 38)
    # plt.ylim(22, 38)
    # plt.xlim(22, 51)
    plt.ylim(22, 51)
    plt.plot([0, 51], [0, 51], 'k', zorder=7, label='1:1')
    plt.plot(np.sort(data['etbc']), res1.intercept + res1.slope * np.sort(data['etbc']),
             zorder=9, color='tab:pink', ls='dashed',
             label='y={:.2f}x+{:.1f} r^2: {:.2f}'.format(res1.slope, res1.intercept, res1.rvalue ** 2))
    plt.plot(np.sort(data['etbc']), res1a.intercept + res1a.slope * np.sort(data['etbc']),
             zorder=9, color='tab:purple', ls='dashed',
             label='y={:.2f}x+{:.1f} r^2: {:.2f}'.format(res1a.slope, res1a.intercept, res1a.rvalue ** 2))
    plt.xlabel('DNRC')
    plt.ylabel('Gridmet ETo (grass reference)')
    plt.legend()

    def closest(sorted_dict, key):
        """ Return closest key in `sorted_dict` to given `key`. """
        assert len(sorted_dict) > 0
        keys = list(islice(sorted_dict.irange(minimum=key), 1))
        keys.extend(islice(sorted_dict.irange(maximum=key, reverse=True), 1))
        return min(keys, key=lambda k: abs(key - k))

    conv = SortedDict(conv)

    # CU comparison
    plt.subplot(132)
    plt.title("Average Seasonal ET (inches)")
    ys_etr_est = []
    for i in data['county'].unique():
        ys = 1.67 * data[data['county'] == i]['etos'] - 10.6
        plt.scatter(data[data['county'] == i]['etbc'], ys,
                    zorder=5, alpha=0.1, edgecolors='none', color='tab:blue', s=24)
        plt.scatter(np.mean(data[data['county'] == i]['etbc']), np.mean(ys),
                    color='k', edgecolors='none', s=12, zorder=6)
        ys_etr_est.append(np.mean(ys))
    # plt.plot(data['etbc'], data['etbc'], 'k', zorder=7)
    # plt.scatter(0, 0, color='k', edgecolors='none', s=12, zorder=6, label='County mean values')
    res2 = linregress(data['etbc'], 1.67 * data['etos'] - 10.6)
    res2a = linregress(xs_et, ys_etr_est)
    plt.plot(np.sort(data['etbc']), res2.intercept + res2.slope * np.sort(data['etbc']),
             zorder=9, color='tab:pink', ls='dashed',
             label='y={:.2f}x+{:.1f} r^2: {:.2f}'.format(res2.slope, res2.intercept, res2.rvalue ** 2))
    plt.plot(np.sort(data['etbc']), res2a.intercept + res2a.slope * np.sort(data['etbc']),
             zorder=9, color='tab:purple', ls='dashed',
             label='y={:.2f}x+{:.1f} r^2: {:.2f}'.format(res2a.slope, res2a.intercept, res2a.rvalue ** 2))
    plt.grid(zorder=3)
    plt.xlim(22, 38)
    # plt.ylim(22, 38)
    # plt.xlim(22, 51)
    plt.ylim(22, 51)
    plt.plot([0, 51], [0, 51], 'k', zorder=7)
    plt.xlabel('DNRC')
    plt.ylabel('Estimated Gridmet ETr (alfalfa reference)')
    plt.legend()

    # CU comparison
    plt.subplot(133)
    plt.title("Average Seasonal ET (inches)")
    ys_etr = []
    for i in data['county'].unique():
        ys = [conv[closest(conv, j)] for j in data[data['county'] == i]['etos']]
        plt.scatter(data[data['county'] == i]['etbc'], ys,
                    zorder=5, alpha=0.1, edgecolors='none', color='tab:blue', s=24)
        plt.scatter(np.mean(data[data['county'] == i]['etbc']), np.mean(ys),
                    color='k', edgecolors='none', s=12, zorder=6)
        ys_etr.append(np.mean(ys))
    # plt.plot(data['etbc'], data['etbc'], 'k', zorder=7)
    # plt.scatter(0, 0, color='k', edgecolors='none', s=12, zorder=6, label='County mean values')
    res3 = linregress(data['etbc'], [conv[closest(conv, j)] for j in data['etos']])
    res3a = linregress(xs_et, ys_etr)
    plt.plot(np.sort(data['etbc']), res3.intercept + res3.slope * np.sort(data['etbc']),
             zorder=9, color='tab:pink', ls='dashed',
             label='y={:.2f}x+{:.1f} r^2: {:.2f}'.format(res3.slope, res3.intercept, res3.rvalue ** 2))
    plt.plot(np.sort(data['etbc']), res3a.intercept + res3a.slope * np.sort(data['etbc']),
             zorder=9, color='tab:purple', ls='dashed',
             label='y={:.2f}x+{:.1f} r^2: {:.2f}'.format(res3a.slope, res3a.intercept, res3a.rvalue ** 2))
    plt.grid(zorder=3)
    plt.xlim(22, 38)
    # plt.ylim(22, 38)
    # plt.xlim(22, 51)
    plt.ylim(22, 51)
    plt.plot([0, 51], [0, 51], 'k', zorder=7)
    plt.xlabel('DNRC')
    plt.ylabel('Gridmet ETr (alfalfa reference)')
    plt.legend()

    plt.tight_layout()


def plot_results_frac(data):
    """Create figure comparing ET, ETf, and consumptive use from two different methods."""

    xs = []
    ys = []
    xs_f = []
    ys_f = []
    xs_et = []
    ys_et = []
    for i in data['county'].unique():
        xs.append(data[data['county'] == i]['dnrc_cu'].mean())
        ys.append(data[data['county'] == i]['opnt_cu'].mean())
        xs_f.append(1.06*data[data['county'] == i]['mfs'].mean())
        ys_f.append(data[data['county'] == i]['etof'].mean())
        xs_et.append(data[data['county'] == i]['etbc'].mean())
        ys_et.append(data[data['county'] == i]['etos'].mean())
        # if data[data['county'] == i]['etos'].mean() < 28:
        #     print(i, data[data['county'] == i]['etos'].mean())

    # ET comparison
    plt.figure(figsize=(15, 5), dpi=200)

    plt.subplot(131)
    plt.title("Average Seasonal ET (inches)")
    for i in data['county'].unique():
        # if i in ['001', '023', '049', '069', '089']:
        #     plt.scatter(data[data['county'] == i]['etbc'], data[data['county'] == i]['etos'],
        #                 zorder=5, alpha=0.1, edgecolors='none', s=24, label=i)
        # if i == '029' or i == '047':
        #     print()
        # if i == '029':
        #     plt.scatter(data[data['county'] == i]['etbc'], data[data['county'] == i]['etos'],
        #                 zorder=8, alpha=0.1, edgecolors='none', color='tab:purple', s=24, label='029')
        # elif i == '047':
        #     plt.scatter(data[data['county'] == i]['etbc'], data[data['county'] == i]['etos'],
        #                 zorder=8, alpha=0.1, edgecolors='none', color='tab:green', s=24, label='047')
        # # elif i == '089':
        # #     plt.scatter(data[data['county'] == i]['etbc'], data[data['county'] == i]['etos'],
        # #                 zorder=8, alpha=0.1, edgecolors='none', color='tab:pink', s=24, label='089')
        # else:
        #     # plt.scatter(data[data['county'] == i]['etbc'], data[data['county'] == i]['etos'],
        #     #             zorder=5, alpha=0.1, edgecolors='none', color='tab:blue', s=24)
        #     plt.scatter(data[data['county'] == i]['etbc'], data[data['county'] == i]['etos'],
        #                 zorder=5, alpha=0.1, edgecolors='none', s=24, label=i)
        plt.scatter(data[data['county'] == i]['etbc'], data[data['county'] == i]['etos'],
                    zorder=5, alpha=0.1, edgecolors='none', color='tab:blue', s=24)
    # plt.plot(data['etbc'], data['etbc'], 'k', zorder=7)
    plt.scatter(xs_et, ys_et, color='k', edgecolors='none', s=12, zorder=6, label='County mean values')
    plt.grid(zorder=3)
    plt.xlim(22, 38)
    plt.ylim(22, 38)
    plt.plot([0, 40], [0, 40], 'k', zorder=7)
    plt.xlabel('DNRC')
    plt.ylabel('Gridmet ETo (grass reference)')
    # leg = plt.legend(fontsize=4)
    # for lh in leg.legendHandles:
    #     lh.set_alpha(1)
    plt.legend()

    # Etf comparison
    plt.subplot(132)
    plt.title("Average Seasonal Crop Coefficient")
    for i in data['county'].unique():
        if i == '029' or i == '047':
            print()
        # if i == '029':
        #     plt.scatter(1.06*data[data['county'] == i]['mfs'], data[data['county'] == i]['etof'],
        #                 zorder=8, alpha=0.1, edgecolors='none', color='tab:purple', s=24, label='029')
        # elif i == '047':
        #     plt.scatter(1.06*data[data['county'] == i]['mfs'], data[data['county'] == i]['etof'],
        #                 zorder=8, alpha=0.1, edgecolors='none', color='tab:green', s=24, label='047')
        else:
            plt.scatter(1.06*data[data['county'] == i]['mfs'], data[data['county'] == i]['etof'],
                        zorder=5, alpha=0.1, edgecolors='none', color='tab:blue', s=24)
    # plt.plot(1.06*data['mfs'], data['etof'], 'k', zorder=7)
    plt.scatter(xs_f, ys_f, color='k', edgecolors='none', s=12, zorder=6, label='County mean values')
    plt.grid(zorder=3)
    plt.xlim(0.2, 1.1)
    plt.ylim(0.2, 1.1)
    plt.plot([0, 2], [0, 2], 'k', zorder=7)
    plt.xlabel('DNRC (Kc)')
    plt.ylabel('OpenET (ETof)')
    plt.legend()

    mean = (data['dnrc_cu'] - data['opnt_cu']).mean()
    std = (data['dnrc_cu'] - data['opnt_cu']).std()

    # CU comparison
    plt.subplot(133)
    plt.title("Average Seasonal Consumptive Use (inches)")
    for i in data['county'].unique():
        if i == '029' or i == '047':
            print()
        # if i == '029':
        #     plt.scatter(data[data['county'] == i]['dnrc_cu'], data[data['county'] == i]['opnt_cu'],
        #                 zorder=8, alpha=0.1, edgecolors='none', color='tab:purple', s=24, label='029')
        # elif i == '047':
        #     plt.scatter(data[data['county'] == i]['dnrc_cu'], data[data['county'] == i]['opnt_cu'],
        #                 zorder=8, alpha=0.1, edgecolors='none', color='tab:green', s=24, label='047')
        else:
            plt.scatter(data[data['county'] == i]['dnrc_cu'], data[data['county'] == i]['opnt_cu'], zorder=5,
                        alpha=0.1, edgecolors='none', color='tab:blue', s=24)
    # plt.plot(data['opnt_cu'], data['opnt_cu'], 'k', zorder=7)
    plt.scatter(xs, ys, color='k', edgecolors='none', s=12, zorder=8, label='County mean values')
    plt.grid(zorder=3)
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.plot([0, 30], [0, 30], 'k', zorder=7)

    # # option 1
    # plt.plot([0, 30], [0-mean, 30-mean], 'tab:pink', zorder=7, label='Mean difference')
    # plt.plot([0, 30], [0-mean+std, 30-mean+std], 'tab:pink', zorder=7, linestyle='dashed')
    # plt.plot([0, 30], [0-mean-std, 30-mean-std], 'tab:pink', zorder=7, linestyle='dashed')
    # plt.fill_between([0, 30], [0-mean+std, 30-mean+std], [0-mean-std, 30-mean-std],
    #                  color='tab:pink', alpha=0.2, ec='none')

    # option 2
    plt.plot([0, 30], [0 - mean, 30 - mean], 'tab:pink', linestyle='dashed', zorder=7, label='Mean difference')
    plt.fill_between([0, 30], [0 - mean + std, 30 - mean + std], [0 - mean - std, 30 - mean - std],
                     color='tab:pink', alpha=0.2, ec='none')

    plt.xlabel('DNRC')
    plt.ylabel('OpenET')
    plt.legend()

    plt.tight_layout()


def plot_results_3(data, iwr):
    """Create figure comparing ET and consumptive use from two different methods."""
    # iwr = iwr[iwr['county'] != '101']  # 'data' does not have data for fields in county 101.
    # print(len(data))
    # print(len(iwr))

    xs = []
    ys = []
    xs_et = []
    ys_et = []
    for i in data['county'].unique():
        xs.append(iwr[iwr['county'] == i]['dnrc_cu'].mean())
        ys.append(data[data['county'] == i]['opnt_cu'].mean())
        xs_et.append(iwr[iwr['county'] == i]['etbc'].mean())
        ys_et.append(data[data['county'] == i]['etos'].mean())

    # ET comparison
    plt.figure(figsize=(10, 5), dpi=200)

    plt.subplot(121)
    plt.title("Average Seasonal ET (inches)")
    for i in data['county'].unique():
        # print(i, len(iwr[iwr['county'] == i]['etbc']), len(data[data['county'] == i]['etos']))
        # print(i, len(iwr[iwr.index.isin(data[data['county'] == i].index)]['etbc']),
        #       len(data[data['county'] == i]['etos']))
        plt.scatter(iwr[iwr.index.isin(data[data['county'] == i].index)]['etbc'], data[data['county'] == i]['etos'],
                    zorder=5, alpha=0.1, edgecolors='none', color='tab:blue', s=24)
    # plt.plot(data['etbc'], data['etbc'], 'k', zorder=7)
    plt.scatter(xs_et, ys_et, color='k', edgecolors='none', s=12, zorder=6, label='County mean values')
    plt.grid(zorder=3)
    plt.xlim(22, 38)
    plt.ylim(22, 38)
    plt.plot([0, 40], [0, 40], 'k', zorder=7)
    plt.xlabel('DNRC')
    plt.ylabel('Gridmet ETo (grass reference)')
    plt.legend()

    mean = (iwr[iwr['county'] != '101']['dnrc_cu'] - data['opnt_cu']).mean()
    std = (iwr[iwr['county'] != '101']['dnrc_cu'] - data['opnt_cu']).std()

    # CU comparison
    plt.subplot(122)
    plt.title("Average Seasonal Consumptive Use (inches)")
    for i in data['county'].unique():
        plt.scatter(iwr[iwr.index.isin(data[data['county'] == i].index)]['dnrc_cu'],
                    data[data['county'] == i]['opnt_cu'], zorder=5,
                    alpha=0.1, edgecolors='none', color='tab:blue', s=24)  # label="{} ({})".format(COUNTIES[i], i),
    # plt.plot(data['opnt_cu'], data['opnt_cu'], 'k', zorder=7)
    plt.scatter(xs, ys, color='k', edgecolors='none', s=12, zorder=8, label='County mean values')
    plt.grid(zorder=3)
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.plot([0, 30], [0, 30], 'k', zorder=7)

    # # option 1
    # plt.plot([0, 30], [0-mean, 30-mean], 'tab:pink', zorder=7, label='Mean difference')
    # plt.plot([0, 30], [0-mean+std, 30-mean+std], 'tab:pink', zorder=7, linestyle='dashed')
    # plt.plot([0, 30], [0-mean-std, 30-mean-std], 'tab:pink', zorder=7, linestyle='dashed')
    # plt.fill_between([0, 30], [0-mean+std, 30-mean+std], [0-mean-std, 30-mean-std],
    #                  color='tab:pink', alpha=0.2, ec='none')

    # option 2
    plt.plot([0, 30], [0 - mean, 30 - mean], 'tab:pink', linestyle='dashed', zorder=7, label='Mean difference')
    plt.fill_between([0, 30], [0 - mean + std, 30 - mean + std], [0 - mean - std, 30 - mean - std],
                     color='tab:pink', alpha=0.2, ec='none')

    plt.xlabel('DNRC')
    plt.ylabel('OpenET')
    plt.legend()

    plt.tight_layout()


def plot_results_4(data, iwr):
    """Create figure comparing DNRC climate ET vs Gridmet ET by county."""
    # iwr = iwr[iwr['county'] != '101']  # 'data' does not have data for fields in county 101.
    # print(len(data))
    # print(len(iwr))

    xs = []
    ys = []
    xs_et = []
    ys_et = []
    for i in data['county'].unique():
        xs.append(iwr[iwr['county'] == i]['dnrc_cu'].mean())
        ys.append(data[data['county'] == i]['opnt_cu'].mean())
        xs_et.append(iwr[iwr['county'] == i]['etbc'].mean())
        ys_et.append(data[data['county'] == i]['etos'].mean())

    county_list = data['county'].unique()

    clrs = mpl.colormaps['tab20'].colors

    # ET comparison
    plt.figure(figsize=(20, 5), dpi=200)
    for j in np.arange(4):
        plt.subplot(1, 4, j+1)
        plt.title("Average Seasonal ET (inches)")
        plt.scatter(xs_et[13*j:13*(j+1)+1], ys_et[13*j:13*(j+1)+1], color='k', edgecolors='none', s=14, marker='s',
                    zorder=2, label='County mean values')
        for i in range(13):
            cnty = county_list[13*j+i]
            plt.scatter(iwr[iwr.index.isin(data[data['county'] == cnty].index)]['etbc'],
                        data[data['county'] == cnty]['etos'],
                        zorder=5, alpha=0.2, edgecolors='none', s=24, color=clrs[i],
                        label="{} ({})".format(COUNTIES[cnty], cnty))
            plt.scatter(xs_et[13*j+i], ys_et[13*j+i], color=clrs[i], edgecolors='k', s=14, marker='s', zorder=6)
        plt.grid(zorder=3)
        plt.xlim(22, 38)
        plt.ylim(22, 38)
        plt.plot([0, 40], [0, 40], 'k', zorder=7)
        plt.xlabel('DNRC (climate)')
        plt.ylabel('Gridmet ETo (grass reference)')
        plt.legend()

    plt.tight_layout()


def plot_results_5(data):
    """Create figure comparing DNRC ET vs Gridmet ET by county."""
    # iwr = iwr[iwr['county'] != '101']  # 'data' does not have data for fields in county 101.
    # print(len(data))
    # print(len(iwr))

    xs = []
    ys = []
    xs_et = []
    ys_et = []
    for i in data['county'].unique():
        xs.append(data[data['county'] == i]['dnrc_cu'].mean())
        ys.append(data[data['county'] == i]['opnt_cu'].mean())
        xs_et.append(data[data['county'] == i]['etbc'].mean())
        ys_et.append(data[data['county'] == i]['etos'].mean())

    county_list = data['county'].unique()

    # clrs = mpl.colormaps['tab20'].colors
    # clrs = mpl.colormaps['Set3'].colors + mpl.colormaps['tab20c'].colors[8:10]
    # clrs = mpl.colormaps['tab20b'].colors[::4] + mpl.colormaps['tab20c'].colors[::4] + mpl.colormaps['tab10'].colors
    clrs = mpl.colormaps['Dark2'].colors + mpl.colormaps['tab10'].colors

    # ET comparison
    plt.figure(figsize=(20, 5), dpi=200)
    for j in np.arange(4):
        plt.subplot(1, 4, j+1)
        plt.title("Average Seasonal ET (inches)")
        plt.scatter(xs_et[13*j:13*(j+1)+1], ys_et[13*j:13*(j+1)+1], color='k', edgecolors='none', s=14, marker='s',
                    zorder=2, label='County mean values')
        for i in range(13):
            cnty = county_list[13*j+i]
            plt.scatter(data[data.index.isin(data[data['county'] == cnty].index)]['etbc'],
                        data[data['county'] == cnty]['etos'],
                        zorder=5, alpha=0.2, edgecolors='none', s=24, color=clrs[i],
                        label="{} ({})".format(COUNTIES[cnty], cnty))
            plt.scatter(xs_et[13*j+i], ys_et[13*j+i], color=clrs[i], edgecolors='k', s=14, marker='s', zorder=6)
        plt.grid(zorder=3)
        plt.xlim(22, 38)
        plt.ylim(22, 38)
        plt.plot([0, 40], [0, 40], 'k', zorder=7)
        plt.xlabel('DNRC')
        plt.ylabel('Gridmet ETo (grass reference)')
        # plt.legend(fontsize=6)
        leg = plt.legend(fontsize=6)
        for lh in leg.legendHandles:
            lh.set_alpha(1)

    plt.tight_layout()


def plot_results_6(data):
    """Create figure comparing DNRC ET vs Gridmet CU by county."""
    # iwr = iwr[iwr['county'] != '101']  # 'data' does not have data for fields in county 101.
    # print(len(data))
    # print(len(iwr))

    xs = []
    ys = []
    xs_et = []
    ys_et = []
    for i in data['county'].unique():
        xs.append(data[data['county'] == i]['dnrc_cu'].mean())
        ys.append(data[data['county'] == i]['opnt_cu'].mean())
        xs_et.append(data[data['county'] == i]['etbc'].mean())
        ys_et.append(data[data['county'] == i]['etos'].mean())

    county_list = data['county'].unique()

    # # How to 'make' extended colormaps.
    # clrs = mpl.colormaps['tab20'].colors
    clrs = mpl.colormaps['Dark2'].colors + mpl.colormaps['tab10'].colors

    # Also works, but no.
    # clrs = mpl.colormaps['viridis'](np.linspace(0, 1, 13))

    # ET comparison
    plt.figure(figsize=(20, 5), dpi=200)
    for j in np.arange(4):
        plt.subplot(1, 4, j+1)
        plt.title("Average Seasonal Consumptive Use (inches)")
        plt.scatter(xs[13*j:13*(j+1)+1], ys[13*j:13*(j+1)+1], color='k', edgecolors='none', s=14, marker='s',
                    zorder=2, label='County mean values')
        for i in range(13):
            cnty = county_list[13*j+i]
            plt.scatter(data[data.index.isin(data[data['county'] == cnty].index)]['dnrc_cu'],
                        data[data['county'] == cnty]['opnt_cu'],
                        zorder=5, alpha=0.2, edgecolors='none', s=24, color=clrs[i],
                        label="{} ({})".format(COUNTIES[cnty], cnty))
            plt.scatter(xs[13*j+i], ys[13*j+i], color=clrs[i], edgecolors='k', s=14, marker='s', zorder=6)
        plt.grid(zorder=3)
        plt.xlim(0, 30)
        plt.ylim(0, 30)
        plt.plot([0, 40], [0, 40], 'k', zorder=7)
        plt.xlabel('DNRC')
        plt.ylabel('OpentET')
        # plt.legend(fontsize=6)
        leg = plt.legend(fontsize=6)
        for lh in leg.legendHandles:
            lh.set_alpha(1)

    plt.tight_layout()


def three_scatterplots_1(data):
    """Create figure comparing ET and consumptive use from two different methods. Also, dif in ET vs dif in CU. """
    xs = []
    ys = []
    xs_et = []
    ys_et = []
    for i in data['county'].unique():
        xs.append(data[data['county'] == i]['dnrc_cu'].mean())
        ys.append(data[data['county'] == i]['opnt_cu'].mean())
        xs_et.append(data[data['county'] == i]['etbc'].mean())
        ys_et.append(data[data['county'] == i]['etos'].mean())
    dif_cu = - np.asarray(xs) + np.asarray(ys)
    dif_et = - np.asarray(xs_et) + np.asarray(ys_et)

    # counting how many fields are in each category.
    top_left = 0
    top_right = 0
    lower_left = 0
    lower_right = 0
    for fid, i in data.iterrows():
        # print(i)
        if i['etos'] <= i['etbc']:
            if i['opnt_cu'] > i['dnrc_cu']:
                top_left += 1
            else:
                lower_left += 1
        else:
            if i['opnt_cu'] > i['dnrc_cu']:
                top_right += 1
            else:
                lower_right += 1
    print("Should match:", len(data), top_left + top_right + lower_left + lower_right)
    print("Numbers", top_left, top_right, lower_left, lower_right)
    print("{:.2f}%, {:.2f}%".format(100 * top_left / len(data), 100 * top_right / len(data)))
    print("{:.2f}%, {:.2f}%".format(100 * lower_left / len(data), 100 * lower_right / len(data)))

    # ET comparison
    plt.figure(figsize=(15, 5), dpi=200)

    plt.subplot(131)
    plt.title("Average Seasonal ET (inches)")
    for i in data['county'].unique():
        plt.scatter(data[data['county'] == i]['etbc'], data[data['county'] == i]['etos'],
                    zorder=5, alpha=0.1, edgecolors='none', color='tab:blue', s=24)
    # plt.plot(data['etbc'], data['etbc'], 'k', zorder=7)
    plt.scatter(xs_et, ys_et, color='k', edgecolors='none', s=12, zorder=6, label='County mean values')
    plt.grid(zorder=3)
    plt.xlim(22, 38)
    plt.ylim(22, 38)
    plt.plot([0, 40], [0, 40], 'k', zorder=7)
    plt.xlabel('DNRC')
    plt.ylabel('Gridmet ETo (grass reference)')
    plt.legend()

    mean = (data['dnrc_cu'] - data['opnt_cu']).mean()
    std = (data['dnrc_cu'] - data['opnt_cu']).std()

    # CU comparison
    plt.subplot(132)
    plt.title("Average Seasonal Consumptive Use (inches)")
    for i in data['county'].unique():
        plt.scatter(data[data['county'] == i]['dnrc_cu'], data[data['county'] == i]['opnt_cu'], zorder=5,
                    alpha=0.1, edgecolors='none', color='tab:blue', s=24)  # label="{} ({})".format(COUNTIES[i], i),
    # plt.plot(data['opnt_cu'], data['opnt_cu'], 'k', zorder=7)
    plt.scatter(xs, ys, color='k', edgecolors='none', s=12, zorder=8, label='County mean values')
    plt.grid(zorder=3)
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.plot([0, 30], [0, 30], 'k', zorder=7)

    # # option 1
    # plt.plot([0, 30], [0-mean, 30-mean], 'tab:pink', zorder=7, label='Mean difference')
    # plt.plot([0, 30], [0-mean+std, 30-mean+std], 'tab:pink', zorder=7, linestyle='dashed')
    # plt.plot([0, 30], [0-mean-std, 30-mean-std], 'tab:pink', zorder=7, linestyle='dashed')
    # plt.fill_between([0, 30], [0-mean+std, 30-mean+std], [0-mean-std, 30-mean-std],
    #                  color='tab:pink', alpha=0.2, ec='none')

    # option 2
    plt.plot([0, 30], [0 - mean, 30 - mean], 'tab:pink', linestyle='dashed', zorder=7, label='Mean difference')
    plt.fill_between([0, 30], [0 - mean + std, 30 - mean + std], [0 - mean - std, 30 - mean - std],
                     color='tab:pink', alpha=0.2, ec='none')

    plt.xlabel('DNRC')
    plt.ylabel('OpenET')
    plt.legend()

    plt.subplot(133)
    plt.title("Differences in ET vs CU (RS - DNRC)")
    for i in data['county'].unique():
        plt.scatter(- data[data['county'] == i]['etbc'] + data[data['county'] == i]['etos'],
                    - data[data['county'] == i]['dnrc_cu'] + data[data['county'] == i]['opnt_cu'],
                    zorder=5, alpha=0.1, edgecolors='none', color='tab:blue', s=24)
    # plt.plot(data['etbc'], data['etbc'], 'k', zorder=7)
    plt.scatter(dif_et, dif_cu, color='k', edgecolors='none', s=12, zorder=6, label='County mean values')
    plt.grid(zorder=3)
    plt.xlim(-25, 10)
    plt.ylim(-25, 10)
    # plt.plot([-10, 40], [-10, 40], 'k', zorder=7)
    plt.hlines(0, -25, 10, 'k', zorder=7)
    plt.vlines(0, -25, 10, 'k', zorder=7)
    plt.text(-12.5, 5, "{:.1f}%".format(100 * top_left / len(data)), ha='center', va='center', zorder=7)
    plt.text(5, 5, "{:.1f}%".format(100 * top_right / len(data)), ha='center', va='center', zorder=7)
    plt.text(-12.5, -12.5, "{:.1f}%".format(100 * lower_left / len(data)), ha='center', va='center', zorder=7)
    plt.text(5, -12.5, "{:.1f}%".format(100 * lower_right / len(data)), ha='center', va='center', zorder=7)
    plt.xlabel('Difference in ET (inches)')
    plt.ylabel('Difference in CU (inches)')
    plt.legend()

    plt.tight_layout()


def three_scatterplots_2(data, iwr):
    """Create figure comparing ET and consumptive use from two different methods. Also, dif in ET vs dif in CU. """
    iwr = iwr[iwr['county'] != '101']  # 'data' does not have data for fields in county 101.
    # print(len(data))
    # print(len(iwr))

    xs = []
    ys = []
    xs_et = []
    ys_et = []
    for i in data['county'].unique():
        xs.append(iwr[iwr['county'] == i]['dnrc_cu'].mean())
        ys.append(data[data['county'] == i]['opnt_cu'].mean())
        xs_et.append(iwr[iwr['county'] == i]['etbc'].mean())
        ys_et.append(data[data['county'] == i]['etos'].mean())
    dif_cu = - np.asarray(xs) + np.asarray(ys)
    dif_et = - np.asarray(xs_et) + np.asarray(ys_et)

    # counting how many fields are in each category.
    top_left = 0
    top_right = 0
    lower_left = 0
    lower_right = 0

    # print(len(data), len(iwr))
    # print(data.columns, iwr.columns)

    # for fid, i in data.iterrows():
    # for num in range(len(data)):
    for ind in data.index:
        i = data.loc[ind]
        j = iwr.loc[ind]
        # print('i', i)
        # print('j', j)
        if i['etos'] <= j['etbc']:
            if i['opnt_cu'] > j['dnrc_cu']:
                top_left += 1
            else:
                lower_left += 1
        else:
            if i['opnt_cu'] > j['dnrc_cu']:
                top_right += 1
            else:
                lower_right += 1
    print("Should match:", len(data), top_left + top_right + lower_left + lower_right)
    print("Numbers", top_left, top_right, lower_left, lower_right)
    print("{:.2f}%, {:.2f}%".format(100 * top_left / len(data), 100 * top_right / len(data)))
    print("{:.2f}%, {:.2f}%".format(100 * lower_left / len(data), 100 * lower_right / len(data)))

    # ET comparison
    plt.figure(figsize=(15, 5), dpi=200)

    plt.subplot(131)
    plt.title("Average Seasonal ET (inches)")
    for i in data['county'].unique():
        # print(i, len(iwr[iwr['county'] == i]['etbc']), len(data[data['county'] == i]['etos']))
        # print(i, len(iwr[iwr.index.isin(data[data['county'] == i].index)]['etbc']),
        #       len(data[data['county'] == i]['etos']))
        plt.scatter(iwr[iwr.index.isin(data[data['county'] == i].index)]['etbc'], data[data['county'] == i]['etos'],
                    zorder=5, alpha=0.1, edgecolors='none', color='tab:blue', s=24)
    # plt.plot(data['etbc'], data['etbc'], 'k', zorder=7)
    plt.scatter(xs_et, ys_et, color='k', edgecolors='none', s=12, zorder=6, label='County mean values')
    plt.grid(zorder=3)
    plt.xlim(22, 38)
    plt.ylim(22, 38)
    plt.plot([0, 40], [0, 40], 'k', zorder=7)
    plt.xlabel('DNRC')
    plt.ylabel('Gridmet ETo (grass reference)')
    plt.legend()

    mean = (iwr[iwr['county'] != '101']['dnrc_cu'] - data['opnt_cu']).mean()
    std = (iwr[iwr['county'] != '101']['dnrc_cu'] - data['opnt_cu']).std()

    # CU comparison
    plt.subplot(132)
    plt.title("Average Seasonal Consumptive Use (inches)")
    for i in data['county'].unique():
        plt.scatter(iwr[iwr.index.isin(data[data['county'] == i].index)]['dnrc_cu'],
                    data[data['county'] == i]['opnt_cu'], zorder=5,
                    alpha=0.1, edgecolors='none', color='tab:blue', s=24)  # label="{} ({})".format(COUNTIES[i], i),
    # plt.plot(data['opnt_cu'], data['opnt_cu'], 'k', zorder=7)
    plt.scatter(xs, ys, color='k', edgecolors='none', s=12, zorder=8, label='County mean values')
    plt.grid(zorder=3)
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.plot([0, 30], [0, 30], 'k', zorder=7)

    # # option 1
    # plt.plot([0, 30], [0-mean, 30-mean], 'tab:pink', zorder=7, label='Mean difference')
    # plt.plot([0, 30], [0-mean+std, 30-mean+std], 'tab:pink', zorder=7, linestyle='dashed')
    # plt.plot([0, 30], [0-mean-std, 30-mean-std], 'tab:pink', zorder=7, linestyle='dashed')
    # plt.fill_between([0, 30], [0-mean+std, 30-mean+std], [0-mean-std, 30-mean-std],
    #                  color='tab:pink', alpha=0.2, ec='none')

    # option 2
    plt.plot([0, 30], [0 - mean, 30 - mean], 'tab:pink', linestyle='dashed', zorder=7, label='Mean difference')
    plt.fill_between([0, 30], [0 - mean + std, 30 - mean + std], [0 - mean - std, 30 - mean - std],
                     color='tab:pink', alpha=0.2, ec='none')

    plt.xlabel('DNRC')
    plt.ylabel('OpenET')
    plt.legend()

    plt.subplot(133)
    plt.title("Differences in ET vs CU (RS - DNRC)")
    for i in data['county'].unique():
        # print(i, len(iwr[iwr['county'] == i]['etbc']), len(data[data['county'] == i]['etos']))
        # print(i, len(iwr[iwr.index.isin(data[data['county'] == i].index)]['etbc']),
        #       len(data[data['county'] == i]['etos']))
        plt.scatter(- iwr[iwr.index.isin(data[data['county'] == i].index)]['etbc'] + data[data['county'] == i]['etos'],
                    - iwr[iwr.index.isin(data[data['county'] == i].index)]['dnrc_cu'] + data[data['county'] == i]['opnt_cu'],
                    zorder=5, alpha=0.1, edgecolors='none', color='tab:blue', s=24)
    # plt.plot(data['etbc'], data['etbc'], 'k', zorder=7)
    plt.scatter(dif_et, dif_cu, color='k', edgecolors='none', s=12, zorder=6, label='County mean values')
    plt.grid(zorder=3)
    plt.xlim(-25, 10)
    plt.ylim(-25, 10)
    # plt.plot([-10, 40], [-10, 40], 'k', zorder=7)
    plt.hlines(0, -25, 10, 'k', zorder=7)
    plt.vlines(0, -25, 10, 'k', zorder=7)
    plt.text(-12.5, 5, "{:.1f}%".format(100 * top_left / len(data)), ha='center', va='center', zorder=7)
    plt.text(5, 5, "{:.1f}%".format(100 * top_right / len(data)), ha='center', va='center', zorder=7)
    plt.text(-12.5, -12.5, "{:.1f}%".format(100 * lower_left / len(data)), ha='center', va='center', zorder=7)
    plt.text(5, -12.5, "{:.1f}%".format(100 * lower_right / len(data)), ha='center', va='center', zorder=7)
    plt.xlabel('Difference in ET (inches)')
    plt.ylabel('Difference in CU (inches)')
    plt.legend()

    plt.tight_layout()


def density_scatter(x, y, ax=None, sort=True, bins=20, **kwargs):
    """ Scatter plot colored by 2d histogram.
    Function from Guillaume's answer at
    https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density.
    """
    if ax is None:
        fig, ax = plt.subplots()
    data_e, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data_e, np.vstack([x, y]).T,
                method="splinef2d", bounds_error=False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, **kwargs)

    norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    # cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
    # cbar.ax.set_ylabel('Density')

    return ax


def three_densityplots_1(data):
    """Create figure comparing ET and consumptive use from two different methods. Also, dif in ET vs dif in CU. """
    # Calculating county means
    xs = []
    ys = []
    xs_et = []
    ys_et = []
    for i in data['county'].unique():
        xs.append(data[data['county'] == i]['dnrc_cu'].mean())
        ys.append(data[data['county'] == i]['opnt_cu'].mean())
        xs_et.append(data[data['county'] == i]['etbc'].mean())
        ys_et.append(data[data['county'] == i]['etos'].mean())
    dif_cu = - np.asarray(xs) + np.asarray(ys)
    dif_et = - np.asarray(xs_et) + np.asarray(ys_et)

    # counting how many fields are in each category for the 3rd plot.
    top_left = 0
    top_right = 0
    lower_left = 0
    lower_right = 0
    for fid, i in data.iterrows():
        # print(i)
        if i['etos'] <= i['etbc']:
            if i['opnt_cu'] > i['dnrc_cu']:
                top_left += 1
            else:
                lower_left += 1
        else:
            if i['opnt_cu'] > i['dnrc_cu']:
                top_right += 1
            else:
                lower_right += 1
    print("Should match:", len(data), top_left + top_right + lower_left + lower_right)
    print("Numbers", top_left, top_right, lower_left, lower_right)
    print("{:.2f}%, {:.2f}%".format(100 * top_left / len(data), 100 * top_right / len(data)))
    print("{:.2f}%, {:.2f}%".format(100 * lower_left / len(data), 100 * lower_right / len(data)))

    # New method
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

    # ET comparison - why does only this one look pixelated/bad?
    axs[0].set_title("Average Seasonal ET (inches)")
    density_scatter(data['etbc'], data['etos'], ax=axs[0], bins=[30, 30], zorder=5, edgecolors='none', s=24)
    # axs[0].scatter(data['etbc'], data['etos'], zorder=5, alpha=0.1, edgecolors='none', color='tab:blue', s=24)
    # axs[0].scatter(xs_et, ys_et, color='k', edgecolors='none', s=12, zorder=6, label='County mean values')
    axs[0].grid(zorder=3)
    axs[0].set_xlim(22, 38)
    axs[0].set_ylim(22, 38)
    axs[0].plot([0, 40], [0, 40], 'k', zorder=7)
    axs[0].set_xlabel('DNRC')
    axs[0].set_ylabel('Gridmet ETo (grass reference)')

    # CU comparison
    axs[1].set_title("Average Seasonal Consumptive Use (inches)")
    density_scatter(data['dnrc_cu'], data['opnt_cu'], ax=axs[1], bins=[30, 30], zorder=5, edgecolors='none', s=24)
    # axs[1].scatter(data['dnrc_cu'], data['opnt_cu'], zorder=5, alpha=0.1, edgecolors='none', color='tab:blue', s=24)
    # axs[1].scatter(xs, ys, color='k', edgecolors='none', s=12, zorder=8, label='County mean values')
    axs[1].grid(zorder=3)
    axs[1].set_xlim(0, 30)
    axs[1].set_ylim(0, 30)
    axs[1].plot([0, 30], [0, 30], 'k', zorder=7)
    axs[1].set_xlabel('DNRC')
    axs[1].set_ylabel('OpenET')
    # axs[1].legend()

    # Difference in ET/CU
    axs[2].set_title("Differences in ET vs CU (RS - DNRC)")
    density_scatter(- data['etbc'] + data['etos'], - data['dnrc_cu'] + data['opnt_cu'],
                    ax=axs[2], bins=[30, 30], zorder=5, edgecolors='none', s=24)
    # axs[2].scatter(- data['etbc'] + data['etos'], - data['dnrc_cu'] + data['opnt_cu'],
    #                zorder=5, alpha=0.1, edgecolors='none', color='tab:blue', s=24)
    # axs[2].scatter(dif_et, dif_cu, color='k', edgecolors='none', s=12, zorder=6, label='County mean values')
    axs[2].grid(zorder=3)
    axs[2].set_xlim(-25, 10)
    axs[2].set_ylim(-25, 10)
    axs[2].hlines(0, -25, 10, 'k', zorder=7)
    axs[2].vlines(0, -25, 10, 'k', zorder=7)
    # axs[2].text(-12.5, 5, "{:.1f}%".format(100 * top_left / len(data)), ha='center', va='center', zorder=7)
    # axs[2].text(5, 5, "{:.1f}%".format(100 * top_right / len(data)), ha='center', va='center', zorder=7)
    # axs[2].text(-12.5, -12.5, "{:.1f}%".format(100 * lower_left / len(data)), ha='center', va='center', zorder=7)
    # axs[2].text(5, -12.5, "{:.1f}%".format(100 * lower_right / len(data)), ha='center', va='center', zorder=7)
    axs[2].set_xlabel('Difference in ET (inches)')
    axs[2].set_ylabel('Difference in CU (inches)')
    # axs[2].legend()

    plt.tight_layout()


def three_densityplots_2(data, iwr):
    """Create figure comparing ET and consumptive use from two different methods. Also, dif in ET vs dif in CU. """
    iwr = iwr[iwr['county'] != '101']  # 'data' does not have data for fields in county 101.
    iwr = iwr[iwr.index.isin(data.index)]
    # print(len(data))
    # print(len(iwr))

    xs = []
    ys = []
    xs_et = []
    ys_et = []
    for i in data['county'].unique():
        xs.append(iwr[iwr['county'] == i]['dnrc_cu'].mean())
        ys.append(data[data['county'] == i]['opnt_cu'].mean())
        xs_et.append(iwr[iwr['county'] == i]['etbc'].mean())
        ys_et.append(data[data['county'] == i]['etos'].mean())
    dif_cu = - np.asarray(xs) + np.asarray(ys)
    dif_et = - np.asarray(xs_et) + np.asarray(ys_et)

    # counting how many fields are in each category.
    top_left = 0
    top_right = 0
    lower_left = 0
    lower_right = 0

    print(len(data), len(iwr))
    # print(data.columns, iwr.columns)

    # for fid, i in data.iterrows():
    # for num in range(len(data)):
    for ind in data.index:
        i = data.loc[ind]
        j = iwr.loc[ind]
        # print('i', i)
        # print('j', j)
        if i['etos'] <= j['etbc']:
            if i['opnt_cu'] > j['dnrc_cu']:
                top_left += 1
            else:
                lower_left += 1
        else:
            if i['opnt_cu'] > j['dnrc_cu']:
                top_right += 1
            else:
                lower_right += 1
    print("Should match:", len(data), top_left + top_right + lower_left + lower_right)
    print("Numbers", top_left, top_right, lower_left, lower_right)
    print("{:.2f}%, {:.2f}%".format(100 * top_left / len(data), 100 * top_right / len(data)))
    print("{:.2f}%, {:.2f}%".format(100 * lower_left / len(data), 100 * lower_right / len(data)))

    # New method
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

    # ET comparison - why does only this one look pixelated/bad?
    axs[0].set_title("Average Seasonal ET (inches)")
    density_scatter(iwr['etbc'], data['etos'], ax=axs[0], bins=[30, 30], zorder=5, edgecolors='none', s=24)
    # axs[0].scatter(data['etbc'], data['etos'], zorder=5, alpha=0.1, edgecolors='none', color='tab:blue', s=24)
    # axs[0].scatter(xs_et, ys_et, color='k', edgecolors='none', s=12, zorder=6, label='County mean values')
    axs[0].grid(zorder=3)
    axs[0].set_xlim(22, 38)
    axs[0].set_ylim(22, 38)
    axs[0].plot([0, 40], [0, 40], 'k', zorder=7)
    axs[0].set_xlabel('DNRC')
    axs[0].set_ylabel('Gridmet ETo (grass reference)')

    # CU comparison
    axs[1].set_title("Average Seasonal Consumptive Use (inches)")
    density_scatter(iwr['dnrc_cu'], data['opnt_cu'], ax=axs[1], bins=[30, 30], zorder=5, edgecolors='none', s=24)
    # axs[1].scatter(data['dnrc_cu'], data['opnt_cu'], zorder=5, alpha=0.1, edgecolors='none', color='tab:blue', s=24)
    # axs[1].scatter(xs, ys, color='k', edgecolors='none', s=12, zorder=8, label='County mean values')
    axs[1].grid(zorder=3)
    axs[1].set_xlim(0, 30)
    axs[1].set_ylim(0, 30)
    axs[1].plot([0, 30], [0, 30], 'k', zorder=7)
    axs[1].set_xlabel('DNRC')
    axs[1].set_ylabel('OpenET')
    # axs[1].legend()

    import matplotlib.patheffects as pe

    # Difference in ET/CU
    axs[2].set_title("Differences in ET vs CU (RS - DNRC)")
    density_scatter(- iwr['etbc'] + data['etos'], - iwr['dnrc_cu'] + data['opnt_cu'],
                    ax=axs[2], bins=[30, 30], zorder=5, edgecolors='none', s=24)
    # axs[2].scatter(- data['etbc'] + data['etos'], - data['dnrc_cu'] + data['opnt_cu'],
    #                zorder=5, alpha=0.1, edgecolors='none', color='tab:blue', s=24)
    # axs[2].scatter(dif_et, dif_cu, color='k', edgecolors='none', s=12, zorder=6, label='County mean values')
    axs[2].grid(zorder=3)
    axs[2].set_xlim(-25, 10)
    axs[2].set_ylim(-25, 10)
    axs[2].hlines(0, -25, 10, 'k', zorder=7)
    axs[2].vlines(0, -25, 10, 'k', zorder=7)
    axs[2].text(-12.5, 5, "{:.1f}%".format(100 * top_left / len(data)), ha='center', va='center', zorder=7,
                path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.7)])
    axs[2].text(5, 5, "{:.1f}%".format(100 * top_right / len(data)), ha='center', va='center', zorder=7,
                path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.7)])
    axs[2].text(-12.5, -12.5, "{:.1f}%".format(100 * lower_left / len(data)), ha='center', va='center', zorder=7,
                path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.7)])
    axs[2].text(5, -12.5, "{:.1f}%".format(100 * lower_right / len(data)), ha='center', va='center', zorder=7,
                path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.7)])
    # axs[2].text(0.5, 0.5, "test", size=20,
    #         color='white',
    #         path_effects=[pe.withStroke(linewidth=4, foreground="white")])
    axs[2].set_xlabel('Difference in ET (inches)')
    axs[2].set_ylabel('Difference in CU (inches)')
    # axs[2].legend()

    plt.tight_layout()


def time_series_plot(ts_data, iwr, var):
    static_avg = iwr.groupby('county').mean()

    vari = {'etbc': 'etbc', 'etos': 'etbc', 'dnrc_cu': 'dnrc_cu', 'opnt_cu': 'dnrc_cu'}

    avg_ts = np.asarray([ts_data[ts_data['st_year'] == i][var].mean() for i in range(1987, 2018, 3)])
    # for i in range(1987, 2018, 3):
    #     avg.append(ts_data[ts_data['st_year'] == i][var].mean())
    county_avg = [static_avg[static_avg.index == i][vari[var]].iloc[0] for i in COUNTIES.keys()]
    # print(county_avg)

    # plt.figure()
    # bins = np.arange(23, 36)+0.5
    # plt.hist(county_avg, bins=bins, zorder=3, rwidth=0.95)
    # plt.title('Distribution of County-averaged DNRC Consumptive Use from IWR Climate')
    # plt.xlabel('Consumptive Use Estimate (inches)')
    # plt.ylabel('count')
    # plt.grid(zorder=1)

    # plt.figure()
    # for i in COUNTIES.keys():
    #     plt.plot(ts_data[ts_data['county'] == i]['st_year'] + 3, ts_data[ts_data['county'] == i][var],
    #              alpha=0.3)
    #     if i == '001':
    #         plt.hlines(static_avg[static_avg.index == i][vari[var]], 1987, 2003,
    #                    color='k', linestyle='dashed', alpha=0.2, label='DNRC (climate)')
    #     else:
    #         plt.hlines(static_avg[static_avg.index == i][vari[var]], 1987, 2003,
    #                    color='k', linestyle='dashed', alpha=0.2)
    #     plt.hlines(static_avg[static_avg.index == i][vari[var]], 2003, 2023,
    #                color='k', linestyle='dotted', alpha=0.2)
    # plt.plot(ts_data[ts_data['county'] == '001']['st_year'] + 3, avg_ts,
    #          alpha=1, label='All county average')
    # plt.hlines(static_avg[vari[var]].mean(), 1987, 2003,
    #            color='k', linestyle='dashed', alpha=1)
    # plt.hlines(static_avg[vari[var]].mean(), 2003, 2023,
    #            color='k', linestyle='dotted', alpha=1)
    # plt.grid()
    # plt.xlabel('year')
    # plt.ylabel(var)
    # plt.legend()

    code = {'etbc': 'DNRC ET (inches, met data minus climate)', 'etos': 'ET (inches, Gridmet minus climate)',
            'dnrc_cu': 'DNRC CU (inches, met data minus climate)',
            'opnt_cu': 'Consumptive Use (inches, DNRC minus climate)'}

    code_title = {'etbc': 'DNRC ET', 'etos': 'ET (DNRC vs Gridmet ETo)',
                  'dnrc_cu': 'DNRC Consumptive Use', 'opnt_cu': 'Consumptive Use (DNRC vs OpenET)'}

    plt.figure()
    plt.title("Difference in {} over time".format(code_title[var]))
    for i in COUNTIES.keys():
        if i == '001':
            plt.plot(ts_data[ts_data['county'] == i]['st_year'] + 3,
                     ts_data[ts_data['county'] == i][var] - static_avg[static_avg.index == i][vari[var]].iloc[0],
                     alpha=0.2, color='tab:blue', label='Counties')
        else:
            plt.plot(ts_data[ts_data['county'] == i]['st_year'] + 3,
                     ts_data[ts_data['county'] == i][var] - static_avg[static_avg.index == i][vari[var]].iloc[0],
                     alpha=0.2, color='tab:blue')
    plt.plot(ts_data[ts_data['county'] == '001']['st_year'] + 3, avg_ts - static_avg[vari[var]].mean(),
             alpha=1, label='All county average', color='tab:pink')
    plt.hlines(0, 1990, 2020, 'k')
    plt.grid()
    plt.xlabel('year')
    plt.ylabel((code[var]))
    plt.legend()


def time_series_plot_2(ts_data, iwr):
    """ Plot the average field fraction irrigated over time in each county and overall. """
    static_avg = iwr.groupby('county').mean()

    avg_ts = np.asarray([ts_data[ts_data['st_year'] == i]['frac_irr'].mean() for i in range(1987, 2018, 3)])
    # for i in range(1987, 2018, 3):
    #     avg.append(ts_data[ts_data['st_year'] == i][var].mean())
    # county_avg = [static_avg[static_avg.index == i][vari[var]].iloc[0] for i in COUNTIES.keys()]
    # print(county_avg)

    # plt.figure()
    # bins = np.arange(23, 36)+0.5
    # plt.hist(county_avg, bins=bins, zorder=3, rwidth=0.95)
    # plt.title('Distribution of County-averaged DNRC Consumptive Use from IWR Climate')
    # plt.xlabel('Consumptive Use Estimate (inches)')
    # plt.ylabel('count')
    # plt.grid(zorder=1)

    plt.figure()
    plt.title('Average fraction of field classified as irrigated (IrrMapper)')
    for i in COUNTIES.keys():
        plt.plot(ts_data[ts_data['county'] == i]['st_year'] + 3, ts_data[ts_data['county'] == i]['frac_irr'],
                 alpha=0.3)
    plt.plot(ts_data[ts_data['county'] == '001']['st_year'] + 3, avg_ts,
             alpha=1, label='All county average', color='k')
    plt.grid()
    plt.xlabel('year')
    plt.ylabel('fraction irrigated')
    plt.legend()
    plt.ylim(0, 1)


def time_series_plot_1(ts_data, iwr):
    """ Make a 2x2 grid figure of 4 variables' differences from the DNRC climate estimate. """
    static_avg = iwr.groupby('county').mean()

    vari = {'etbc': 'etbc', 'dnrc_cu': 'dnrc_cu', 'etos': 'etbc', 'opnt_cu': 'dnrc_cu'}

    code = {'etbc': 'DNRC ET (inches, met data minus climate)', 'etos': 'ET (inches, Gridmet minus climate)',
            'dnrc_cu': 'DNRC CU (inches, met data minus climate)',
            'opnt_cu': 'Consumptive Use (inches, DNRC minus climate)'}

    code_title = {'etbc': 'DNRC ET', 'etos': 'ET (DNRC vs Gridmet ETo)',
                  'dnrc_cu': 'DNRC Consumptive Use', 'opnt_cu': 'Consumptive Use (DNRC vs OpenET)'}

    plt.figure(figsize=(12, 8))
    j = 1
    for k in vari.keys():
        plt.subplot(2, 2, j)
        avg_ts = np.asarray([ts_data[ts_data['st_year'] == i][k].mean() for i in range(1987, 2018, 3)])
        # for i in range(1987, 2018, 3):
        #     avg.append(ts_data[ts_data['st_year'] == i][var].mean())
        county_avg = [static_avg[static_avg.index == i][vari[k]].iloc[0] for i in COUNTIES.keys()]
        # print(county_avg)

        plt.title("Difference in {} over time".format(code_title[k]))
        for i in COUNTIES.keys():
            if i == '001':
                plt.plot(ts_data[ts_data['county'] == i]['st_year'] + 3,
                         ts_data[ts_data['county'] == i][k] - static_avg[static_avg.index == i][vari[k]].iloc[0],
                         alpha=0.2, color='tab:blue', label='Counties')
            else:
                plt.plot(ts_data[ts_data['county'] == i]['st_year'] + 3,
                         ts_data[ts_data['county'] == i][k] - static_avg[static_avg.index == i][vari[k]].iloc[0],
                         alpha=0.2, color='tab:blue')
        plt.plot(ts_data[ts_data['county'] == '001']['st_year'] + 3, avg_ts - static_avg[vari[k]].mean(),
                 alpha=1, label='All county average', color='tab:pink')
        plt.hlines(0, 1990, 2020, 'k')
        plt.grid()
        if j > 2:
            plt.xlabel('year')
        else:
            plt.ylim(-8, 8)
        plt.ylabel((code[k]))
        if j == 2:
            plt.legend()
        elif j == 3:
            plt.ylim(-8, 8)
        elif j == 4:
            plt.ylim(-14, 2)

        j += 1
    plt.tight_layout()


def stats(data, selection=()):
    """ Not very interesting, needs work. """
    if len(selection) == 0:
        counties = data['county'].unique()
    else:
        counties = selection

    all_results = {'neg_cu': data['opnt_cu'].lt(0).sum()}  # / len(data)

    results = pd.DataFrame(columns=['neg_cu'], index=counties)

    for i in counties:
        # fields with a negative CU
        results.loc[i, 'neg_cu'] = data[data['county'] == i]['opnt_cu'].lt(0).sum()  # / len(data[data['county'] == i])

    return all_results, results


def crop_stuff():
    # # Too big of a pain to drop tiny fields?
    # crops1 = gpd.read_file('C:/Users/CND571/Documents/Data/NASS_cropland_2007_2023_SID.csv', index_col='FID')

    crops = pd.read_csv('C:/Users/CND571/Documents/Data/NASS_cropland_2007_2023_SID.csv', index_col='FID')
    # print(crops['.geo'].iloc[0])
    from shapely import wkt
    from shapely import geometry
    import json
    # crops['.geo'] = ["{} {}".format(i[9:16], i[33:-2]) for i in crops['.geo']]
    # crops['.geo'] = [i.replace('[', '(') for i in crops['.geo']]
    # crops['.geo'] = [i.replace(']', ')') for i in crops['.geo']]
    crops['.geo'] = [geometry.shape(json.loads(i)) for i in crops['.geo']]
    # print(crops['.geo'].iloc[0])
    # crops['.geo'] = crops['.geo'].apply(wkt.loads)
    crops1 = gpd.GeoDataFrame(crops, geometry='.geo')
    crops1['.geo'] = crops1['.geo'].set_crs(4326)
    crops1['.geo'] = crops1['.geo'].to_crs(5071)
    # print(crops1.columns)
    # print(crops1)
    # print(crops.columns)
    # print(crops)

    crops['area'] = crops1['.geo'].area / 4047
    print(crops['area'])

    crops.drop(columns=['system:index', 'COUNTYNAME', 'COUNTY_NO', 'ITYPE',
                        'MAPPEDBY', 'SOURCECODE', 'USAGE', '.geo'], inplace=True)
    crops.index = crops.index.rename('fid')
    crops = crops.fillna(0)
    crops = crops.round().astype(int)
    crops['mean'] = crops.mode(axis='columns')[0].astype(int)
    # print(crops.head())

    all_crops = set()
    for i in range(2007, 2024):
        temp = crops[str(i)].unique()
        for j in temp:
            all_crops.add(j)
    # print(len(all_crops))  # 54 "crops" identified across all years
    # print(sorted(list(all_crops)))
    #
    # print(len(crops['mean'].unique()))  # 30 dominant "crops" identified
    # print(sorted(list(crops['mean'].unique())))

    crop_key = {0: 'Unknown', 1: 'Corn', 4: 'Sorghum', 5: 'Soybeans', 21: 'Barley', 22: 'Durum Wheat',
                23: 'Spring Wheat', 24: 'Winter Wheat', 28: 'Oats', 29: 'Millet', 31: 'Canola', 36: 'Alfalfa',
                37: 'Other Hay/Non Alfalfa', 41: 'Sugarbeets', 42: 'Dry Beans', 43: 'Potatoes', 53: 'Peas',
                57: 'Herbs', 59: 'Sod/Grass Seed', 61: 'Fallow/Idle Cropland', 111: 'Open Water',
                121: 'Developed/Open Space', 122: 'Developed/Low Intensity', 123: 'Developed/Medium Intensity',
                142: 'Evergreen Forest', 152: 'Shrubland', 176: 'Grassland/Pasture', 190: 'Woody Wetlands',
                195: 'Herbaceous Wetlands', 205: 'Triticale'}

    # gen_crop_key = {'Unknown': 'Unknown', 'Corn': 'Barley, Corn, and Wheat', 'Sorghum': 'Other Crops',
    #                 'Soybeans': 'Other Crops', 'Barley': 'Barley, Corn, and Wheat',
    #                 'Durum Wheat': 'Barley, Corn, and Wheat', 'Spring Wheat': 'Barley, Corn, and Wheat',
    #                 'Winter Wheat': 'Barley, Corn, and Wheat', 'Oats': 'Other Crops', 'Millet': 'Other Crops',
    #                 'Canola': 'Other Crops', 'Alfalfa': 'Alfalfa', 'Other Hay/Non Alfalfa': 'Other Hay/Non Alfalfa',
    #                 'Sugarbeets': 'Other Crops', 'Dry Beans': 'Other Crops', 'Potatoes': 'Other Crops',
    #                 'Peas': 'Other Crops', 'Herbs': 'Other Crops', 'Sod/Grass Seed': 'Other Crops',
    #                 'Fallow/Idle Cropland': 'Fallow/Idle Cropland', 'Open Water': 'Natural Land Cover',
    #                 'Developed/Open Space': 'Developed', 'Developed/Low Intensity': 'Developed',
    #                 'Developed/Medium Intensity': 'Developed', 'Evergreen Forest': 'Natural Land Cover',
    #                 'Shrubland': 'Natural Land Cover', 'Grassland/Pasture': 'Grassland/Pasture',
    #                 'Woody Wetlands': 'Natural Land Cover', 'Herbaceous Wetlands': 'Natural Land Cover',
    #                 'Triticale': 'Other Crops'}  # Barley, corn and wheat grouped

    gen_crop_key = {'Unknown': 'Unknown', 'Corn': 'Corn', 'Sorghum': 'Other Crops',
                    'Soybeans': 'Other Crops', 'Barley': 'Barley',
                    'Durum Wheat': 'Wheat', 'Spring Wheat': 'Wheat',
                    'Winter Wheat': 'Wheat', 'Oats': 'Other Crops', 'Millet': 'Other Crops',
                    'Canola': 'Other Crops', 'Alfalfa': 'Alfalfa', 'Other Hay/Non Alfalfa': 'Other Hay/Non Alfalfa',
                    'Sugarbeets': 'Other Crops', 'Dry Beans': 'Other Crops', 'Potatoes': 'Other Crops',
                    'Peas': 'Other Crops', 'Herbs': 'Other Crops', 'Sod/Grass Seed': 'Other Crops',
                    'Fallow/Idle Cropland': 'Fallow/Idle Cropland', 'Open Water': 'Natural Land Cover',
                    'Developed/Open Space': 'Developed', 'Developed/Low Intensity': 'Developed',
                    'Developed/Medium Intensity': 'Developed', 'Evergreen Forest': 'Natural Land Cover',
                    'Shrubland': 'Natural Land Cover', 'Grassland/Pasture': 'Grassland/Pasture',
                    'Woody Wetlands': 'Natural Land Cover', 'Herbaceous Wetlands': 'Natural Land Cover',
                    'Triticale': 'Other Crops'}  # Wheat grouped

    crops['crop'] = [crop_key[i] for i in crops['mean']]
    crops['gen_crop'] = [gen_crop_key[i] for i in crops['crop']]
    # print(crops.head())

    # print(crops['crop'].value_counts())

    # plt.figure()
    # plt.bar(crops['crop'].value_counts().index, crops['crop'].value_counts().values, zorder=3)
    # plt.grid(zorder=1)
    # plt.yscale('log')
    # plt.tick_params(axis='x', rotation=90)
    # plt.tight_layout()

    # plt.figure()
    # plt.title('SID Average Crops')
    # plt.pie(crops['crop'].value_counts().values, labels=crops['crop'].value_counts().index, autopct='%1.1f%%')

    # plt.figure()
    # plt.title('SID Average Crops')
    # plt.pie(crops['gen_crop'].value_counts().values, labels=crops['gen_crop'].value_counts().index, autopct='%1.1f%%')

    # Add column for difference in consumptive use estimate
    # print(por_data.head())
    # print(len(por_data))
    por_data['dif'] = por_data['dnrc_cu'] - por_data['opnt_cu']
    crop_cu = pd.merge(por_data[['dnrc_cu', 'opnt_cu', 'dif']], crops[['mean', 'crop', 'gen_crop', 'area']],
                       how='left', on='fid')
    crop_cu['county'] = [i[:3] for i in crop_cu.index]
    # print(crop_cu)

    types = crop_cu['gen_crop'].value_counts().index

    clrs = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    # print(51273 - crop_cu['mean'].count())  # No missing crop data
    # print(crops[crops['crop'] == 'Unknown'].sort_index())  # Unknown appears to just/mostly be tiny fields.

    # fig, ax = plt.subplots()
    # plt.title('SID Fields by NASS Average Crop Type/Land Cover')
    # plt.bar(crop_cu['gen_crop'].value_counts().index, crop_cu['gen_crop'].value_counts().values,
    #         label=crop_cu['gen_crop'].value_counts().index, color=clrs, zorder=3)
    # plt.grid(zorder=1)
    # # plt.tick_params(axis='x', rotation=45, ha='left')
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # # plt.xticklabels(ha='left')
    # plt.ylabel('Number of Fields')
    # plt.tight_layout()

    # fig, ax = plt.subplots()
    # plt.title('SID Fields by NASS Average Crop Type/Land Cover')
    # plt.bar(crop_cu['gen_crop'].value_counts().index, crop_cu['gen_crop'].value_counts().values,
    #         label=crop_cu['gen_crop'].value_counts().index, color=clrs, zorder=3)
    # plt.grid(zorder=1)
    # # plt.tick_params(axis='x', rotation=45, ha='left')
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # # plt.xticklabels(ha='left')
    # plt.ylabel('Number of Fields')
    # plt.tight_layout()

    plt.figure()
    plt.title('SID Fields by NASS Average Crop/Land Cover Type')
    plt.bar(crop_cu['gen_crop'].value_counts().index, crop_cu['gen_crop'].value_counts().values,
            label=crop_cu['gen_crop'].value_counts().index, color=clrs, zorder=3)
    plt.grid(which='both', zorder=1)
    plt.xticks([])
    # plt.yticks()
    # plt.xticks(color='w')
    # plt.xticks(alpha=0)
    # plt.tick_params(axis='x', rotation=45, ha='left')
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # plt.xticklabels(ha='left')
    plt.ylabel('Number of Fields')
    plt.xlabel('Crop/Land Cover Type')
    plt.tight_layout()
    plt.legend()

    ys = [crop_cu[crop_cu['gen_crop'] == i]['area'].sum() for i in crop_cu['gen_crop'].value_counts().index]

    plt.figure()
    plt.title('SID Fields by NASS Average Crop/Land Cover Type')
    plt.bar(crop_cu['gen_crop'].value_counts().index, ys,
            label=crop_cu['gen_crop'].value_counts().index, color=clrs, zorder=3)
    plt.grid(which='both', zorder=1)
    plt.xticks([])
    # plt.yticks()
    # plt.xticks(color='w')
    # plt.xticks(alpha=0)
    # plt.tick_params(axis='x', rotation=45, ha='left')
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # plt.xticklabels(ha='left')
    plt.ylabel('Irrigated Acreage')
    plt.xlabel('Crop/Land Cover Type')
    plt.tight_layout()
    plt.legend()

    # plt.figure()
    # plt.title('SID Fields by NASS Average Crop Type/Land Cover')
    # plt.pie(crop_cu['gen_crop'].value_counts().values, labels=crop_cu['gen_crop'].value_counts().index,
    #         autopct='%1.1f%%')
    # plt.tight_layout()
    #
    # plt.figure()
    # plt.title('SID Fields by NASS Average Crop Type/Land Cover')
    # plt.pie(ys, labels=crop_cu['gen_crop'].value_counts().index,
    #         autopct='%1.1f%%')
    # plt.tight_layout()

    # # # # # # # # # # # #
    # Number of fields
    species = COUNTIES.keys()
    weight_counts = {}
    for crop in crop_cu['gen_crop'].value_counts().index:
        vals = []
        one_crop = crop_cu[crop_cu['gen_crop'] == crop]
        for i in species:
            vals.append(one_crop[one_crop['county'] == i]['gen_crop'].count())  # num fields
        weight_counts[crop] = np.array(vals)

    width = 0.9

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(52)

    for boolean, weight_count in weight_counts.items():
        ax.bar(species, weight_count, width, label=boolean, bottom=bottom, zorder=3)
        bottom += weight_count

    ax.grid(zorder=0)
    ax.set_title('SID fields by County and Crop type')
    # ax.legend(loc="upper right")
    ax.tick_params(axis='x', rotation=90)
    ax.set_xlabel('County')
    ax.set_ylabel('Number of SID Fields')
    ax.legend(title='Crop/Land Cover Type', ncol=2)

    # Acreage
    species = COUNTIES.keys()
    weight_counts = {}
    for crop in crop_cu['gen_crop'].value_counts().index:
        vals = []
        one_crop = crop_cu[crop_cu['gen_crop'] == crop]
        for i in species:
            vals.append(one_crop[one_crop['county'] == i]['area'].sum())
        weight_counts[crop] = np.array(vals)

    width = 0.9

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(52)

    for boolean, weight_count in weight_counts.items():
        ax.bar(species, weight_count, width, label=boolean, bottom=bottom, zorder=3)
        bottom += weight_count

    ax.grid(zorder=0)
    ax.set_title('SID fields by County and Crop type')
    # ax.legend(loc="upper right")
    ax.tick_params(axis='x', rotation=90)
    ax.set_xlabel('County')
    ax.set_ylabel('Irrigated Acreage')
    ax.legend(title='Crop/Land Cover Type', ncol=2)
    # # # # # # # # # # # #

    # plt.figure()
    # plt.title('SID Fields by NASS Average Crop/Land Cover Type')
    # plt.bar(crop_cu['gen_crop'].value_counts().index, crop_cu['gen_crop'].value_counts().values,
    #         label=crop_cu['gen_crop'].value_counts().index, color=clrs, zorder=3)
    # plt.grid(zorder=1)
    # # plt.xticks([])
    # # plt.xticks(color='w')
    # plt.xticks(alpha=0)
    # # plt.tick_params(axis='x', rotation=45, ha='left')
    # # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # # plt.xticklabels(ha='left')
    # plt.ylabel('Number of Fields')
    # plt.xlabel('Crop/Land Cover Type')
    # plt.tight_layout()
    # plt.legend()

    # plt.figure()
    # bins = np.arange(-10, 25)
    # for i in range(10):
    #     temp = crop_cu[crop_cu['gen_crop'] == types[i]]
    #     per_overest = 100 * len(temp[temp['dif'] > 0]) / len(temp)
    #
    #     # # Most crop types are distributed across most counties
    #     # print(types[i])
    #     # print("{} counties: ".format(len(temp['county'].unique())), sorted(temp['county'].unique()))
    #
    #     plt.subplot(2, 5, i+1)
    #     plt.title(types[i])
    #     plt.hist(crop_cu[crop_cu['gen_crop'] == types[i]]['dif'], bins=bins, zorder=3,
    #              label="{:.0f}%".format(per_overest))
    #     plt.vlines(crop_cu[crop_cu['gen_crop'] == types[i]]['dif'].mean(), 0, 70, zorder=5,
    #                label="{:.2f}".format(crop_cu[crop_cu['gen_crop'] == types[i]]['dif'].mean()),
    #                color='tab:orange')
    #     plt.grid(zorder=1)
    #     plt.legend()

    # plt.figure()
    # plt.title("Consumptive Use Estimate (Inches)")
    # for i in range(10):
    #     temp = crop_cu[crop_cu['gen_crop'] == types[i]]
    #     plt.scatter(temp['dnrc_cu'], temp['opnt_cu'], zorder=3,  edgecolors='none', alpha=0.1, label=types[i])
    #     plt.grid(zorder=1)
    #     plt.xlim(0, 30)
    #     plt.ylim(0, 30)
    #     plt.plot([0, 30], [0, 30], 'k')
    #     plt.xlabel('DNRC')
    #     plt.ylabel('OpenET')
    #     plt.legend()
    #     for lh in plt.legend().legendHandles:
    #         lh.set_alpha(1)

    # plt.figure(figsize=(17.5, 7))
    # plt.suptitle("Consumptive Use Estimates (Inches) by NASS Average Crop Type/Land Cover")
    # bins = np.arange(-10, 25)
    # for i in range(10):
    #     temp = crop_cu[crop_cu['gen_crop'] == types[i]]
    #     plt.subplot(2, 5, i+1)
    #     plt.scatter(temp['dnrc_cu'], temp['opnt_cu'], zorder=5,  color=clrs[i], edgecolors='none',
    #                 alpha=0.1, label="{} ({:.1f}%)".format(types[i], 100*len(temp)/len(crop_cu)))
    #     plt.scatter(crop_cu['dnrc_cu'], crop_cu['opnt_cu'], zorder=3, color='lightgrey', edgecolors='none',
    #                 alpha=0.1, label='All Crops')
    #     plt.grid(zorder=1)
    #     plt.xlim(0, 30)
    #     plt.ylim(0, 30)
    #     plt.plot([0, 30], [0, 30], 'k', zorder=4)
    #     if i == 0 or i == 5:
    #         plt.ylabel('OpenET')
    #     if i > 4:
    #         plt.xlabel('DNRC')
    #     leg = plt.legend(loc='upper left')
    #     for lh in leg.legendHandles:
    #         lh.set_alpha(1)
    # plt.tight_layout()


def gridmet_refet(con, save=True):
    gridmets = pd.read_sql("SELECT DISTINCT gfid FROM field_data", con)
    # print(gridmets.iloc[0].values[0])
    # grd = pd.read_sql("SELECT strftime('%m', DATE(date)), eto_mm FROM gridmet_ts WHERE gfid={} AND "
    #                   "date(date) BETWEEN date('2002-07-01') AND date('2002-07-28')".format(gridmets.iloc[0].values[0]), con)
    # print(grd)

    grd = pd.read_sql("SELECT date, eto_mm, etr_mm FROM gridmet_ts WHERE gfid=90397.0", con)
    print(len(grd))
    grd.index = pd.to_datetime(grd['date'])
    # grd['month'] = grd.index.month
    # grd_m = grd.groupby('month').sum()
    grd = grd.drop(columns='date')
    grd_m = grd.resample('ME').mean()
    print(grd)
    print(grd_m)

    grd1 = pd.read_sql("SELECT date, eto_mm, etr_mm FROM gridmet_ts WHERE gfid=90397.0 AND "
                       "strftime('%m', date(date)) IN ('04', '05', '06', '07', '08', '09')", con)
    grd1.index = pd.to_datetime(grd1['date'])
    grd1 = grd1.drop(columns='date')
    grd1['month'] = grd1.index.month
    grd1['year'] = grd1.index.year
    print(grd1)
    grd1_m = grd1.resample('ME').mean()
    # grd1_m = grd1.groupby('month').mean()
    grd1_y = grd1.groupby('year').sum()
    print(grd1_y)

    plt.figure(figsize=(15, 5))
    plt.title('Gridmet Reference ET in Helena Valley (GFID 90397)')
    # plt.plot(grd['eto_mm'] / 25.4, label='ETo')
    # plt.plot(grd['etr_mm'] / 25.4, label='ETr')
    plt.plot(grd_m['eto_mm'] / 25.4, label='ETo')
    plt.plot(grd_m['etr_mm'] / 25.4, label='ETr')
    plt.plot(grd1_m['etr_mm'] / grd1_m['eto_mm'])
    plt.ylabel('Inches/Day')
    plt.grid()
    plt.legend()

    # plt.figure()
    # plt.title('Gridmet Reference ET in Helena Valley (GFID 90397)')
    # plt.scatter(grd['eto_mm'] / 25.4, grd['etr_mm'] / 25.4, zorder=3)
    # plt.plot([0, 0.4], [0, 0.4], 'k', zorder=4)
    # plt.ylabel('ETr')
    # plt.xlabel('ETo')
    # plt.grid(zorder=2)
    # plt.legend()

    # etos = []
    # etrs = []
    # for i in tqdm(gridmets['gfid'], total=len(gridmets)):
    #     # only select dates in growing season
    #     grd = pd.read_sql("SELECT date, eto_mm, etr_mm FROM gridmet_ts WHERE gfid={} AND "
    #                       "strftime('%m', date(date)) IN ('04', '05', '06', '07', '08', '09')".format(i), con)
    #     # print(len(grd))
    #     grd.index = pd.to_datetime(grd['date'])
    #     grd['year'] = grd.index.year
    #     grd_y = grd.groupby('year').sum()
    #     etos.append(grd_y['eto_mm'].mean() / 25.4)
    #     etrs.append(grd_y['etr_mm'].mean() / 25.4)
    #
    # if save:
    #     np.savetxt("C:/Users/CND571/Documents/Data/all_gm_eto_20240709.txt", etos)
    #     np.savetxt("C:/Users/CND571/Documents/Data/all_gm_etr_20240709.txt", etrs)

    etos = np.loadtxt("C:/Users/CND571/Documents/Data/all_gm_eto_20240709.txt")
    etrs = np.loadtxt("C:/Users/CND571/Documents/Data/all_gm_etr_20240709.txt")

    res1 = linregress(etos, etrs)
    # mean = np.mean(etrs - etos)
    std = np.std(etrs - etos)
    print(std)
    # print(len(etos))
    # print(len(pd.Series(etos).unique()))
    conv = dict(zip(etos, etrs))
    # print(conv)

    plt.figure(figsize=(5, 5), dpi=200)
    plt.title('Gridmet Ref ET (inches)')
    plt.scatter(etos, etrs, zorder=5, alpha=0.1, edgecolors='none')
    plt.scatter(grd1_y['eto_mm'] / 25.4, grd1_y['etr_mm'] / 25.4, zorder=5, edgecolors='none')
    plt.scatter(grd1_y['eto_mm'].mean() / 25.4, grd1_y['etr_mm'].mean() / 25.4, zorder=5, edgecolors='none')
    plt.plot([20, max([max(etos), max(etrs)])], [20, max([max(etos), max(etrs)])], 'k', zorder=7)
    plt.plot(np.sort(etos), res1.intercept + res1.slope * np.sort(etos), zorder=9, color='tab:pink', ls='dashed',
             label='y={:.2f}x{:.1f} r^2: {:.2f}'.format(res1.slope, res1.intercept, res1.rvalue ** 2))
    # There is a zero value, can't use minimum.
    plt.fill_between(np.sort(etos), res1.intercept + res1.slope * np.sort(etos) - std,
                     res1.intercept + res1.slope * np.sort(etos) + std,
                     color='tab:pink', alpha=0.2, ec='none', zorder=4)
    plt.xlim(20, max([max(etos), max(etrs)]))
    plt.ylim(20, max([max(etos), max(etrs)]))
    plt.grid(zorder=3)
    plt.xlabel('ETo (grass)')
    plt.ylabel('ETr (alfalfa)')
    plt.legend()

    return conv


if __name__ == '__main__':

    if os.path.exists('F:/FileShare'):
        main_dir = 'F:/FileShare/openet_pilot'
    else:
        main_dir = 'F:/openet_pilot'

    # fix county list
    # note: 101 should have data, but etof didn't load, under investigation
    for key in ['011', '025', '101', '109']:
        COUNTIES.pop(key, None)

    # iwr_data = pd.read_csv(os.path.join(main_dir, "iwr_cu_results_mf1997-2006.csv"),
    #                        dtype={'county': str}, index_col='fid')
    # print(iwr_data.columns)

    # static_avg = iwr_data.groupby('county').mean()
    # print('static_avg')
    # print(static_avg)
    # print(static_avg.index)
    # print(static_avg[static_avg.index == '001']['dnrc_cu'])

    # # Establish sqlite connection
    # conec = sqlite3.connect(os.path.join(main_dir, "opnt_analysis_03042024_Copy.db"))  # full project
    # # conec = sqlite3.connect("C:/Users/CND571/Documents/Data/random_05082024.db")  # test
    #
    # # extract data
    # # gm_ct = gridmet_ct(conec)
    # # print(gm_ct)
    # # sum_data(conec, irrmapper=1, save="C:/Users/CND571/Documents/Data")
    #
    # # gm_data = gridmet_refet(conec)
    #
    # # checking time format
    # cursor = conec.cursor()
    # # # This works. Why do other dates not? (ex: '2021-06-28'
    # # cursor.execute("SELECT time, fid, acres FROM opnt_etof WHERE date(time)=date('{}')".format('1985-01-01'))
    # # # cursor.execute("SELECT time, fid, acres FROM opnt_etof")
    # # print(cursor.fetchone())
    # # cursor.execute("SELECT count(fid) FROM opnt_etof WHERE date(time)=date('{}')".format('1985-01-01'))
    # # # Okay, that's the right number of results, I think. Too small?
    # # print(cursor.fetchone())
    #
    # # cursor.execute("SELECT DISTINCT fid, acres FROM opnt_etof")
    # # # cursor.execute("SELECT time, fid, acres FROM opnt_etof")
    # # print(cursor.fetchone())
    # # stuff = cursor.fetchall()
    # # print(len(stuff))
    # # cursor.execute("SELECT DISTINCT count(fid) FROM opnt_etof")
    # # Okay, that's the right number of results, I think. Too small?
    # # print(cursor.fetchone())
    #
    # areas = pd.read_sql("SELECT DISTINCT fid, acres FROM opnt_etof", conec)
    # areas.to_csv("C:/Users/CND571/Documents/Data/SID_areas.csv")
    # # areas = pd.read_sql("SELECT fid, acres FROM opnt_etof WHERE time={}".format("6/1/2020"), conec)
    # print(areas)
    #
    # # time_series_data(conec, main_dir)
    #
    # # por_data = sum_data(conec)
    #
    # # close connection
    # conec.close()

    # por_data_im = pd.read_csv("C:/Users/CND571/Documents/Data/cu_results_1987_2023_im1_mf1997-2006.csv",
    #                           dtype={'county': str}, index_col='fid')
    por_data = pd.read_csv("C:/Users/CND571/Documents/Data/cu_results_1987_2023_im0_mf1997-2006.csv",
                           dtype={'county': str}, index_col='fid')
    iwr_data = pd.read_csv("C:/Users/CND571/Documents/Data/iwr_cu_results_mf1997-2006.csv",
                           dtype={'county': str}, index_col='fid')
    # print(por_data.columns)
    # print(iwr_data.columns)

    # ts_df = pd.read_csv(os.path.join(main_dir, 'ts_6yrs_im0_mf1997-2006.csv'),
    #                     dtype={'county': str})  # , index_col=['county', 'st_year'])
    # print(ts_df)
    # print(ts_df.columns)

    # areas = pd.read_csv("C:/Users/CND571/Documents/Data/SID_areas.csv", index_col='fid')
    areas = pd.read_csv("C:/Users/CND571/Documents/Data/SID_areas.csv")  # starts at 56531
    areas = areas.drop_duplicates(subset=['fid'])  # this goes to 39214 when index is assigned first, stays the same otherwise.
    areas.index = areas['fid']
    areas = areas.drop(columns=['Unnamed: 0', 'fid'])

    # print(areas)

    # time_series_plot(ts_df, iwr_data, var='etbc')
    # time_series_plot(ts_df, iwr_data, var='dnrc_cu')
    # time_series_plot(ts_df, iwr_data, var='etos')
    # time_series_plot(ts_df, iwr_data, var='opnt_cu')
    # time_series_plot_1(ts_df, iwr_data)
    # time_series_plot_2(ts_df, iwr_data)

    # print(len(por_data_im['dnrc_cu'].unique()))
    # print(len(por_data['dnrc_cu'].unique()))
    # print(len(iwr_data['dnrc_cu'].unique()))

    # all_stats, county_stats = stats(por_data)

    # neg_cu = por_data['opnt_cu'].lt(0).sum()
    # neg_cu_fids = por_data[por_data['opnt_cu'].lt(0)].index
    # print(neg_cu, neg_cu_fids)

    # neg_cu_dif = (por_data['dnrc_cu']-por_data['opnt_cu']).lt(0).sum()
    # print(neg_cu_dif)

    # plot_results_2(por_data)  # this looks good. Daily values for DNRC
    # three_scatterplots_1(por_data)  # same as line above, but adds comparison of differences in ET vs CU.
    # three_densityplots_1(por_data)  # same as line above, but is a density plot w/ different colors.
    # plot_results_2_1(por_data, gm_data)
    # plot_results_frac(por_data) # includes etof, doesn't look great.
    # plot_results_3(por_data, iwr_data)  # climate values for DNRC
    # three_scatterplots_2(por_data, iwr_data)  # same as line above, but adds comparison of differences in ET vs CU.
    three_densityplots_2(por_data, iwr_data)  # same as line above, but is a density plot w/ different colors.
    # county_hist(por_data, iwr_data, selection=('029',), val=500)
    # county_hist(por_data, iwr_data, selection=('047',))  # , val=500)
    # county_hist(por_data, iwr_data, selection=('067',))
    # dif_hist(por_data)  # this looks good
    # dif_hist_1(por_data, iwr_data)  # this has (most) of the data I need
    # dif_hist_2(por_data, iwr_data)
    # scatter(por_data)  # this is nice
    # all_hist_1(por_data, iwr_data)  # single plot, all fields

    # plot_results_2(por_data_im)  # woah! don't use irrmapper. 2 scatterplot ET/CU, all fields, w/ irrmapper
    # plot_results_3(por_data_im, iwr_data)  # 2 scatterplots ET/CU, all fields, climate
    # county_hist(por_data_im, iwr_data, selection=('001',))  # 2 plots ET/CU, histograms w/ 3 sources for one county
    # dif_hist(por_data_im)  # histogram of difference in CU (daily), all fields
    # dif_hist_1(por_data_im, iwr_data)  # histogram of difference in CU (climate), all fields
    # dif_hist_2(por_data_im, iwr_data)  # overlayed histogram of difference in CU (daily,climate vs OpenET), all fields
    # scatter(por_data_im)  # scatterplot fo CU w/ county means and big circles, not good, shouldn't use irrmapper
    # all_hist_1(por_data_im, iwr_data)  # overlayed CU histogram, all fields

    # all_hist_2(por_data, por_data_im)  # overlayed CU histogram, all fields

    # plot_results_4(por_data, iwr_data)  # 4 scatterplots w/ ref ET split by county (climate)
    # plot_results_5(por_data)  # 4 scatterplots w/ ref ET split by county (daily)
    # plot_results_6(por_data)  # 4 scatterplots w/ CU split by county

    # county_hist_1(por_data, iwr_data)  # 1 CU histogram for each of 52 counties
    # county_hist_1(por_data_im, iwr_data)
    # county_scatter_1_cu(por_data, por_data_im, iwr_data)  # 1 CU scatterplot for each of 52 counties
    # county_scatter_1_et(por_data, por_data_im, iwr_data)  # 1 ET scatterplot for each of 52 counties

    # # Adding some crop stuff
    # crop_stuff()  # crs is broken... I guess I couldn't just ignore the problem.

    # print(areas)
    # print(por_data)
    # print(iwr_data)

    # # putting all the data together for a total statistic
    # every = pd.merge(por_data['opnt_cu'], iwr_data['dnrc_cu'], how='inner', on='fid')
    # every = pd.merge(every, areas, how='inner', on='fid')
    #
    # every['opnt_cu_af'] = (every['opnt_cu']/12.0) * every['acres']
    # every['dnrc_cu_af'] = (every['dnrc_cu']/12.0) * every['acres']
    # every['dif'] = every['opnt_cu_af'] - every['dnrc_cu_af']
    # print(every)
    # print(every.sum())
    # print()
    # per_dif = 100 * (every['dnrc_cu_af'].sum() - every['opnt_cu_af'].sum())/every['dnrc_cu_af'].sum()
    # print("Total reduction in statewide seasonal consumptive use estimate: {:.2f}%".format(per_dif))

    plt.show()

# ========================= EOF ====================================================================
