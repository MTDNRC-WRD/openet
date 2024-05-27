
import os

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

from run_all import COUNTIES


def sum_data(con, start=1987, end=2023, irrmapper=0, mf_periods='1997-2006', static_too=False, save=""):
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


def gridmet_ct(con, pront=False):
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
        # plt.title("Average Seasonal ET (in)")
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


def plot_results_1(data):
    """Create figure comparing ET and consumptive use from two different methods."""
    # ET comparison
    plt.figure(figsize=(10, 5), dpi=200)

    plt.subplot(121)
    plt.title("Average Seasonal ET (in)")
    for i in data['county'].unique():
        plt.scatter(data[data['county'] == i]['etbc'], data[data['county'] == i]['etos'],
                    zorder=5,
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
                    label="{} ({})".format(COUNTIES[i], i))
    plt.plot(data['opnt_cu'], data['opnt_cu'], 'k', zorder=4, label="1:1")
    plt.grid(zorder=3)
    plt.xlabel('DNRC')
    plt.ylabel('OpenET')
    plt.legend(title='County')

    plt.tight_layout()


def county_hist(data, iwr, selection=()):
    if len(selection) == 0:
        counties = data['county'].unique()
    else:
        counties = selection
    for i in counties:
        plt.figure(figsize=(10, 5), dpi=200)

        plt.subplot(121)
        plt.title("Average Seasonal ET (in)")
        # plt.hist(data[data['county'] == i]['etbc'], label='DNRC', zorder=5)
        # plt.hist(data[data['county'] == i]['etos'], label='Gridmet ETo', zorder=5)
        plt.hist([data[data['county'] == i]['etbc'], data[data['county'] == i]['etos'],
                  iwr[iwr['county'] == i]['etbc']],
                 label=['DNRC', 'Gridmet ETo', 'IWR climate'], zorder=5)
        plt.grid(zorder=1)
        plt.legend()

        # CU comparison
        plt.subplot(122)
        plt.title("Average Seasonal Consumptive Use (in)")
        # plt.hist(data[data['county'] == i]['dnrc_cu'], label='DNRC', zorder=5)
        # plt.hist(data[data['county'] == i]['opnt_cu'], label='OpenET', zorder=5)
        plt.hist([data[data['county'] == i]['dnrc_cu'], data[data['county'] == i]['opnt_cu'],
                  iwr[iwr['county'] == i]['dnrc_cu']],
                 label=['DNRC', 'OpenET', 'IWR climate'], zorder=5)
        plt.grid(zorder=1)
        plt.legend()

    plt.tight_layout()


def county_hist_1(data, iwr):
    plt.figure(figsize=(30, 10), dpi=200)
    plt.suptitle("Average Seasonal Consumptive Use (in)")
    # plt.suptitle("Average Seasonal ET (in)")

    print("total average CU: dnrc: {:.2f}, opnt: {:.2f}".format(data['dnrc_cu'].mean(), data['opnt_cu'].mean()))

    plot = 1
    for i in data['county'].unique():
        plt.subplot(4, 13, plot)
        # plt.hist([data[data['county'] == i]['etbc'], data[data['county'] == i]['etos']],
        #          label=['DNRC', 'Gridmet ETo'], zorder=5)
        plt.title("{} ({})".format(COUNTIES[i], i), size=8)
        bins = range(30)
        # plt.hist([data[data['county'] == i]['dnrc_cu'], data[data['county'] == i]['opnt_cu']], bins=bins,
        #          label=['DNRC', 'OpenET'], zorder=5)
        # plt.hist([data[data['county'] == i]['dnrc_cu'], data[data['county'] == i]['opnt_cu'],
        #          iwr[iwr['county'] == i]['dnrc_cu']], bins=bins,
        #          label=['DNRC', 'OpenET', 'IWR climate'], zorder=5)
        plt.hist([data[data['county'] == i]['dnrc_cu'], iwr[iwr['county'] == i]['dnrc_cu']], bins=bins,
                 label=['DNRC', 'IWR climate'], zorder=5)
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


def dif_hist(data):
    plt.figure(figsize=(10, 5))
    plt.title("Difference in Consumptive Use Estimate (inches, DNRC minus OpenET)")
    bins = range(-10, 25)
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
    bins = range(-10, 25)
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
    bins = range(-10, 25)
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
    bins = range(30)

    # Option 1
    plt.hist(data['dnrc_cu'], bins=bins, zorder=5, alpha=0.5, label='DNRC', color='tab:blue')
    # plt.hist(iwr['dnrc_cu'], bins=bins, zorder=5, alpha=0.5, label='DNRC (climate)', color='tab:blue')
    plt.hist(data['opnt_cu'], bins=bins, zorder=5, alpha=0.5, label='OpenET', color='tab:orange')

    plt.vlines(data['dnrc_cu'].mean(), 0, 7000, zorder=7, color='tab:blue',
               label='mean: {:.2f} in'.format(data['dnrc_cu'].mean()))
    # plt.vlines(iwr['dnrc_cu'].mean(), 0, 8000, zorder=7, color='tab:blue',
    #            label='mean: {:.2f} in'.format(iwr['dnrc_cu'].mean()))
    plt.vlines(data['opnt_cu'].mean(), 0, 7000, zorder=7, color='tab:orange',
               label='mean: {:.2f} in'.format(data['opnt_cu'].mean()))

    # Option 2
    # plt.hist([data['dnrc_cu'], data['opnt_cu'], iwr['dnrc_cu']], bins=bins, zorder=5, label=['DNRC', 'OpenET', 'IWR climate'])
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
    from matplotlib.patches import Rectangle, Ellipse

    def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='tab:blue',
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
        ax.add_collection(pc)

        # Plot errorbars
        artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror,
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


def time_series(iwr, con, var):
    # print(len(range(1987, 2018, 3)))
    # iterables = [["bar", "baz", "foo", "qux"], ["one", "two"]]

    the_index = pd.MultiIndex.from_product([COUNTIES.keys(), range(1987, 2018, 3)], names=["county", "st_year"])
    mov_avg = pd.DataFrame(index=the_index, columns=[''])
    for i in range(1987, 2018, 3):
        decade = sum_data(con, start=i, end=i+6)
        print(i, i+6)


def stats(data, selection=()):
    if len(selection) == 0:
        counties = data['county'].unique()
    else:
        counties = selection

    all_results = {'neg_cu': data['opnt_cu'].lt(0).sum()}  # / len(data)

    results = pd.DataFrame(columns=['neg_cu'], index=counties)

    for i in counties:
        # percent of fields with a negative CU
        results.loc[i, 'neg_cu'] = data[data['county'] == i]['opnt_cu'].lt(0).sum()  # / len(data[data['county'] == i])

    return all_results, results


if __name__ == '__main__':

    if os.path.exists('F:/FileShare'):
        main_dir = 'F:/FileShare/openet_pilot'
    else:
        main_dir = 'F:/openet_pilot'

    # fix county list
    # note: 101 should have data, but etof didn't load, under investigation
    for key in ['011', '025', '101', '109']:
        COUNTIES.pop(key, None)

    # # Establish sqlite connection
    # conec = sqlite3.connect(os.path.join(main_dir, "opnt_analysis_03042024_Copy.db"))  # full project
    # # conec = sqlite3.connect("C:/Users/CND571/Documents/Data/random_05082024.db")  # test
    #
    # # extract data
    # # gm_ct = gridmet_ct(conec)
    # # print(gm_ct)
    # sum_data(conec, irrmapper=1, save="C:/Users/CND571/Documents/Data")
    # # por_data = sum_data(conec)
    #
    # # close connection
    # conec.close()

    por_data_im = pd.read_csv("C:/Users/CND571/Documents/Data/cu_results_1987_2023_im1_mf1997-2006.csv",
                              dtype={'county': str}, index_col='fid')
    por_data = pd.read_csv("C:/Users/CND571/Documents/Data/cu_results_1987_2023_im0_mf1997-2006.csv",
                           dtype={'county': str}, index_col='fid')
    iwr_data = pd.read_csv("C:/Users/CND571/Documents/Data/iwr_cu_results_mf1997-2006.csv",
                           dtype={'county': str}, index_col='fid')
    print(por_data.columns)
    print(iwr_data.columns)

    # print(len(por_data_im['dnrc_cu'].unique()))
    # print(len(por_data['dnrc_cu'].unique()))
    # print(len(iwr_data['dnrc_cu'].unique()))

    # all_stats, county_stats = stats(por_data)

    # neg_cu = por_data['opnt_cu'].lt(0).sum()
    # neg_cu_fids = por_data[por_data['opnt_cu'].lt(0)].index
    # print(neg_cu, neg_cu_fids)

    # neg_cu_dif = (por_data['dnrc_cu']-por_data['opnt_cu']).lt(0).sum()
    # print(neg_cu_dif)

    # plot_results_2(por_data)  # this looks good
    # plot_results_3(por_data, iwr_data)
    # county_hist(por_data, iwr_data, selection=('001',))
    # county_hist_1(por_data, iwr_data)
    # dif_hist(por_data)  # this looks good
    # dif_hist_1(por_data, iwr_data)
    # dif_hist_2(por_data, iwr_data)
    # scatter(por_data)  # this is nice
    # all_hist_1(por_data, iwr_data)

    # plot_results_2(por_data_im)  # woah!
    # plot_results_3(por_data_im, iwr_data)
    # county_hist(por_data_im, iwr_data, selection=('001',))
    # county_hist_1(por_data_im, iwr_data)
    # dif_hist(por_data_im)  # this looks good
    # dif_hist_1(por_data_im, iwr_data)
    # dif_hist_2(por_data_im, iwr_data)
    # scatter(por_data_im)  # this is nice
    # all_hist_1(por_data_im, iwr_data)

    # todo: time series w/ moving averages?

    time_series(iwr_data)

    plt.show()

# ========================= EOF ====================================================================
