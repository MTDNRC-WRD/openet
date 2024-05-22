
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
from geopy import distance

from run_all import COUNTIES


def plot_results(con, met_too=False, irrmapper=0, mf_periods='1997-2006'):
    """Create figure comparing ET and consumptive use from two different methods."""
    gridmets = pd.read_sql("SELECT DISTINCT gfid FROM gridmet_ts", con)
    print("gridmet points: {}".format(len(gridmets)))

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
                    c='tab:pink', zorder=5,
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
                    c='tab:pink', label="{} ({})".format(COUNTIES[i], i))
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


if __name__ == '__main__':

    if os.path.exists('F:/FileShare'):
        main_dir = 'F:/FileShare/openet_pilot'
    else:
        main_dir = 'F:/openet_pilot'

    # Establish sqlite connection
    conec = sqlite3.connect(os.path.join(main_dir, "opnt_analysis_03042024_Copy.db"))  # full project
    # conec = sqlite3.connect("C:/Users/CND571/Documents/Data/random_05082024.db")  # test
    # sqlite database table names
    gm_ts, fields_db, results, etof_db = 'gridmet_ts', 'field_data', 'field_cu_results', 'opnt_etof'
    irr_db, iwr_cu_db = 'irrmapper', 'static_iwr_results'

    plot_results(conec)

    conec.commit()
    conec.close()

# ========================= EOF ====================================================================
