import os
from copy import deepcopy
from datetime import timedelta, datetime
from calendar import monthrange

import pandas as pd
import numpy as np
from pypxlib import Table
from matplotlib import pyplot as plt


# import xarray as xr ## not used? Commented out to avoid having to install module. Used elsewhere, so now installed.

lat_to_sunshine = {50: [5.99, 6.32, 8.24, 9.24, 10.68, 10.92, 10.99, 9.99, 8.46, 7.44, 6.08, 5.65],
                   49: [6.08, 6.36, 8.25, 9.20, 10.60, 10.82, 10.90, 9.94, 8.46, 7.48, 6.16, 5.75],
                   48: [6.17, 6.41, 8.26, 9.17, 10.52, 10.72, 10.81, 9.89, 8.45, 7.51, 6.24, 5.85],
                   47: [6.25, 6.45, 8.27, 9.14, 10.45, 10.63, 10.73, 9.84, 8.44, 7.54, 6.31, 5.95],
                   46: [6.33, 6.50, 8.28, 9.11, 10.38, 10.53, 10.65, 9.79, 8.43, 7.58, 6.37, 6.05],
                   45: [6.40, 6.54, 8.29, 9.08, 10.31, 10.46, 10.57, 9.75, 8.42, 7.61, 6.43, 6.14],
                   44: [6.48, 6.57, 8.29, 9.05, 10.25, 10.39, 10.49, 9.71, 8.41, 7.64, 6.50, 6.22],
                   43: [6.55, 6.61, 8.30, 9.02, 10.19, 10.31, 10.42, 9.66, 8.40, 7.67, 6.56, 6.31],
                   42: [6.61, 6.65, 8.30, 8.99, 10.13, 10.24, 10.35, 9.62, 8.40, 7.70, 6.62, 6.39],
                   41: [6.68, 6.68, 8.31, 8.96, 10.07, 10.16, 10.29, 9.59, 8.39, 7.72, 6.68, 6.47],
                   40: [6.75, 6.72, 8.32, 8.93, 10.01, 10.09, 10.22, 9.55, 8.39, 7.75, 6.73, 6.54]}

alfalfa_kc = [0.63, 0.73, 0.86, 0.99, 1.08, 1.13, 1.11, 1.06, 0.99, 0.91, 0.78, 0.64]

alfalfa_kc_1 = [0.6, 0.63, 0.68, 0.73, 0.79, 0.86, 0.92, 0.98, 1.04, 1.08, 1.12, 1.13,
                1.12, 1.11, 1.09, 1.06, 1.03, 0.99, 0.95, 0.91, 0.85, 0.78, 0.72, 0.64]


def get_iwr_db_temp(station):
    """ Returns the monthly mean temperatures from the IWR database as a pandas series given the last 4 digits of an IWR weather station
    as a string """
    table = Table('C:/Users/CND571/Documents/IWR/Database/climate.db')

    stations = {}
    for i in range(len(table)):
        stations[table[i]['Station No'][2:]] = i

    row = table[stations[station]]
    temps = pd.Series({1: row['T Jan'], 2: row['T Feb'], 3: row['T Mar'], 4: row['T Apr'], 5: row['T May'],
                       6: row['T Jun'], 7: row['T Jul'], 8: row['T Aug'], 9: row['T Sep'], 10: row['T Oct'],
                       11: row['T Nov'], 12: row['T Dec']})
    return temps


def effective_ppt_table():
    # NEH 2-148, table 2-43
    # _file = os.path.join(os.path.dirname(__file__), 'eff_precip_neh_chap2.csv')
    d = 'C:/Users/CND571/Documents/Data'
    _file = os.path.join(d, 'eff_precip_neh_chap2.csv')
    return pd.read_csv(_file, index_col=0)


def modified_blaney_criddle_2d(df, lat_degrees=None):
    """
    INCOMPLETE
    This is an implementation of the below SCS Blaney Criddle that will accept a 2D ndarray as input.

    :param df:
    :param lat_degrees:
    :return:
    """

    # INCOMPLETE
    # TODO: complete this function
    mid_months = [i for i, _ in enumerate(df.time.values) if pd.to_datetime(_).day == 15]
    t = df['MM'][mid_months].resample(time='1M').mean()
    t = t.groupby('time.month').mean() * 9 / 5 + 32

    p = df['PP'].resample(time='1M').sum()
    p = p.groupby('time.month').mean() / 25.4

    dtmm = df['MM'].groupby('time.dayofyear').mean() - 273.15
    dtmm = dtmm * 9 / 5 + 32
    season_start = dtmm.where(lambda x: x > 50.0).idxmin(dim='dayofyear')

    dtmn = df['MN'].groupby('time.dayofyear').mean() - 273.15
    dtmn = dtmn * 9 / 5 + 32
    cutoff = 183
    dtmn = dtmn[cutoff:, :, :]
    didx = np.ones_like(dtmn) * 377
    tidx = np.ones_like(dtmn) * 28.0
    doy_cube = np.broadcast_to(dtmn.dayofyear.data, dtmn.data.T.shape).T
    arr = dtmn.data
    season_end = np.where(arr < tidx, doy_cube, didx).argmin(axis=0) + cutoff + 1

    # TODO: pick up from here
    season_length = (season_end - season_start).days

    lat = round(lat_degrees)
    sunshine = lat_to_sunshine[lat]

    first_period = []
    d = season_start
    while d.day != 1:
        first_period.append(d)
        d += timedelta(days=1)

    midpoint = season_start + (d - season_start) / 2
    counter = (midpoint - season_start).days
    t_prev, t_next = t.loc[midpoint.month], t.loc[midpoint.month + 1]
    remaining_days = (midpoint - season_start).days
    month_len = monthrange(2000, midpoint.month)[1]
    month_fraction = remaining_days / month_len
    temp = t_prev + (month_fraction * (t_next - t_prev))

    day_prev, day_next = sunshine[midpoint.month - 1], sunshine[midpoint.month]
    daylight = (day_prev + (month_fraction * (day_next - day_prev))) * month_fraction

    dates, d_accum, pct_season = [midpoint], [counter], [counter / season_length]
    temps, pct_day_hrs = [temp], [daylight]
    for d, v in dtmm.loc[midpoint: season_end].iteritems():
        counter += 1
        if d.day == 15:
            dates.append(d)
            d_accum.append(counter)
            pct_season.append(counter / season_length)
            temps.append(t.loc[d.month])
            pct_day_hrs.append(sunshine[d.month - 1])

    second_period = []
    d = dates[-1]
    while d != season_end:
        second_period.append(d)
        d += timedelta(days=1)

    midpoint = second_period[0] + (second_period[-1] - second_period[0]) / 2
    t_prev, t_next = t.loc[midpoint.month], t.loc[midpoint.month + 1]
    remaining_days = (midpoint - dates[-1]).days
    month_len = monthrange(2000, midpoint.month)[1]
    month_fraction = remaining_days / month_len
    temp = t_prev + (month_fraction * (t_next - t_prev))
    accum_days_last = (midpoint - season_start).days

    day_prev, day_next = sunshine[midpoint.month - 1], sunshine[midpoint.month]
    daylight = (day_prev + (month_fraction * (day_next - day_prev))) * month_fraction

    dates.append(midpoint)
    d_accum.append(accum_days_last)
    pct_season.append(accum_days_last / season_length)
    temps.append(temp)
    pct_day_hrs.append(daylight)

    # temps = []

    dates = [pd.to_datetime('2000-{}-{}'.format(d.month, d.day)) for d in dates]
    df = pd.DataFrame(np.array([d_accum, pct_season, temps, pct_day_hrs]).T,
                      columns=['accum_day', 'pct_season', 't', 'p'],
                      index=dates)

    df['f'] = df['t'] * df['p'] / 100.

    df['kt'] = df['t'] * 0.0173 - 0.314

    kc = pd.Series(alfalfa_kc, index=[d for d in yr_ind if d.day == 15])
    kc = kc.reindex(yr_ind)
    kc.iloc[0] = 0.6
    kc.iloc[-1] = 0.6
    kc = kc.interpolate()
    df['kc'] = kc.loc[df.index]
    df['k'] = df['kc'] * df['kt']
    df['u'] = df['k'] * df['f']
    df['ref_u'] = df['kt'] * df['f']
    return df, season_start, season_end


def modified_blaney_criddle_neh_ex(df, lat_degrees=None, elev=None, season_start=None, season_end=None,
                            mid_month=False):
    """
        Changing implementation in order to replicate examples shown in NEH 2-233 through 2-234
        Custom implementation of the SCS Blaney Criddle method.
        :param df:
        :param lat_degrees:
        :param elev:
        :param season_start:
        :param season_end:
        :param mid_month:
        :return:
        """

    if mid_month:
        # NEH 2-233 "...mean temperature is assumed to occur on the 15th day of each month..."
        # however, this gives results that differ substantially from NRCS IWR database files
        mid_months = [d for d in df.index if d.day == 15]
        t = df['MM'].loc[mid_months].resample('M').mean()
    else:
        # this matches IWR database files almost exactly
        t = df['MM'].resample('M').mean()
    t = t.groupby(t.index.month).mean() * 9 / 5 + 32
    # t0 = pd.Series({1: 24.0, 2: 28.8, 3: 35.6, 4: 43.2, 5: 51.6, 6: 59.4, 7: 65.4, 8: 64.1,
    #                 9: 55.4, 10: 45.7, 11: 32.0, 12: 24.1})  # for Dillon station
    # print('t difference', t-t0)

    p = df['PP'].resample('M').sum()
    ppt_quantiles = pd.DataFrame(np.zeros((12, 1)) * np.nan, index=[x for x in range(1, 13)])
    eff_ppt_lookup = effective_ppt_table()

    for m in range(1, 13):
        mdata = np.array([p.loc[i] / 25.4 for i in p.index if i.month == m])
        # print('mdata',m,mdata)
        quantile = np.quantile(mdata, 0.2)
        ppt_quantiles.loc[m] = quantile

    dtmm = df['MM'].groupby([df.index.month, df.index.day]).mean() * 9 / 5 + 32
    yr_ind = pd.date_range('2000-01-01', '2000-12-31', freq='d')
    dtmm.index = yr_ind

    if not season_start:
        season_start = dtmm[dtmm > 50.].index[0]
    else:
        season_start = pd.to_datetime(season_start)

    if not season_end:
        dtmn = df['MN'].groupby([df.index.month, df.index.day]).mean() * 9 / 5 + 32
        yr_ind = pd.date_range('2000-01-01', '2000-12-31', freq='d')
        dtmn.index = yr_ind
        season_end = dtmn.loc['2000-07-01':][dtmn < 28.].index[0]
    else:
        season_end = pd.to_datetime(season_end)

    season_length = (season_end - season_start).days

    lat = round(lat_degrees)
    sunshine = lat_to_sunshine[lat]

    first_period = []
    d = season_start
    while d.day != 1:
        first_period.append(d)
        d += timedelta(days=1)

    midpoint = season_start + (d - season_start) / 2
    counter = (midpoint - season_start).days
    t_prev, t_next = t.loc[midpoint.month], t.loc[midpoint.month + 1]
    month_len = monthrange(2000, midpoint.month)[1]
    remaining_days = month_len - season_start.day
    month_fraction = remaining_days / month_len
    temp = t_prev + (month_fraction * (t_next - t_prev))

    day_prev, day_next = sunshine[midpoint.month - 1], sunshine[midpoint.month]
    daylight = (day_prev + (month_fraction * (day_next - day_prev))) * month_fraction

    dates, d_accum, pct_season = [midpoint], [counter], [counter / season_length]
    temps, pct_day_hrs = [temp], [daylight]

    # This is where we go wrong? Check that end of season is not in between the 15th and the 30th
    # Problem does not occur if season start is before the 15th of the month, I don't think.

    # Loop searches for another half-month in the same month as the season end, and appends appropriate data.
    # Except we don't want that.
    for d, v in dtmm.loc[midpoint: season_end].items():
        counter += 1
        if d.day == 15:
            dates.append(d)
            d_accum.append(counter)
            pct_season.append(counter / season_length)
            # why are we using mean monthly vs. 15th day mean?
            # temps.append(t.loc[d.month]) ## use v here?
            temps.append(v)
            # print(t.loc[d.month], v)
            pct_day_hrs.append(sunshine[d.month - 1])

    second_period = []
    d = dates[-1]
    while d != season_end:
        second_period.append(d)
        d += timedelta(days=1)

    if season_end.day > 15:  # remove last entry above, make sure second period is correct.
        dates = dates[:-1]
        d_accum = d_accum[:-1]
        pct_season = pct_season[:-1]
        pct_day_hrs = pct_day_hrs[:-1]
        temps = temps[:-1]

        # start of second period should be first of the month
        second_period = []
        d = dates[-1]
        while d.day != 1:
            d += timedelta(days=1)
        while d != season_end:
            second_period.append(d)
            d += timedelta(days=1)

    midpoint = second_period[0] + (second_period[-1] - second_period[0]) / 2
    t_prev, t_next = t.loc[midpoint.month - 1], t.loc[midpoint.month]  # adjusted to call last month and this one.
    remaining_days = len(second_period) + 1  # (midpoint - dates[-1]).days
    month_len = monthrange(2000, midpoint.month)[1]
    month_fraction = remaining_days / month_len
    temp = t_prev + (month_fraction * (t_next - t_prev))
    accum_days_last = (midpoint - season_start).days

    day_prev, day_next = sunshine[midpoint.month - 2], sunshine[midpoint.month - 1]  # adjusted as t above.
    daylight = (day_prev + (month_fraction * (day_next - day_prev))) * month_fraction

    dates.append(midpoint)
    d_accum.append(accum_days_last)
    pct_season.append(accum_days_last / season_length)
    temps.append(temp)
    pct_day_hrs.append(daylight)

    print(len(temps),temps)
    # overwriting temps for illustration purposes
    temps = [51.1, 56.3, 66.4, 72.8, 71.3, 62.7, 53.5]
    print(len(temps),temps)

    # error in pct_day_hrs, otherwise good.
    # pct_day_hrs = [1.87, 9.99, 10.07, 10.20, 9.54, 8.39, 6.31]

    dates = [pd.to_datetime('2000-{}-{}'.format(d.month, d.day)) for d in dates]
    df = pd.DataFrame(np.array([d_accum, pct_season, temps, pct_day_hrs]).T,
                      columns=['accum_day', 'pct_season', 't', 'p'],
                      index=dates)
    # t = mean monthly air temp.
    # p = monthly percentage of annual daylight hours

    df['f'] = df['t'] * df['p'] / 100.  # monthly consumptive use factor

    df['kt'] = df['t'] * 0.0173 - 0.314
    df['kt'][df['t'] < 36.] = 0.3

    elevation_corr = 1 + (0.1 * np.floor(elev / 1000.))  # from footnote 3 on IWR results page

    kc = pd.Series(alfalfa_kc, index=[d for d in yr_ind if d.day == 15])
    kc = kc.reindex(yr_ind)
    kc.iloc[0] = 0.6
    kc.iloc[-1] = 0.6
    kc = kc.interpolate()
    df['kc'] = kc.loc[df.index]  # growth stage from table
    df['k'] = df['kc'] * elevation_corr * df['kt']  # empirical crop coefficient, corrected for air temp.
    df['u'] = df['k'] * df['f']  # monthly consumptive use, inches
    df['ref_u'] = df['kt'] * df['f']  # no crop coefficient or elevation correction.

    # rounded = [np.round(ppt * 2) / 2 for ppt in ppt_quantiles.values]
    # ppt_quantiles['eff_ppt'] = [eff_ppt_lookup.loc[r,]]

    # print('no elevation correction:', (df['kc'] * df['kt'] * df['f']).sum() )

    # print(df['t']) ## different from reference in database

    print(df)

    return df, season_start, season_end, kc


# Remove this one too.
def modified_blaney_criddle(df, lat_degrees=None, elev=None, season_start=None, season_end=None,
                            mid_month=True):
    """
    Custom implementation of the SCS Blaney Criddle method.
    Assumes inout data is in Celsius.
    :param df:
    :param lat_degrees:
    :param elev:
    :param season_start:
    :param season_end:
    :param mid_month:
    :return:
    """

    # TODO: implement effective precipitation calculations
    if mid_month:
        # NEH 2-233 "...mean temperature is assumed to occur on the 15th day of each month..."
        # however, this gives results that differ substantially from NRCS IWR database files
        mid_months = [d for d in df.index if d.day == 15]
        t = df['MM'].loc[mid_months].resample('M').mean()
    else:
        # this matches IWR database files almost exactly
        t = df['MM'].resample('M').mean()
    t = t.groupby(t.index.month).mean() * 9 / 5 + 32
    # print(t)

    # precipitation data, not used in current calculations
    p = df['PP'].resample('M').sum()
    ppt_quantiles = pd.DataFrame(np.zeros((12, 1)) * np.nan, index=[x for x in range(1, 13)])
    eff_ppt_lookup = effective_ppt_table()

    for m in range(1, 13):
        mdata = np.array([p.loc[i] / 25.4 for i in p.index if i.month == m])
        # print('mdata',m,mdata)
        quantile = np.quantile(mdata, 0.2)
        ppt_quantiles.loc[m] = quantile

    dtmm = df['MM'].groupby([df.index.month, df.index.day]).mean() * 9 / 5 + 32
    yr_ind = pd.date_range('2000-01-01', '2000-12-31', freq='d')
    dtmm.index = yr_ind

    if not season_start:
        season_start = dtmm[dtmm > 50.].index[0]

        # # calculating season start, method for only monthly data. Above method is close.
        # month = t[t >= 50].index[0]  # get month when temp gets above 50
        # month_len = monthrange(2000, month - 1)[1]  # get length of preceding month
        # season_start1 = datetime(year=2000, month=month - 1, day=15)  # get midpoint of preceding month
        # # calculate number of days past midpoint of preceding month when we reach average temp of 50
        # days = round(((50. - t[month - 1]) * month_len) / (t[month] - t[month - 1]))
        # season_start1 = season_start1 + timedelta(days=days)  # add days
        # print('season_start1: ', season_start1)
    else:
        season_start = pd.to_datetime(season_start)

    if not season_end:
        # # the daily method yields a very different result from IWR
        # dtmn = df['MN'].groupby([df.index.month, df.index.day]).mean() * 9 / 5 + 32
        # yr_ind = pd.date_range('2000-01-01', '2000-12-31', freq='d')
        # dtmn.index = yr_ind
        # season_end = dtmn.loc['2000-07-01':][dtmn < 28.].index[0]

        # try this: (gets pretty close to IWR freeze dates)
        month = t[t >= 53].index[-1]  # get last month when temp gets above 53
        month_len = monthrange(2000, month)[1]  # get length of month
        season_end = datetime(year=2000, month=month, day=15)  # get midpoint of month
        # calculate number of days past midpoint of month when we reach average temp of 53
        days = round(((53. - t[month]) * month_len) / (t[month + 1] - t[month]))
        season_end = season_end + timedelta(days=days)  # add days
        # print(season_end)
    else:
        season_end = pd.to_datetime(season_end)

    season_length = (season_end - season_start).days

    lat = round(lat_degrees)
    sunshine = lat_to_sunshine[lat]

    first_period = []
    d = season_start
    while d.day != 1:
        first_period.append(d)
        d += timedelta(days=1)

    if d != season_start:
        midpoint = season_start + (d - season_start) / 2
        if midpoint.hour != 0:
            midpoint = midpoint - timedelta(hours=12)  # to avoid splitting a day in half
        counter = (midpoint - season_start).days
        t_prev, t_next = t.loc[midpoint.month], t.loc[midpoint.month + 1]
        month_len = monthrange(2000, midpoint.month)[1]
        interp_fraction = (midpoint.day - 15) / month_len
        # print('interp_fraction 1: ', interp_fraction)
        remaining_days = month_len - season_start.day
        month_fraction = remaining_days / month_len
        temp = t_prev + (interp_fraction * (t_next - t_prev))

        day_prev, day_next = sunshine[midpoint.month - 1], sunshine[midpoint.month]
        daylight = (day_prev + (interp_fraction * (day_next - day_prev))) * month_fraction
    else:
        midpoint = season_start + timedelta(days=15)
        counter = (midpoint - season_start).days
        temp = t.loc[midpoint.month]
        daylight = sunshine[midpoint.month - 1]

    dates, d_accum, pct_season = [midpoint], [counter], [counter / season_length]
    temps, pct_day_hrs = [temp], [daylight]
    for d in pd.date_range(midpoint, season_end, freq='d'):
        counter += 1
        if d.day == 15:
            dates.append(d)
            d_accum.append(counter)
            pct_season.append(counter / season_length)
            temps.append(t.loc[d.month])
            pct_day_hrs.append(sunshine[d.month - 1])

    # second_period = []
    # d = dates[-1]
    # while d != season_end:
    #     second_period.append(d)
    #     d += timedelta(days=1)

    if season_end.day > 15:  # remove last entry
        dates = dates[:-1]
        d_accum = d_accum[:-1]
        pct_season = pct_season[:-1]
        pct_day_hrs = pct_day_hrs[:-1]
        temps = temps[:-1]

    # start of second period should always be first of the month
    second_period = []
    d = dates[-1]
    while d.day != 1:
        d += timedelta(days=1)
    while d != season_end:
        second_period.append(d)
        d += timedelta(days=1)

    if len(second_period) > 0:
        midpoint = second_period[0] + (second_period[-1] - second_period[0]) / 2
        if midpoint.hour != 0:
            midpoint = midpoint - timedelta(hours=12)  # to avoid splitting a day in half
    else:
        midpoint = season_end

    t_prev, t_next = t.loc[midpoint.month - 1], t.loc[midpoint.month]
    remaining_days = len(second_period) + 1
    month_len = monthrange(2000, midpoint.month)[1]
    month_fraction = remaining_days / month_len
    # prev_month_len = monthrange(2000, midpoint.month - 1)[1] ## do we need this?
    interp_fraction = (midpoint.day + 15) / month_len
    # print('interp_fraction 2: ', interp_fraction)
    temp = t_prev + (interp_fraction * (t_next - t_prev))
    accum_days_last = (midpoint - season_start).days

    day_prev, day_next = sunshine[midpoint.month - 2], sunshine[midpoint.month - 1]
    daylight = (day_prev + (interp_fraction * (day_next - day_prev))) * month_fraction

    dates.append(midpoint)
    d_accum.append(accum_days_last)
    pct_season.append(accum_days_last / season_length)
    temps.append(temp)
    pct_day_hrs.append(daylight)

    dates = [pd.to_datetime('2000-{}-{}'.format(d.month, d.day)) for d in dates]
    df = pd.DataFrame(np.array([d_accum, pct_season, temps, pct_day_hrs]).T,
                      columns=['accum_day', 'pct_season', 't', 'p'],
                      index=dates)
    # t = mean monthly air temp.
    # p = monthly percentage of annual daylight hours

    df['f'] = df['t'] * df['p'] / 100.  # monthly consumptive use factor

    df['kt'] = df['t'] * 0.0173 - 0.314
    df['kt'][df['t'] < 36.] = 0.3

    elevation_corr = 1 + (0.1 * np.floor(elev / 1000.))  # from footnote 3 on IWR results page

    kc = pd.Series(alfalfa_kc, index=[d for d in yr_ind if d.day == 15])
    kc = kc.reindex(yr_ind)
    kc.iloc[0] = 0.6
    kc.iloc[-1] = 0.6
    kc = kc.interpolate()
    df['kc'] = kc.loc[df.index]  # growth stage from table
    df['k'] = df['kc'] * elevation_corr * df['kt']  # empirical crop coefficient, corrected for air temp.
    df['u'] = df['k'] * df['f']  # monthly consumptive use, inches
    df['ref_u'] = df['kt'] * df['f']  # no crop coefficient or elevation correction.

    # rounded = [np.round(ppt * 2) / 2 for ppt in ppt_quantiles.values]
    # ppt_quantiles['eff_ppt'] = [eff_ppt_lookup.loc[r,]]

    # print(df)

    return df, season_start, season_end, kc


# Keep this one, make sure values match with _1_ep
def modified_blaney_criddle_ep(df, lat_degrees=None, elev=None, season_start=None, season_end=None,
                            mid_month=True):
    """
    Custom implementation of the SCS Blaney Criddle method.
    Assumes inout data is in Celsius and rain in mm.
    Includes calculation of effective precip per NEH Ch2, pgs 147-152 (pdf 165-170)
    :param df:
    :param lat_degrees:
    :param elev:
    :param season_start:
    :param season_end:
    :param mid_month:
    :return:
    """

    if mid_month:
        # NEH 2-233 "...mean temperature is assumed to occur on the 15th day of each month..."
        # however, this gives results that differ substantially from NRCS IWR database files
        mid_months = [d for d in df.index if d.day == 15]
        t = df['MM'].loc[mid_months].resample('M').mean()
    else:
        # this matches IWR database files almost exactly
        t = df['MM'].resample('M').mean()
    t = t.groupby(t.index.month).mean() * 9 / 5 + 32
    # print(t)

    # precipitation data
    p = df['PP'].resample('M').sum()
    p = p.groupby(p.index.month).mean() / 25.4  # mm to in
    annual_p = df['PP'].resample('Y').sum() / 25.4
    # ppt_quantiles = pd.DataFrame(np.zeros((12, 1)) * np.nan, index=[x for x in range(1, 13)])
    # eff_ppt_lookup = effective_ppt_table()

    # for m in range(1, 13):
    #     mdata = np.array([p.loc[i] / 25.4 for i in p.index if i.month == m])
    #     # print('mdata',m,mdata)
    #     quantile = np.quantile(mdata, 0.2)
    #     ppt_quantiles.loc[m] = quantile

    dtmm = df['MM'].groupby([df.index.month, df.index.day]).mean() * 9 / 5 + 32
    yr_ind = pd.date_range('2000-01-01', '2000-12-31', freq='d')
    dtmm.index = yr_ind

    if not season_start:
        season_start = dtmm[dtmm > 50.].index[0]

        # # calculating season start, method for only monthly data. Above method is close.
        # month = t[t >= 50].index[0]  # get month when temp gets above 50
        # month_len = monthrange(2000, month - 1)[1]  # get length of preceding month
        # season_start1 = datetime(year=2000, month=month - 1, day=15)  # get midpoint of preceding month
        # # calculate number of days past midpoint of preceding month when we reach average temp of 50
        # days = round(((50. - t[month - 1]) * month_len) / (t[month] - t[month - 1]))
        # season_start1 = season_start1 + timedelta(days=days)  # add days
        # print('season_start1: ', season_start1)
    else:
        season_start = pd.to_datetime(season_start)

    if not season_end:
        # # the daily method yields a very different result from IWR
        # dtmn = df['MN'].groupby([df.index.month, df.index.day]).mean() * 9 / 5 + 32
        # yr_ind = pd.date_range('2000-01-01', '2000-12-31', freq='d')
        # dtmn.index = yr_ind
        # season_end = dtmn.loc['2000-07-01':][dtmn < 28.].index[0]

        # try this: (gets pretty close to IWR freeze dates)
        month = t[t >= 53].index[-1]  # get last month when temp gets above 53
        month_len = monthrange(2000, month)[1]  # get length of month
        season_end = datetime(year=2000, month=month, day=15)  # get midpoint of month
        # calculate number of days past midpoint of month when we reach average temp of 53
        days = round(((53. - t[month]) * month_len) / (t[month + 1] - t[month]))
        season_end = season_end + timedelta(days=days)  # add days
        # print(season_end)
    else:
        season_end = pd.to_datetime(season_end)

    season_length = (season_end - season_start).days

    lat = round(lat_degrees)
    sunshine = lat_to_sunshine[lat]

    first_period = []
    d = season_start
    while d.day != 1:
        first_period.append(d)
        d += timedelta(days=1)

    if d != season_start:
        midpoint = season_start + (d - season_start) / 2
        if midpoint.hour != 0:
            midpoint = midpoint - timedelta(hours=12)  # to avoid splitting a day in half
        counter = (midpoint - season_start).days
        t_prev, t_next = t.loc[midpoint.month], t.loc[midpoint.month + 1]
        month_len = monthrange(2000, midpoint.month)[1]
        interp_fraction = (midpoint.day - 15) / month_len
        # print('interp_fraction 1: ', interp_fraction)
        remaining_days = month_len - season_start.day
        month_fraction1 = remaining_days / month_len
        temp = t_prev + (interp_fraction * (t_next - t_prev))

        day_prev, day_next = sunshine[midpoint.month - 1], sunshine[midpoint.month]
        daylight = (day_prev + (interp_fraction * (day_next - day_prev))) * month_fraction1

        ppt_prev, ppt_next = p.loc[midpoint.month], p.loc[midpoint.month + 1]
        precip = (ppt_prev + (interp_fraction * (ppt_next - ppt_prev))) * month_fraction1
    else:
        month_fraction1 = 1
        midpoint = season_start + timedelta(days=15)
        counter = (midpoint - season_start).days
        temp = t.loc[midpoint.month]
        daylight = sunshine[midpoint.month - 1]
        precip = p.loc[midpoint.month]

    dates, d_accum, pct_season = [midpoint], [counter], [counter / season_length]
    temps, pct_day_hrs, precips = [temp], [daylight], [precip]
    for d in pd.date_range(midpoint, season_end, freq='d'):
        counter += 1
        if d.day == 15:
            dates.append(d)
            d_accum.append(counter)
            pct_season.append(counter / season_length)
            temps.append(t.loc[d.month])
            pct_day_hrs.append(sunshine[d.month - 1])
            precips.append(p.loc[d.month])

    if season_end.day > 15:  # remove last entry
        dates = dates[:-1]
        d_accum = d_accum[:-1]
        pct_season = pct_season[:-1]
        pct_day_hrs = pct_day_hrs[:-1]
        temps = temps[:-1]
        precips = precips[:-1]

    # start of second period should always be first of the month
    second_period = []
    d = dates[-1]
    while d.day != 1:
        d += timedelta(days=1)
    while d != season_end:
        second_period.append(d)
        d += timedelta(days=1)

    if len(second_period) > 0:
        midpoint = second_period[0] + (second_period[-1] - second_period[0]) / 2
        if midpoint.hour != 0:
            midpoint = midpoint - timedelta(hours=12)  # to avoid splitting a day in half
    else:
        midpoint = season_end

    t_prev, t_next = t.loc[midpoint.month - 1], t.loc[midpoint.month]
    remaining_days = len(second_period) + 1
    month_len = monthrange(2000, midpoint.month)[1]
    month_fraction2 = remaining_days / month_len
    # prev_month_len = monthrange(2000, midpoint.month - 1)[1] ## do we need this?
    interp_fraction = (midpoint.day + 15) / month_len
    # print('interp_fraction 2: ', interp_fraction)
    temp = t_prev + (interp_fraction * (t_next - t_prev))
    accum_days_last = (midpoint - season_start).days

    day_prev, day_next = sunshine[midpoint.month - 2], sunshine[midpoint.month - 1]
    daylight = (day_prev + (interp_fraction * (day_next - day_prev))) * month_fraction2

    ppt_prev, ppt_next = p.loc[midpoint.month - 1], p.loc[midpoint.month]
    precip = (ppt_prev + (interp_fraction * (ppt_next - ppt_prev))) * month_fraction2

    dates.append(midpoint)
    d_accum.append(accum_days_last)
    pct_season.append(accum_days_last / season_length)
    temps.append(temp)
    pct_day_hrs.append(daylight)
    precips.append(precip)

    dates = [pd.to_datetime('2000-{}-{}'.format(d.month, d.day)) for d in dates]
    df = pd.DataFrame(np.array([d_accum, pct_season, temps, precips, pct_day_hrs]).T,
                      columns=['accum_day', 'pct_season', 't', 'rain', 'p'],
                      index=dates)
    # t = mean monthly air temp.
    # p = monthly percentage of annual daylight hours

    df['f'] = df['t'] * df['p'] / 100.  # monthly consumptive use factor

    df['kt'] = df['t'] * 0.0173 - 0.314
    df['kt'][df['t'] < 36.] = 0.3

    elevation_corr = 1 + (0.1 * np.floor(elev / 1000.))  # from footnote 3 on IWR results page

    kc = pd.Series(alfalfa_kc, index=[d for d in yr_ind if d.day == 15])
    kc = kc.reindex(yr_ind)
    kc.iloc[0] = 0.6
    kc.iloc[-1] = 0.6
    kc = kc.interpolate()
    df['kc'] = kc.loc[df.index]  # growth stage from table
    df['k'] = df['kc'] * elevation_corr * df['kt']  # empirical crop coefficient, corrected for air temp.
    df['u'] = df['k'] * df['f']  # monthly consumptive use, inches
    df['ref_u'] = df['kt'] * df['f']  # no crop coefficient or elevation correction.

    # rounded = [np.round(ppt * 2) / 2 for ppt in ppt_quantiles.values]
    # ppt_quantiles['eff_ppt'] = [eff_ppt_lookup.loc[rounded]]  # needs updated keys here.

    # print(df)

    # Effective Precip Spaghetti!
    # From NEH Ch2 Table 2-43
    table_ep = effective_ppt_table()
    # Keys are water storage depth in inches. Is that information available?
    # Assume that this does not change with time.
    wsd = None
    if wsd:
        factors = {0.75: 0.72, 1.0: 0.77, 1.5: 0.86, 2.0: 0.93, 2.5: 0.97, 3.0: 1.00,
                   4.0: 1.02, 5.0: 1.04, 6.0: 1.06, 7.0: 1.07}
        # wsd = 3.3
        key = min(factors.keys(), key=lambda x: abs(x - wsd))
        factor = factors[key]
    else:  # assume default wsd of 3 in.
        factor = 1.0

    # Equations for sf and p_e from NEH CH2 P147 (165 in pdf)
    wsd = 3
    sf = 0.531747 + 0.295164 * wsd - 0.057697 * (wsd ** 2) + 0.003804 * (wsd ** 3)

    # Get monthly ET, round to nearest 1
    et = df['u'].to_list()
    # correct the first and last to be full month estimates.
    et[0] = et[0] / month_fraction1
    et[-1] = et[-1] / month_fraction2

    # create monthly precip above
    # Get monthly mean precip, round to nearest 0.5
    # NEH says to use full monthly precip and et, then do the table,
    # then multiply by the month fraction again to get effective precip.
    # Is that before or after I interpolate the values? - Let's say after.

    pm = df['rain']
    pm.iloc[0] = pm.iloc[0] / month_fraction1
    pm.iloc[-1] = pm.iloc[-1] / month_fraction2

    # We now have mean monthly precip. Need to get ratio for 80% chance.
    # Let's see if I can replicate that.
    # total growing season precip for each year on record. that would need to come from .txt files...
    # This would be easier to implement with the daily data. Leave it alone for the IWR stuff.
    avg_ann_ppt = annual_p.mean()
    avg_precip = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90]
    ratios_80 = [0.45, 0.5, 0.54, 0.57, 0.60, 0.62, 0.63, 0.65, 0.69, 0.71, 0.73, 0.74, 0.75, 0.77,
                 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90]
    ratios_table_80 = dict(zip(avg_precip, ratios_80))  # column from NEH table 2-46 (pdf 171)
    key = min(ratios_table_80.keys(), key=lambda x: abs(x - avg_ann_ppt))
    ratio80 = ratios_table_80[key]
    pm = pm * ratio80
    # print("avg ann precip: ", avg_ann_ppt, "ratio: ", ratio80)

    # Then round the results to input into table.
    pmr = (round(pm * 2) / 2)

    ep_tb = []
    ep_eq = []
    for i in range(len(df)):
        ep_tb.append(factor * table_ep[str(int(et[i]))][pmr.iloc[i]])  # 5 and 3.5 should yield 2.48... And it does! :)
        p_e = sf * (0.70917 * pow(pmr.iloc[i], 0.82416)) * pow(10, (0.02426 * int(et[i])))
        ep_eq.append(p_e)

    # Get first and last months back into fractions.
    ep_tb[0] = ep_tb[0] * month_fraction1
    ep_tb[-1] = ep_tb[-1] * month_fraction2
    ep_eq[0] = ep_eq[0] * month_fraction1
    ep_eq[-1] = ep_eq[-1] * month_fraction2

    df['ep'] = ep_tb
    # df['ep_eq'] = ep_eq

    # How to incorporate net irrigation/carryover? It does clearly affect this.

    return df, season_start, season_end, kc


# This is obsolete now, I think.
def modified_blaney_criddle_1(clim_db_loc, station, lat_degrees=None, elev=None, fullmonth=False):
    """
    Custom implementation of the SCS Blaney Criddle method.
    Updated to not rely on information outside IWR climate db
    IWR climate db is already in Fahrenheit.
    :param station: last 4 digits of station number
    :param lat_degrees:
    :param elev:
    :param clim_db_loc: path to IWR climate database
    :param fullmonth: set growing season to a predefined period of full months
    :return:
    """
    # 2000 is used arbitrarily to get a year's worth of daily time stamps. Right?
    yr_ind = pd.date_range('2000-01-01', '2000-12-31', freq='d')

    table = Table(clim_db_loc)
    # finding correct row in IWR db
    i = 0
    while table[i]['Station No'][2:] != station:
        i += 1
    row = table[i]
    print(row['Station Name'])
    # loading monthly mean temps from IWR db as dict
    t = pd.Series({1: row['T Jan'], 2: row['T Feb'], 3: row['T Mar'], 4: row['T Apr'], 5: row['T May'],
                  6: row['T Jun'], 7: row['T Jul'], 8: row['T Aug'], 9: row['T Sep'], 10: row['T Oct'],
                  11: row['T Nov'], 12: row['T Dec']})
    ppt = pd.Series({1: row['P Jan'], 2: row['P Feb'], 3: row['P Mar'], 4: row['P Apr'], 5: row['P May'],
                   6: row['P Jun'], 7: row['P Jul'], 8: row['P Aug'], 9: row['P Sep'], 10: row['P Oct'],
                   11: row['P Nov'], 12: row['P Dec']})
    # print(t)
    print(ppt)
    season_end = pd.to_datetime(row['Fall mo/dy 28'] + '/2000')  # average freeze date from database

    # # calculating season end?
    # month = t[t >= 53].index[-1]  # get last month when temp gets above 50
    # month_len = monthrange(2000, month)[1]  # get length of month
    # season_end1 = datetime(year=2000, month=month, day=15)  # get midpoint of month
    # # calculate number of days past midpoint of month when we reach average temp of 50
    # days = round(((53. - t[month]) * month_len) / (t[month + 1] - t[month]))
    # season_end1 = season_end1 + timedelta(days=days)  # add days
    # print(season_end1)

    # calculating season start
    month = t[t >= 50].index[0]  # get month when temp gets above 50
    month_len = monthrange(2000, month - 1)[1]  # get length of preceding month
    season_start = datetime(year=2000, month=month - 1, day=15)  # get midpoint of preceding month
    # calculate number of days past midpoint of preceding month when we reach average temp of 50
    days = round(((50. - t[month - 1]) * month_len) / (t[month] - t[month - 1]))
    season_start = season_start + timedelta(days=days)  # add days

    if fullmonth:
        # something a little weird with start dates/first period... non-inclusive of last day of month?
        season_start = pd.to_datetime('2000-04-30') # ex: 4-30 vs 5-01 doesn't change result.
        season_end = pd.to_datetime('2000-10-01')

    season_length = (season_end - season_start).days

    lat = round(lat_degrees)
    sunshine = lat_to_sunshine[lat]

    first_period = []
    d = season_start
    while d.day != 1:
        first_period.append(d)
        d += timedelta(days=1)

    if d != season_start:
        midpoint = season_start + (d - season_start) / 2
        if midpoint.hour != 0:
            midpoint = midpoint - timedelta(hours=12) # to avoid splitting a day in half
        counter = (midpoint - season_start).days
        t_prev, t_next = t.loc[midpoint.month], t.loc[midpoint.month + 1]
        month_len = monthrange(2000, midpoint.month)[1]
        interp_fraction = (midpoint.day - 15) / month_len
        # print('interp_fraction 1: ', interp_fraction)
        remaining_days = month_len - season_start.day
        month_fraction = remaining_days / month_len
        temp = t_prev + (interp_fraction * (t_next - t_prev))

        day_prev, day_next = sunshine[midpoint.month - 1], sunshine[midpoint.month]
        daylight = (day_prev + (interp_fraction * (day_next - day_prev))) * month_fraction

        ppt_prev, ppt_next = ppt.loc[midpoint.month], ppt.loc[midpoint.month + 1]
        precip = (ppt_prev + (interp_fraction * (ppt_next - ppt_prev))) * month_fraction
    else:
        midpoint = season_start + timedelta(days=15)
        counter = (midpoint - season_start).days
        temp = t.loc[midpoint.month]
        daylight = sunshine[midpoint.month - 1]
        precip = ppt.loc[midpoint.month]

    dates, d_accum, pct_season = [midpoint], [counter], [counter / season_length]
    temps, pct_day_hrs, precips = [temp], [daylight], [precip]
    for d in pd.date_range(midpoint, season_end, freq='d'):
        counter += 1
        if d.day == 15:
            dates.append(d)
            d_accum.append(counter)
            pct_season.append(counter / season_length)
            temps.append(t.loc[d.month])
            pct_day_hrs.append(sunshine[d.month - 1])
            precips.append(ppt.loc[d.month])

    # second_period = []
    # d = dates[-1]
    # while d != season_end:
    #     second_period.append(d)
    #     d += timedelta(days=1)

    if season_end.day > 15:  # remove last entry
        dates = dates[:-1]
        d_accum = d_accum[:-1]
        pct_season = pct_season[:-1]
        pct_day_hrs = pct_day_hrs[:-1]
        temps = temps[:-1]
        precips = precips[:-1]

    # start of second period should always be first of the month
    second_period = []
    d = dates[-1]
    while d.day != 1:
        d += timedelta(days=1)
    while d != season_end:
        second_period.append(d)
        d += timedelta(days=1)

    if len(second_period) > 0:
        midpoint = second_period[0] + (second_period[-1] - second_period[0]) / 2
        if midpoint.hour != 0:
            midpoint = midpoint - timedelta(hours=12)  # to avoid splitting a day in half
    else:
        midpoint = season_end

    t_prev, t_next = t.loc[midpoint.month - 1], t.loc[midpoint.month]
    remaining_days = len(second_period) + 1
    month_len = monthrange(2000, midpoint.month)[1]
    month_fraction = remaining_days / month_len
    # prev_month_len = monthrange(2000, midpoint.month - 1)[1] ## do we need this?
    interp_fraction = (midpoint.day + 15) / month_len
    # print('interp_fraction 2: ', interp_fraction)
    temp = t_prev + (interp_fraction * (t_next - t_prev))
    accum_days_last = (midpoint - season_start).days

    day_prev, day_next = sunshine[midpoint.month - 2], sunshine[midpoint.month - 1]
    daylight = (day_prev + (interp_fraction * (day_next - day_prev))) * month_fraction

    ppt_prev, ppt_next = ppt.loc[midpoint.month - 1], ppt.loc[midpoint.month]
    precip = (ppt_prev + (interp_fraction * (ppt_next - ppt_prev))) * month_fraction

    dates.append(midpoint)
    d_accum.append(accum_days_last)
    pct_season.append(accum_days_last / season_length)
    temps.append(temp)
    pct_day_hrs.append(daylight)
    precips.append(precip)

    dates = [pd.to_datetime('2000-{}-{}'.format(d.month, d.day)) for d in dates]
    df = pd.DataFrame(np.array([d_accum, pct_season, temps, precips, pct_day_hrs]).T,
                      columns=['accum_day', 'pct_season', 't', 'rain', 'p'],
                      index=dates)
    # t = mean air temp by period
    # p = percentage of annual daylight hours by period

    df['f'] = df['t'] * df['p'] / 100.  # monthly consumptive use factor

    df['kt'] = df['t'] * 0.0173 - 0.314
    df['kt'][df['t'] < 36.] = 0.3

    elevation_corr = 1 + (0.1 * np.floor(elev / 1000.))  # from footnote 3 on IWR results page
    # elevation_corr = 1 + (0.1 * round(elev / 1000.))  # I think floor is better
    # The rounding made Dillon much worse, did not affect Busby, and improved Chinook slightly (over vs undershoot).

    # kc = pd.Series(alfalfa_kc, index=[d for d in yr_ind if d.day == 15])
    kc = pd.Series(alfalfa_kc_1, index=[d for d in yr_ind if (d.day == 1) or (d.day == 15)])
    kc = kc.reindex(yr_ind)
    kc.iloc[0] = 0.6
    kc.iloc[-1] = 0.6
    kc = kc.interpolate()
    df['kc'] = kc.loc[df.index]  # growth stage from table
    df['k'] = df['kc'] * elevation_corr * df['kt']  # empirical crop coefficient, corrected for air temp.
    df['u'] = df['k'] * df['f']  # monthly consumptive use, inches
    df['ref_u'] = df['kt'] * df['f']  # no crop coefficient or elevation correction.

    # print(df)

    # Effective Precip Spaghetti!
    # From NEH Ch2 Table 2-43
    table_ep = effective_ppt_table()
    # Keys are water storage depth in inches. Where is that information?
    # Assume that this does not change with time.
    wsd = None
    if wsd:
        factors = {0.75: 0.72, 1.0: 0.77, 1.5: 0.86, 2.0: 0.93, 2.5: 0.97, 3.0: 1.00,
                   4.0: 1.02, 5.0: 1.04, 6.0: 1.06, 7.0: 1.07}
        # wsd = 3.3
        key = min(factors.keys(), key=lambda x: abs(x - wsd))
        factor = factors[key]
    else:  # assume default wsd of 3 in.
        factor = 1.0

    # Equations for sf and p_e from NEH CH2 P147 (165 in pdf)
    wsd = 3
    sf = 0.531747 + 0.295164*wsd - 0.057697*(wsd**2) + 0.003804*(wsd**3)

    # Get monthly ET, round to nearest 1
    et = df['u'].to_list()

    # create monthly precip above
    # Get monthly mean precip, round to nearest 0.5
    # NEH says to use full monthly precip and et, then do the table,
    # then multiply by the month fraction again to get effective precip.
    # Is that before or after I interpolate the values?
    pm = df['rain'] * 0.73  # for livingston 12 S
    pmr = (round(pm * 2) / 2)
    print(pmr.iloc[0])

    ep_tb = []
    ep_eq = []
    for i in range(len(df)):
        ep_tb.append(factor * table_ep[str(int(et[i]))][pmr.iloc[i]])  # 5 and 3.5 should yield 2.48... And it does! :)
        p_e = sf * (0.70917 * pow(pmr.iloc[i], 0.82416)) * pow(10, (0.02426 * int(et[i])))
        ep_eq.append(p_e)

    df['ep_tb'] = ep_tb
    df['ep_eq'] = ep_eq

    return df, season_start, season_end, kc


# This is the "authoritative" version
def modified_blaney_criddle_1_ep(clim_db_loc, station, lat_degrees=None, elev=None, fullmonth=False, pivot=True):
    """
    Custom implementation of the SCS Blaney Criddle method.
    Updated to not rely on information outside IWR climate db
    IWR climate db is already in Fahrenheit.
    This one also calculates effective precip, assuming a dry year/80% chance.
    :param station: last 4 digits of station number
    :param lat_degrees:
    :param elev:
    :param clim_db_loc: path to IWR climate database
    :param fullmonth: set growing season to a predefined period of full months
    :param pivot: boolean for irrigation type: either pivot (True) or other (False)
    :return:
    """
    # 2000 is used arbitrarily to get a year's worth of daily time stamps. Right?
    yr_ind = pd.date_range('2000-01-01', '2000-12-31', freq='d')

    table = Table(clim_db_loc)
    # finding correct row in IWR db
    i = 0
    while table[i]['Station No'][2:] != station:
        i += 1
    row = table[i]
    # print(row['Station Name'])
    # loading monthly mean temps from IWR db as dict
    t = pd.Series({1: row['T Jan'], 2: row['T Feb'], 3: row['T Mar'], 4: row['T Apr'], 5: row['T May'],
                  6: row['T Jun'], 7: row['T Jul'], 8: row['T Aug'], 9: row['T Sep'], 10: row['T Oct'],
                  11: row['T Nov'], 12: row['T Dec']})
    ppt = pd.Series({1: row['P Jan'], 2: row['P Feb'], 3: row['P Mar'], 4: row['P Apr'], 5: row['P May'],
                   6: row['P Jun'], 7: row['P Jul'], 8: row['P Aug'], 9: row['P Sep'], 10: row['P Oct'],
                   11: row['P Nov'], 12: row['P Dec']})
    # print(t)
    # print(ppt)
    season_end = pd.to_datetime(row['Fall mo/dy 28'] + '/2000')  # average freeze date from database

    # # calculating season end?
    # month = t[t >= 53].index[-1]  # get last month when temp gets above 50
    # month_len = monthrange(2000, month)[1]  # get length of month
    # season_end1 = datetime(year=2000, month=month, day=15)  # get midpoint of month
    # # calculate number of days past midpoint of month when we reach average temp of 50
    # days = round(((53. - t[month]) * month_len) / (t[month + 1] - t[month]))
    # season_end1 = season_end1 + timedelta(days=days)  # add days
    # print(season_end1)

    # calculating season start
    month = t[t >= 50].index[0]  # get month when temp gets above 50
    month_len = monthrange(2000, month - 1)[1]  # get length of preceding month
    season_start = datetime(year=2000, month=month - 1, day=15)  # get midpoint of preceding month
    # calculate number of days past midpoint of preceding month when we reach average temp of 50
    days = round(((50. - t[month - 1]) * month_len) / (t[month] - t[month - 1]))
    season_start = season_start + timedelta(days=days)  # add days

    if fullmonth:
        # something a little weird with start dates/first period... non-inclusive of last day of month?
        season_start = pd.to_datetime('2000-05-01')  # ex: 4-30 vs 5-01 doesn't change result.
        season_end = pd.to_datetime('2000-10-01')

    season_length = (season_end - season_start).days

    lat = round(lat_degrees)
    sunshine = lat_to_sunshine[lat]

    first_period = []
    d = season_start
    while d.day != 1:
        first_period.append(d)
        d += timedelta(days=1)

    if d != season_start:
        midpoint = season_start + (d - season_start) / 2
        if midpoint.hour != 0:
            midpoint = midpoint - timedelta(hours=12) # to avoid splitting a day in half
        counter = (midpoint - season_start).days
        t_prev, t_next = t.loc[midpoint.month], t.loc[midpoint.month + 1]
        month_len = monthrange(2000, midpoint.month)[1]
        interp_fraction = (midpoint.day - 15) / month_len
        # print('interp_fraction 1: ', interp_fraction)
        remaining_days = month_len - season_start.day
        month_fraction1 = remaining_days / month_len
        temp = t_prev + (interp_fraction * (t_next - t_prev))

        day_prev, day_next = sunshine[midpoint.month - 1], sunshine[midpoint.month]
        daylight = (day_prev + (interp_fraction * (day_next - day_prev))) * month_fraction1

        ppt_prev, ppt_next = ppt.loc[midpoint.month], ppt.loc[midpoint.month + 1]
        precip = (ppt_prev + (interp_fraction * (ppt_next - ppt_prev))) * month_fraction1
    else:
        month_fraction1 = 1  # here for effective precip estimates
        midpoint = season_start + timedelta(days=15)
        counter = (midpoint - season_start).days
        temp = t.loc[midpoint.month]
        daylight = sunshine[midpoint.month - 1]
        precip = ppt.loc[midpoint.month]

    dates, d_accum, pct_season = [midpoint], [counter], [counter / season_length]
    temps, pct_day_hrs, precips = [temp], [daylight], [precip]
    for d in pd.date_range(midpoint, season_end, freq='d'):
        counter += 1
        if d.day == 15:
            dates.append(d)
            d_accum.append(counter)
            pct_season.append(counter / season_length)
            temps.append(t.loc[d.month])
            pct_day_hrs.append(sunshine[d.month - 1])
            precips.append(ppt.loc[d.month])

    if season_end.day >= 15:  # remove last entry
        dates = dates[:-1]
        d_accum = d_accum[:-1]
        pct_season = pct_season[:-1]
        pct_day_hrs = pct_day_hrs[:-1]
        temps = temps[:-1]
        precips = precips[:-1]

    # start of second period should always be first of the month
    second_period = []
    d = dates[-1]
    while d.day != 1:
        d += timedelta(days=1)
    while d != season_end:
        second_period.append(d)
        d += timedelta(days=1)

    if len(second_period) > 0:
        midpoint = second_period[0] + (second_period[-1] - second_period[0]) / 2
        if midpoint.hour != 0:
            midpoint = midpoint - timedelta(hours=12)  # to avoid splitting a day in half
    else:
        midpoint = season_end

    t_prev, t_next = t.loc[midpoint.month - 1], t.loc[midpoint.month]
    remaining_days = len(second_period) + 1
    month_len = monthrange(2000, midpoint.month)[1]
    month_fraction2 = remaining_days / month_len
    # prev_month_len = monthrange(2000, midpoint.month - 1)[1] ## do we need this?
    interp_fraction = (midpoint.day + 15) / month_len
    # print('interp_fraction 2: ', interp_fraction)
    temp = t_prev + (interp_fraction * (t_next - t_prev))
    accum_days_last = (midpoint - season_start).days

    day_prev, day_next = sunshine[midpoint.month - 2], sunshine[midpoint.month - 1]
    daylight = (day_prev + (interp_fraction * (day_next - day_prev))) * month_fraction2

    ppt_prev, ppt_next = ppt.loc[midpoint.month - 1], ppt.loc[midpoint.month]
    precip = (ppt_prev + (interp_fraction * (ppt_next - ppt_prev))) * month_fraction2

    dates.append(midpoint)
    d_accum.append(accum_days_last)
    pct_season.append(accum_days_last / season_length)
    temps.append(temp)
    pct_day_hrs.append(daylight)
    precips.append(precip)

    dates = [pd.to_datetime('2000-{}-{}'.format(d.month, d.day)) for d in dates]
    df = pd.DataFrame(np.array([d_accum, pct_season, temps, precips, pct_day_hrs]).T,
                      columns=['accum_day', 'pct_season', 't', 'rain', 'p'],
                      index=dates)
    # t = mean air temp by period
    # p = percentage of annual daylight hours by period

    df['f'] = df['t'] * df['p'] / 100.  # monthly consumptive use factor

    df['kt'] = df['t'] * 0.0173 - 0.314
    df['kt'][df['t'] < 36.] = 0.3

    elevation_corr = 1 + (0.1 * np.floor(elev / 1000.))  # from footnote 3 on IWR results page
    # elevation_corr = 1 + (0.1 * round(elev / 1000.))  # I think floor is better
    # The rounding made Dillon much worse, did not affect Busby, and improved Chinook slightly (over vs undershoot).

    # kc = pd.Series(alfalfa_kc, index=[d for d in yr_ind if d.day == 15])
    kc = pd.Series(alfalfa_kc_1, index=[d for d in yr_ind if (d.day == 1) or (d.day == 15)])
    kc = kc.reindex(yr_ind)
    kc.iloc[0] = 0.6
    kc.iloc[-1] = 0.6
    kc = kc.interpolate()
    df['kc'] = kc.loc[df.index]  # growth stage from table
    df['k'] = df['kc'] * elevation_corr * df['kt']  # empirical crop coefficient, corrected for air temp.
    df['u'] = df['k'] * df['f']  # monthly consumptive use, inches
    df['ref_u'] = df['kt'] * df['f']  # no crop coefficient or elevation correction.

    # print(df)

    # Effective Precip Spaghetti!
    # From NEH Ch2 Table 2-43
    table_ep = effective_ppt_table()
    # Keys are water storage depth in inches. Where is that information?
    # Assume that this does not change with time.
    wsd = None
    if wsd:
        factors = {0.75: 0.72, 1.0: 0.77, 1.5: 0.86, 2.0: 0.93, 2.5: 0.97, 3.0: 1.00,
                   4.0: 1.02, 5.0: 1.04, 6.0: 1.06, 7.0: 1.07}
        # wsd = 3.3
        key = min(factors.keys(), key=lambda x: abs(x - wsd))
        factor = factors[key]
    else:  # assume default wsd of 3 in.
        factor = 1.0

    # Equations for sf and p_e from NEH CH2 P147 (165 in pdf)
    wsd = 3
    sf = 0.531747 + 0.295164*wsd - 0.057697*(wsd**2) + 0.003804*(wsd**3)

    # create monthly precip above
    # Get monthly mean precip, round to nearest 0.5
    # NEH says to use full monthly precip and et, then do the table,
    # then multiply by the month fraction again to get effective precip.
    # Is that before or after I interpolate the values? - Let's say after.

    # Get monthly ET, round to nearest 1 (in table lookup below)
    et = df['u'].to_list()
    pm = df['rain']
    # correct the first and last to be full month estimates.
    if month_fraction1 != 0:
        et[0] = et[0] / month_fraction1
        pm.iloc[0] = pm.iloc[0] / month_fraction1
    else:
        et[0] = 0
        pm.iloc[0] = 0
    if month_fraction2 != 0:
        et[-1] = et[-1] / month_fraction2
        pm.iloc[-1] = pm.iloc[-1] / month_fraction2
    else:
        et[-1] = 0
        pm.iloc[-1] = 0

    # Accounting for irrigation type
    if pivot:
        net_irr = 1.0
    else:
        net_irr = 4.0
    carryover = net_irr / 4.0  # start and end value, total seasonal is twice this value.

    # Evenly distributing net irrigation application across growing season
    pct_ssn_new = df['pct_season']
    pct_ssn_shift = np.zeros(len(df))
    pct_ssn_shift[1:] = df['pct_season'].iloc[:-1]
    pct_ssn_new = pct_ssn_new - pct_ssn_shift
    monthly_irr = pct_ssn_new * net_irr

    # We now have mean monthly precip. Need to get ratio for 80% chance.
    # Let's see if I can replicate that.
    # total growing season precip for each year on record. that would need to come from .txt files...
    # This would be easier to implement with the daily data. Leave it alone for the IWR stuff.
    avg_ann_ppt = sum(ppt.values)
    avg_precip = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90]
    ratios_80 = [0.45, 0.5, 0.54, 0.57, 0.60, 0.62, 0.63, 0.65, 0.69, 0.71, 0.73, 0.74, 0.75, 0.77,
                 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90]
    ratios_table_80 = dict(zip(avg_precip, ratios_80))  # column from NEH table 2-46 (pdf 171)
    key = min(ratios_table_80.keys(), key=lambda x: abs(x - avg_ann_ppt))
    ratio80 = ratios_table_80[key]
    pm = (pm + monthly_irr) * ratio80  # Irrigation applied before ratio.
    # Assumes irrigation volume is affected by dry years?
    # print("avg ann precip: ", avg_ann_ppt, "ratio: ", ratio80)

    # Then round the results to input into table.
    pmr = (round(pm * 2) / 2)

    ep_tb = []
    ep_eq = []
    for i in range(len(df)):
        ep_tb.append(factor * table_ep[str(int(et[i]))][pmr.iloc[i]])
        p_e = sf * (0.70917 * pow(pmr.iloc[i], 0.82416)) * pow(10, (0.02426 * int(et[i])))
        ep_eq.append(p_e)

    # Get first and last months back into fractions.
    ep_tb[0] = ep_tb[0] * month_fraction1
    ep_tb[-1] = ep_tb[-1] * month_fraction2
    ep_eq[0] = ep_eq[0] * month_fraction1
    ep_eq[-1] = ep_eq[-1] * month_fraction2

    df['ep_tb'] = ep_tb
    df['ep_eq'] = ep_eq

    # Net irrigation requirements ("consumptive use")
    df['cu'] = df['u'] - df['ep_tb']

    df['cu_no_co'] = df['cu']  # column can be removed later, for verification purposes
    # Accounting for start and end of season carryover

    # start of season carryover
    beg_co = carryover
    i = 0
    while beg_co != 0:
        if df['cu'].iloc[i] > beg_co:
            df['cu'].iloc[i] = df['cu'].iloc[i] - beg_co
            beg_co = 0
        else:
            beg_co = beg_co - df['cu'].iloc[i]
            df['cu'].iloc[i] = 0
            i += 1

    # end of season carryover
    end_co = carryover
    i = 1
    while end_co != 0:
        if df['cu'].iloc[-i] > end_co:
            df['cu'].iloc[-i] = df['cu'].iloc[-i] - end_co
            end_co = 0
        else:
            end_co = end_co - df['cu'].iloc[-i]
            df['cu'].iloc[-i] = 0
            i += 1

    df['co'] = df['cu'] - df['cu_no_co']  # to check implementation

    return df, season_start, season_end, kc


if __name__ == '__main__':
    x1 = np.linspace(0.5, 11.5, 12)
    # print(x1)
    x2 = np.linspace(0, 11.5, 24)
    # print(x2)

    plt.figure()
    plt.plot(x1+1, alfalfa_kc)
    plt.plot(x2+1, alfalfa_kc_1)
    plt.xlabel('month')
    plt.ylabel('k_c')
    # plt.vlines([5,10], 0.6, 1.15)
    plt.grid()
    # plt.show()

    pass
# ========================= ========================================================
