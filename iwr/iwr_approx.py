import os
from datetime import timedelta, datetime, date
from calendar import monthrange
import pandas as pd
import numpy as np
from pypxlib import Table
import matplotlib.pyplot as plt

# Required for running IWR on daily weather data files.
from utils.elevation import elevation_from_coordinate

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

alfalfa_kc = [0.6, 0.63, 0.68, 0.73, 0.79, 0.86, 0.92, 0.98, 1.04, 1.08, 1.12, 1.13,
              1.12, 1.11, 1.09, 1.06, 1.03, 0.99, 0.95, 0.91, 0.85, 0.78, 0.72, 0.64]


def effective_ppt_table(loc=None):
    """ Load effective precip table from file.

    From National Engineering Handbook (NEH) Ch 2, pg 148, Table 2-43.

    Parameters
    ----------
    loc: str, optional; full filepath to effective precip table stored as csv.

    Returns
    -------
    pandas DataFrame of effective precipitation table.
    """
    if loc:
        return pd.read_csv(loc, index_col=0)
    else:
        if os.path.exists('F:/FileShare'):
            main_dir = 'F:/FileShare/openet_pilot'
        else:
            main_dir = 'F:/openet_pilot'
        _file = os.path.join(main_dir, 'eff_precip_neh_chap2.csv')
        return pd.read_csv(_file, index_col=0)


def iwr_database(clim_db_loc, station, fullmonth=False, pivot=True):
    """ Replicate functionality of IWR as used by MT DNRC for HUA analysis using IWR databases.

    Only works for stations listed in Table 1 in Rule 36.12.1902
    Custom implementation of the SCS Blaney Criddle method.
    Does not rely on information outside climate database included in IWR
    Climate db is already in Fahrenheit.
    Calculates effective precip assuming a dry year/80% chance.

    Parameters
    ----------
    station: str, last 4 digits of station number
    clim_db_loc: path to IWR climate database
    fullmonth: optional, bool, set growing season to a predefined period of full months (for testing purposes)
    pivot: optional, bool describing irrigation type: either pivot (True) or other (False)

    Returns
    -------
    pandas DataFrame with intermediate calculations, and the dates of the growing season.
    Last 3 columns of dataframe are first 3 columns of IWR program output. Sum columns to get totals.
    """

    # 2000 is used arbitrarily to get a year's worth of daily time stamps.
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

    # # calculating season end? This produces poorer results.
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
        season_start = pd.to_datetime('2000-04-01')  # ex: 4-30 vs 5-01 doesn't change result.
        season_end = pd.to_datetime('2000-09-30')

    season_length = (season_end - season_start).days

    lat = round(row['Latitude'] / 100)
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
    # prev_month_len = monthrange(2000, midpoint.month - 1)[1]  # Is this needed?
    interp_fraction = (midpoint.day + 15) / month_len
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

    elev = row['Elevation'] / 3.281  # feet to meters
    elevation_corr = 1 + (0.1 * (elev / 1000.))  # from footnote 3 on IWR results page

    kc = pd.Series(alfalfa_kc, index=[d for d in yr_ind if (d.day == 1) or (d.day == 15)])
    kc = kc.reindex(yr_ind)
    kc.iloc[0] = 0.6
    kc.iloc[-1] = 0.6
    kc = kc.interpolate()
    df['kc'] = kc.loc[df.index]  # growth stage from table
    df['k'] = df['kc'] * elevation_corr * df['kt']  # empirical crop coefficient, corrected for air temp.
    df['ref_u'] = df['kt'] * df['f']  # no crop coefficient or elevation correction.
    df['u'] = df['k'] * df['f']  # monthly consumptive use, inches, "total monthly ET" in IWR.

    # Effective precipitation calculations (From NEH Ch2)

    # Keys are water storage depth in inches. Dict replaces eq. 2-85.
    wsd = None
    if wsd:
        factors = {0.75: 0.72, 1.0: 0.77, 1.5: 0.86, 2.0: 0.93, 2.5: 0.97, 3.0: 1.00,
                   4.0: 1.02, 5.0: 1.04, 6.0: 1.06, 7.0: 1.07}
        key = min(factors.keys(), key=lambda x: abs(x - wsd))
        factor = factors[key]
    else:  # assume default wsd of 3 in.
        factor = 1.0

    # NEH says to use full monthly precip and et, then do the table,
    # then multiply by the month fraction again to get effective precip.
    # Assumed to be after the values are interpolated for partial months.

    # Get monthly ET and precip
    et = df['u'].to_list()
    pm = df['rain']
    # correct the first and last periods to be full month estimates.
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

    # We now have mean monthly precip. Next get ratio for 80% chance.
    avg_ann_ppt = sum(ppt.values)
    avg_precip = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90]
    ratios_80 = [0.45, 0.5, 0.54, 0.57, 0.60, 0.62, 0.63, 0.65, 0.69, 0.71, 0.73, 0.74, 0.75, 0.77,
                 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90]
    ratios_table_80 = dict(zip(avg_precip, ratios_80))  # column from NEH table 2-46 (pdf 171)
    key = min(ratios_table_80.keys(), key=lambda x: abs(x - avg_ann_ppt))
    ratio80 = ratios_table_80[key]
    pm = (pm + monthly_irr) * ratio80  # Irrigation applied before ratio.
    # Assumes irrigation volume is affected by dry years?

    # Then round the results to input into table.
    pmr = (round(pm * 2) / 2)

    ep = []
    # table lookup replaced by eq. 2-84 to avoid limits on max ET. rounded inputs kept to match table. (and IWR?)
    for i in range(len(df)):
        epi = factor * ((0.70917 * pmr.iloc[i] ** 0.82416) - 0.11556) * (10 ** (0.02426 * int(et[i])))
        epi = np.round(epi, 2)
        if epi < et[i] and epi < pmr.iloc[i]:
            ep.append(epi)
        else:
            ep.append(min(et[i], pm.iloc[i]))

    # Get first and last month back into fractions.
    ep[0] = ep[0] * month_fraction1
    ep[-1] = ep[-1] * month_fraction2
    df['ep'] = ep

    # Net irrigation requirements ("consumptive use")
    df['cu'] = df['u'] - df['ep']

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

    table.close()
    return df, season_start, season_end


def iwr_daily_fm(df, lat_degrees=None, elev=None, season_start='2000-04-01', season_end='2000-09-30', pivot=True):
    """
    Replicates functionality of IWR as used by MT DNRC for HUA analysis with daily time series data.

    Location not restricted to IWR stations, given required data.
    Custom implementation of the SCS Blaney Criddle method.
    Assumes inout data is in Celsius and rain in mm.
    Includes calculation of effective precip per NEH Ch2, pgs 147-152 (pdf 165-170)
    Will only do full-month periods, calculations for partial months
    at the start and end of the growing season have been removed.
    It is not advised to change the default start and end dates.

    Parameters
    ----------
    df: pandas DataFrame with meteorological time series data: average daily temperature (Celsius)
    and daily precipitation (mm)
    lat_degrees: number, latitude of location
    elev: number, elevation of location
    season_start: optional, str, should be first of month
    season_end: optional, str, should be last of month
    pivot: optional, bool describing irrigation type: either pivot (True) or other (False)

    Returns
    -------
    dataframe with intermediate calculations, and the dates of the growing season.
    Last 3 columns of dataframe are first 3 columns of IWR program output. Sum columns to get totals.
    """

    # NEH 2-233 "...mean temperature is assumed to occur on the 15th day of each month..."
    # however, this gives results that differ substantially from NRCS IWR database files
    mid_months = [d for d in df.index if d.day == 15]
    t = df['MM'].loc[mid_months].resample('M').mean()
    t = t.groupby(t.index.month).mean() * 9 / 5 + 32
    # print(t)

    # precipitation data
    p = df['PP'].resample('M').sum()
    p = p.groupby(p.index.month).mean() / 25.4  # mm to in
    annual_p = df['PP'].resample('Y').sum() / 25.4

    dtmm = df['MM'].groupby([df.index.month, df.index.day]).mean() * 9 / 5 + 32
    if len(dtmm) > 365:
        # For leap years/longer periods of record
        yr_ind = pd.date_range('2000-01-01', '2000-12-31', freq='d')
    else:
        # For single non-leap years
        yr_ind = pd.date_range('1998-01-01', '1998-12-31', freq='d')
    dtmm.index = yr_ind

    if not season_start:  # Using daily data
        season_start = dtmm[dtmm > 50.].index[0]
        start_month = season_start.month - 1

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
        start_month = season_start.month - 1

    if not season_end:  # Using monthly interpolation
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
        end_month = season_end.month
        # print(season_end)
    else:
        season_end = pd.to_datetime(season_end)
        end_month = season_end.month

    season_length = (season_end - season_start).days

    lat = round(lat_degrees)
    sunshine = lat_to_sunshine[lat]

    dates = pd.date_range(season_start, season_end, freq='MS') + timedelta(days=14)
    d_accum = (dates - season_start).days
    pct_season = d_accum/season_length
    temps = t.loc[dates.month]
    precips = p.loc[dates.month]
    # months = (dates1.month - 1).to_list()
    pct_day_hrs = sunshine[start_month:end_month]

    if len(dtmm) > 365:
        dates = [pd.to_datetime('2000-{}-{}'.format(d.month, d.day)) for d in dates]
    else:
        dates = [pd.to_datetime('1998-{}-{}'.format(d.month, d.day)) for d in dates]
    df = pd.DataFrame(np.array([d_accum, pct_season, temps, precips, pct_day_hrs]).T,
                      columns=['accum_day', 'pct_season', 't', 'rain', 'p'],
                      index=dates)
    # t = mean monthly air temp
    # p = monthly percentage of annual daylight hours

    df['f'] = df['t'] * df['p'] / 100.  # monthly consumptive use factor

    df['kt'] = df['t'] * 0.0173 - 0.314
    df['kt'][df['t'] < 36.] = 0.3

    elevation_corr = 1 + (0.1 * (elev / 1000.))  # from footnote 3 on IWR results page

    kc = pd.Series(alfalfa_kc, index=[d for d in yr_ind if (d.day == 1) or (d.day == 15)])
    kc = kc.reindex(yr_ind)
    kc.iloc[0] = 0.6
    kc.iloc[-1] = 0.6
    kc = kc.interpolate()
    df['kc'] = kc.loc[df.index]  # growth stage from table
    df['k'] = df['kc'] * elevation_corr * df['kt']  # empirical crop coefficient, corrected for air temp.
    df['ref_u'] = df['kt'] * df['f']  # no crop coefficient or elevation correction.
    df['u'] = df['k'] * df['f']  # monthly consumptive use, inches, "total monthly ET" in IWR.

    # Effective precipitation calculations (From NEH Ch2)

    # Keys are water storage depth in inches. Dict replaces eq. 2-85.
    wsd = None
    if wsd:
        factors = {0.75: 0.72, 1.0: 0.77, 1.5: 0.86, 2.0: 0.93, 2.5: 0.97, 3.0: 1.00,
                   4.0: 1.02, 5.0: 1.04, 6.0: 1.06, 7.0: 1.07}
        key = min(factors.keys(), key=lambda x: abs(x - wsd))
        factor = factors[key]
    else:  # assume default wsd of 3 in.
        factor = 1.0

    # Get monthly ET and precip
    et = df['u'].to_list()
    pm = df['rain']

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

    # We now have mean monthly precip. Next get ratio for 80% chance.
    # avg_ann_ppt = sum(ppt.values)
    avg_precip = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90]
    ratios_80 = [0.45, 0.5, 0.54, 0.57, 0.60, 0.62, 0.63, 0.65, 0.69, 0.71, 0.73, 0.74, 0.75, 0.77,
                 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90]
    ratios_table_80 = dict(zip(avg_precip, ratios_80))  # column from NEH table 2-46 (pdf 171)
    key = min(ratios_table_80.keys(), key=lambda x: abs(x - annual_p.mean()))
    ratio80 = ratios_table_80[key]
    pm = (pm + monthly_irr) * ratio80  # Irrigation applied before ratio.
    # Assumes irrigation volume is affected by dry years?

    # Then round the results to input into table.
    pmr = (round(pm * 2) / 2)

    ep = []
    # table lookup replaced by eq. 2-84 to avoid limits on max ET. rounded inputs kept to match table. (and IWR?)
    for i in range(len(df)):
        epi = factor * ((0.70917 * pmr.iloc[i]**0.82416) - 0.11556) * (10**(0.02426*int(et[i])))
        epi = np.round(epi, 2)
        if epi < et[i] and epi < pmr.iloc[i]:
            ep.append(epi)
        else:
            ep.append(min(et[i], pm.iloc[i]))
    df['ep'] = ep

    # Net irrigation requirements (almost "consumptive use", just needs management factor)
    df['cu'] = df['u'] - df['ep']

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

    return df, season_start, season_end


def iwr_daily(df, lat_degrees=None, elev=None, season_start=None, season_end=None, pivot=True):
    """
    Replicates functionality of IWR as used by MT DNRC for HUA analysis with daily time series data.
    This function can implement the algortihm at any location given the required data
    Custom implementation of the SCS Blaney Criddle method.
    Assumes inout data is in Celsius and rain in mm.
    Includes calculation of effective precip per NEH Ch2, pgs 147-152 (pdf 165-170)

    Parameters
    ----------
    df:
    lat_degrees:
    elev:
    season_start:
    season_end:
    pivot: bool, optional; whether the field is irrgiated with a center pivot (True) or not (False)

    Returns
    -------
    df:
    season_start:
    season_end:
    """

    # NEH 2-233 "...mean temperature is assumed to occur on the 15th day of each month..."
    # however, this gives results that differ substantially from NRCS IWR database files
    mid_months = [d for d in df.index if d.day == 15]
    t = df['MM'].loc[mid_months].resample('M').mean()
    t = t.groupby(t.index.month).mean() * 9 / 5 + 32
    # print(t)

    # precipitation data
    p = df['PP'].resample('M').sum()
    p = p.groupby(p.index.month).mean() / 25.4  # mm to in
    annual_p = df['PP'].resample('Y').sum() / 25.4

    dtmm = df['MM'].groupby([df.index.month, df.index.day]).mean() * 9 / 5 + 32
    yr_ind = pd.date_range('2000-01-01', '2000-12-31', freq='d')
    dtmm.index = yr_ind

    if not season_start:  # Using daily data
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

    if not season_end:  # Using monthly interpolation
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
    # prev_month_len = monthrange(2000, midpoint.month - 1)[1]  # Is this needed?
    interp_fraction = (midpoint.day + 15) / month_len
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
    # t = mean monthly air temp
    # p = monthly percentage of annual daylight hours

    df['f'] = df['t'] * df['p'] / 100.  # monthly consumptive use factor

    df['kt'] = df['t'] * 0.0173 - 0.314
    df['kt'][df['t'] < 36.] = 0.3

    elevation_corr = 1 + (0.1 * (elev / 1000.))  # from footnote 3 on IWR results page

    kc = pd.Series(alfalfa_kc, index=[d for d in yr_ind if (d.day == 1) or (d.day == 15)])
    kc = kc.reindex(yr_ind)
    kc.iloc[0] = 0.6
    kc.iloc[-1] = 0.6
    kc = kc.interpolate()
    df['kc'] = kc.loc[df.index]  # growth stage from table
    df['k'] = df['kc'] * elevation_corr * df['kt']  # empirical crop coefficient, corrected for air temp.
    df['ref_u'] = df['kt'] * df['f']  # no crop coefficient or elevation correction.
    df['u'] = df['k'] * df['f']  # monthly consumptive use, inches, "total monthly ET" in IWR.

    # Effective precipitation calculations (From NEH Ch2)

    # Keys are water storage depth in inches. Dict replaces eq. 2-85.
    wsd = None
    if wsd:
        factors = {0.75: 0.72, 1.0: 0.77, 1.5: 0.86, 2.0: 0.93, 2.5: 0.97, 3.0: 1.00,
                   4.0: 1.02, 5.0: 1.04, 6.0: 1.06, 7.0: 1.07}
        key = min(factors.keys(), key=lambda x: abs(x - wsd))
        factor = factors[key]
    else:  # assume default wsd of 3 in.
        factor = 1.0

    # NEH says to use full monthly precip and et, then do the table,
    # then multiply by the month fraction again to get effective precip.
    # Assumed to be after the values are interpolated for partial months.

    # Get monthly ET and precip
    et = df['u'].to_list()
    pm = df['rain']
    # correct the first and last periods to be full month estimates.
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

    # We now have mean monthly precip. Next get ratio for 80% chance.
    # avg_ann_ppt = sum(ppt.values)
    avg_precip = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90]
    ratios_80 = [0.45, 0.5, 0.54, 0.57, 0.60, 0.62, 0.63, 0.65, 0.69, 0.71, 0.73, 0.74, 0.75, 0.77,
                 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90]
    ratios_table_80 = dict(zip(avg_precip, ratios_80))  # column from NEH table 2-46 (pdf 171)
    key = min(ratios_table_80.keys(), key=lambda x: abs(x - annual_p.mean()))
    ratio80 = ratios_table_80[key]
    pm = (pm + monthly_irr) * ratio80  # Irrigation applied before ratio.
    # Assumes irrigation volume is affected by dry years?

    # Then round the results to input into table.
    pmr = (round(pm * 2) / 2)

    ep = []
    # table lookup replaced by eq. 2-84 to avoid limits on max ET. rounded inputs kept to match table. (and IWR?)
    for i in range(len(df)):
        epi = factor * ((0.70917 * pmr.iloc[i] ** 0.82416) - 0.11556) * (10 ** (0.02426 * int(et[i])))
        epi = np.round(epi, 2)
        if epi < et[i] and epi < pmr.iloc[i]:
            ep.append(epi)
        else:
            ep.append(min(et[i], pm.iloc[i]))

    # Get first and last month back into fractions.
    ep[0] = ep[0] * month_fraction1
    ep[-1] = ep[-1] * month_fraction2
    df['ep'] = ep

    # Net irrigation requirements ("consumptive use")
    df['cu'] = df['u'] - df['ep']

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

    return df, season_start, season_end


def run_one_iwr_station(station='2409', clim_db_loc=None, data_dir=None,
                        start='1970-01-01', end='2000-12-31', pivot=True):
    """ Runs the IWR algorithm for a single location, and prints out the results.

    Runs either iwr_db, iwr_daily, or both depending on which file paths are provided as parameters.
    Start and end dates do not affect iwr_db.

    Parameters
    ----------
    station: str, optional; the 4-digit identifier of the IWR station
    clim_db_loc:
    data_dir:
    start:
    end:
    pivot: bool, optional; whether the field is irrgiated with a center pivot (True) or not (False)
    """
    if clim_db_loc:
        print('Using IWR database:')
        bc, start1, end1 = iwr_database(clim_db_loc, station, fullmonth=False, pivot=pivot)
        print('Season: ', start1.date(), ' to ', end1.date())
        print(bc[['u', 'ep', 'cu']])
        print('total ET: ', bc['u'].sum())
        print('total EP: ', bc['ep'].sum())
        print('total CU: ', bc['cu'].sum())
        print()
    if data_dir:
        print('Using daily met data:')
        _file = os.path.join(data_dir, 'USC0024{}.csv'.format(station))
        df = pd.read_csv(_file)
        dt_index = pd.date_range(start, end)
        df.index = pd.to_datetime(df['DATE'])
        df = df.reindex(dt_index)

        lat = df.iloc[0]['LATITUDE']
        lon = df.iloc[0]['LONGITUDE']
        elev = elevation_from_coordinate(lat, lon)

        df = df[['TMAX', 'TMIN', 'PRCP']]

        df['MX'] = df['TMAX'] / 10.
        df['MN'] = df['TMIN'] / 10.
        df['PP'] = df['PRCP'] / 10.
        df = df[['MX', 'MN', 'PP']]
        df['MM'] = (df['MX'] + df['MN']) / 2
        bc, start, end = iwr_daily_fm(df, lat_degrees=lat, elev=elev, pivot=pivot)
        print('Season: ', start.date(), ' to ', end.date())
        print(bc[['u', 'ep', 'cu']])
        print('total ET: ', bc['u'].sum())
        print('total EP: ', bc['ep'].sum())
        print('total CU: ', bc['cu'].sum())


def run_all_iwr_stations(clim_db_loc, out_file, data_dir=None, start='1971-01-01', end='2000-12-31'):
    """ Runs IWR algorthm (iwr_database) for all stations in the IWR database.
    If data_dir is provided, also runs iwr_daily on stations where daily data file exists."""
    table = Table(clim_db_loc)
    out = pd.DataFrame(index=range(1, 181), columns=['station_num', 'station_name', 'db_et', 'db_flood_ep_80',
                                                     'db_flood_cu_80', 'db_pivot_ep_80', 'db_pivot_cu_80',
                                                     'db_season_start', 'db_season_end'])
    for i in range(len(table)):
        station = table[i]['Station No'][2:]
        # print(station)
        bc1, start1, end1 = iwr_database(clim_db_loc, station, fullmonth=False, pivot=False)
        bc2, start2, end2 = iwr_database(clim_db_loc, station, fullmonth=False, pivot=True)
        out.at[i+1, 'station_num'] = station
        out.at[i+1, 'station_name'] = table[i]['Station Name']
        out.at[i+1, 'db_et'] = bc1['u'].sum()
        out.at[i+1, 'db_flood_ep_80'] = bc1['ep'].sum()
        out.at[i+1, 'db_flood_cu_80'] = bc1['cu'].sum()
        out.at[i+1, 'db_pivot_ep_80'] = bc2['ep'].sum()
        out.at[i+1, 'db_pivot_cu_80'] = bc2['cu'].sum()
        out.at[i+1, 'db_season_start'] = start1
        out.at[i+1, 'db_season_end'] = end1
    if data_dir:  # run on daily data too
        print('Running with daily data...')
        out[['daily_et', 'daily_flood_ep_80', 'daily_flood_cu_80', 'daily_pivot_ep_80', 'daily_pivot_cu_80',
             'daily_season_start', 'daily_season_end']] = None
        for i in range(len(table)):
            station = table[i]['Station No'][2:]
            # print(station)
            _file = os.path.join(data_dir, 'USC0024{}.csv'.format(station))
            if os.path.isfile(_file):
                df = pd.read_csv(_file)

                lat = df.iloc[0]['LATITUDE']
                lon = df.iloc[0]['LONGITUDE']
                elev = elevation_from_coordinate(lat, lon)

                dt_index = pd.date_range(start, end)
                df.index = pd.to_datetime(df['DATE'])
                df = df.reindex(dt_index)

                df = df[['TMAX', 'TMIN', 'PRCP']]

                df['MX'] = df['TMAX'] / 10.
                df['MN'] = df['TMIN'] / 10.
                df['PP'] = df['PRCP'] / 10.
                df = df[['MX', 'MN', 'PP']]
                df['MM'] = (df['MX'] + df['MN']) / 2
                bc1, start1, end1 = iwr_daily(df, lat_degrees=lat, elev=elev, pivot=False)
                bc2, start2, end2 = iwr_daily(df, lat_degrees=lat, elev=elev, pivot=True)

                out.at[i + 1, 'daily_et'] = bc1['u'].sum()
                out.at[i + 1, 'daily_flood_ep_80'] = bc1['ep'].sum()
                out.at[i + 1, 'daily_flood_cu_80'] = bc1['cu'].sum()
                out.at[i + 1, 'daily_pivot_ep_80'] = bc2['ep'].sum()
                out.at[i + 1, 'daily_pivot_cu_80'] = bc2['cu'].sum()
                out.at[i + 1, 'daily_season_start'] = start1
                out.at[i + 1, 'daily_season_end'] = end1
            else:
                continue
    out.to_csv(out_file)


def plot_growing_season_starts_and_ends(clim_db_loc):
    table = Table(clim_db_loc)

    # Looking at season end dates
    end_dates = []
    doy = []
    months = []
    for i in range(len(table)):
        temp = table[i]['Fall mo/dy 28']
        date_m = temp[:2]
        date_d = temp[3:]
        end_date = date(year=2000, month=int(date_m), day=int(date_d))
        end_dates.append(end_date)
        doy.append(end_date.timetuple().tm_yday)
        months.append(end_date.month)

    # print(min(end_dates))
    # print(max(end_dates))
    # print(min(doy))
    # print(max(doy))

    bins = np.arange(230, 303)
    labels = pd.date_range(start='2000-08-17', end='2000-10-28').date

    # Looking at season start dates
    starts = []
    doys = []
    monthss = []
    for i in range(len(table)):
        row = table[i]
        t = pd.Series({1: row['T Jan'], 2: row['T Feb'], 3: row['T Mar'], 4: row['T Apr'], 5: row['T May'],
                       6: row['T Jun'], 7: row['T Jul'], 8: row['T Aug'], 9: row['T Sep'], 10: row['T Oct'],
                       11: row['T Nov'], 12: row['T Dec']})
        month = t[t >= 50].index[0]  # get month when temp gets above 50
        month_len = monthrange(2000, month - 1)[1]  # get length of preceding month
        season_start = datetime(year=2000, month=month - 1, day=15)  # get midpoint of preceding month
        # calculate number of days past midpoint of preceding month when we reach average temp of 50
        days = round(((50. - t[month - 1]) * month_len) / (t[month] - t[month - 1]))
        season_start = season_start + timedelta(days=days)  # add days

        starts.append(season_start)
        doys.append(season_start.timetuple().tm_yday)
        monthss.append(season_start.month)

    # print(min(starts))
    # print(max(starts))
    # print(min(doys))
    # print(max(doys))

    binss = np.arange(107, 171)
    labelss = pd.date_range(start='2000-04-16', end='2000-6-18').date

    # Plotting

    # # Start and end days
    # plt.figure()
    # plt.subplot(211)
    # plt.title('Starts')
    # plt.hist(doys, bins=binss, align='left', zorder=5)
    # plt.xticks(binss, labelss, rotation='vertical')
    # plt.grid(zorder=0)
    #
    # plt.subplot(212)
    # plt.title('Ends')
    # plt.hist(doy, bins=bins, align='left', zorder=5)
    # plt.xticks(bins, labels, rotation='vertical')
    # plt.grid(zorder=0)
    # plt.tight_layout()

    # # Start and end months
    # plt.figure()
    # plt.subplot(121)
    # plt.title('Starts')
    # counts1, edges1, bars1 = plt.hist(monthss)
    # plt.bar_label(bars1)
    # plt.xlabel('Month')
    #
    # plt.subplot(122)
    # plt.title('Ends')
    # counts, edges, bars = plt.hist(months)
    # plt.bar_label(bars)
    # plt.xlabel('Month')

    # Default growing season lengths
    lens = np.array(doy) - np.array(doys)
    bins = np.arange(70, 190, 10)

    plt.figure()
    plt.title('Lengths')
    counts1, edges1, bars1 = plt.hist(lens, bins=bins)
    plt.bar_label(bars1)
    plt.xlabel('Days')

    # Default growing seasons
    year = pd.date_range("01-01-2000", "12-15-2000", freq='SMS')
    year_day = pd.date_range("01-01-2000", "12-15-2000", freq='D')

    daily_kc = pd.Series(alfalfa_kc, year)
    daily_kc = daily_kc.reindex(year_day)
    daily_kc = daily_kc.interpolate()

    ys = np.linspace(0, 1.8, 180)
    print(starts[0])
    starts = [i.date() for i in starts]
    times = pd.DataFrame({'start': pd.to_datetime(starts), 'end': pd.to_datetime(end_dates), 'length': lens})
    print(type(times['end'].iloc[0]), type(times['start'].iloc[0]))
    print(times['end'].iloc[0], times['start'].iloc[0])
    times['middle'] = times['start'] + (times['end'] - times['start'])/2
    times = times.sort_values('length', ascending=False)
    print(times)

    bins = np.arange(182, 214)
    labels = pd.date_range(start='2000-07-01', end='2000-08-01').date

    plt.figure()
    plt.title('Growing Season Midpoints')
    counts1, edges1, bars1 = plt.hist(lens, bins=bins)
    plt.hist(times['middle'], bins=labels, align='left', zorder=5)
    plt.bar_label(bars1)
    plt.xticks(rotation='vertical')
    plt.grid(zorder=0)

    # plt.figure()
    # for i in range(180):
    #     plt.hlines(ys[i], times['start'].iloc[i], times['end'].iloc[i])
    # plt.scatter(times['middle'], ys, color='tab:pink')
    # plt.vlines(times['middle'].mean(), 0, 1.8, 'tab:orange',
    #            label='Avg GS midpoint: {}'.format(times['middle'].mean().date()))
    # plt.fill_between([times['middle'].mean() - 2*times['middle'].std(), times['middle'].mean() + 2*times['middle'].std()],
    #                  [0], [1.8], color='tab:orange', alpha=0.6, ec='none',
    #                  label='+/- {}'.format(2*times['middle'].std()))
    #                  # label="{} {}".format((times['middle'].mean() - 2*times['middle'].std()),
    #                  #                      (times['middle'].mean() + 2*times['middle'].std())))
    # plt.plot(daily_kc, 'k')
    # # plt.ylabel('IWR Alfalfa KC')
    # plt.xlabel('Month')
    # plt.xlim(date(year=2000, month=4, day=1), date(year=2000, month=11, day=1))
    # plt.legend()
    # plt.grid()
    # plt.show()

    plt.show()


def plot_kc_and_gs():
    year = pd.date_range("01-01-2001", "12-15-2001", freq='SMS')
    year_day = pd.date_range("01-01-2001", "12-15-2001", freq='D')

    daily_kc = pd.Series(alfalfa_kc, year)
    daily_kc = daily_kc.reindex(year_day)
    daily_kc = daily_kc.interpolate()

    plt.figure()
    plt.plot(daily_kc)

    plt.vlines(date(month=4, day=1, year=2001), 0.6, 1.2, 'tab:pink')
    plt.vlines(date(month=9, day=30, year=2001), 0.6, 1.2, 'tab:pink')
    plt.hlines(daily_kc.loc['04-01-2001':'09-30-2001'].mean(), date(month=4, day=1, year=2001),
               date(month=9, day=30, year=2001), 'tab:pink',
               label="GS Avg: {:.2f}".format(daily_kc.loc['04-01-2001':'09-30-2001'].mean()))

    # # David's growing season for Park and Sweet Grass counties in memo
    # plt.vlines(date(month=5, day=9, year=2001),0.6, 1.2, 'tab:purple')
    # plt.vlines(date(month=9, day=19, year=2001),0.6, 1.2, 'tab:purple')
    # plt.hlines(daily_kc.loc['05-09-2001':'09-19-2001'].mean(), date(month=5, day=9, year=2001),
    #            date(month=9, day=19, year=2001), 'tab:purple',
    #            label="GS Avg: {:.2f}".format(daily_kc.loc['05-09-2001':'09-19-2001'].mean()))

    plt.ylabel('IWR Alfalfa KC')
    plt.xlabel('Month')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dir_ = 'C:/Users/CND571/Documents'

    # File path for iwr climate database for use in 'iwr_db'
    iwr_clim_db_loc = 'C:/Users/CND571/Documents/IWR/Database/climate.db'

    # run_one_iwr_station('1995', clim_db_loc=iwr_clim_db_loc)

    # File path for output generated by 'check_all_iwr'
    iwr_sum = os.path.join(dir_, 'iwr_cu_est_all_02062024.csv')
    # run_all_iwr_stations(iwr_clim_db_loc, iwr_sum)

    # Directory of daily historical weather data for use in 'iwr_daily'
    # One file per station
    daily_data_dir = os.path.join(dir_, 'data', 'from_ghcn')
    # time period to use for daily data, if different one needed
    # pos_start, pos_end = '1971-01-01', '2000-12-31'  # default, period used in IWR

    # run_one_iwr_station('2409', data_dir=daily_data_dir)

    # run_all_iwr_stations(iwr_clim_db_loc, iwr_sum, daily_data_dir)

    plot_growing_season_starts_and_ends(iwr_clim_db_loc)

    # plot_kc_and_gs()

# ========================= EOF ====================================================================
