
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from shapely.geometry import Point
import xarray
import os
import sqlite3
from netCDF4 import Dataset
from cftime import num2pydate

import requests
import json
from tqdm import tqdm
import geopandas as gpd
import pyproj
from rasterstats import zonal_stats
from chmdata.thredds import GridMet
from scipy.stats import linregress

# For density plots
# from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
import matplotlib as mpl

from reference_et.combination import calc_press, calc_psy, calc_vpc, calc_ea, calc_es, get_rn, calc_e0
from utils.agrimet import load_stations, Agrimet
from utils.elevation import elevation_from_coordinate
from reference_et.rad_utils import extraterrestrial_r, calc_rso


def wind_2m(uz, zw=10):
    """ Convert wind measured at a height of zw meters to 2 meters. For measurements over clipped grass.

    See https://www.apogeeinstruments.com/content/EWRI-ASCE-Reference-ET-Appendices.pdf
    pdf page 47, page B-10 if more detailed information is needed.

    Parameters
    ----------
    uz: wind speed measured at zw meters, m/s
    zw: height of wind measurement, meters
    """
    u2 = (uz * 4.87) / np.log(67.8 * zw - 5.42)
    return u2


def calc_qsat(tmean, pres):
    """ Saturation specific humidity.

    Intermediate values:
    es: saturation vapor pressure in kPa

    Parameters
    ----------
    tmean: temperature in C
    pres: air pressure in kPa

    Returns
    -------
    qs: saturation specific humidty in g/kg


    For reference, see https://pressbooks-dev.oer.hawaii.edu/atmo/chapter/chapter-4-water-vapor/#:~:text=Specific%20humidity%2C%20q%2C%20is%20the,es%20instead%20of%20e.
    """
    ep = 0.622
    es = calc_es(tmean)  # kPa
    # qs = (ep * es) / pres  # Estimation?
    qs = (ep * es) / (pres - es * (1 - ep))  # more precise? g/kg
    return qs


def calculate_vpd_temponly(t_min, t_max):
    """ Calculate vapor pressure deficit using only max and min temp in Celsius. """
    etmin = 0.6108 * np.exp((17.27 * t_min) / (t_min + 237.3))
    etmax = 0.6108 * np.exp((17.27 * t_max) / (t_max + 237.3))
    es = (etmax + etmin) / 2.0
    ea = etmin
    vpd = es - ea

    return vpd


def pm_fao56_ref(tmean, wind, rs=None, rn=None, g=0, tmax=None, tmin=None,
                 rhmax=None, rhmin=None, rh=None, pressure=None, elevation=None,
                 lat=None, n=None, nn=None, rso=None, a=1.35, b=-0.35,
                 albedo=0.23, ref='grass'):
    """ Evaporation calculated according to [allen_1998]_.

    Parameters
    ----------
    tmean: pandas.Series
        average day temperature [°C]
    wind: pandas.Series
        mean day wind speed [m/s] (I think it's 2m)
    rs: pandas.Series, optional
        incoming solar radiation [MJ m-2 d-1]
    rn: pandas.Series, optional
        net radiation [MJ m-2 d-1]
    g: pandas.Series/int, optional
        soil heat flux [MJ m-2 d-1]
    tmax: pandas.Series, optional
        maximum day temperature [°C]
    tmin: pandas.Series, optional
        minimum day temperature [°C]
    rhmax: pandas.Series, optional
        maximum daily relative humidity [%]
    rhmin: pandas.Series, optional
        mainimum daily relative humidity [%]
    rh: pandas.Series, optional
        mean daily relative humidity [%]
    pressure: float, optional
        atmospheric pressure [kPa]
    elevation: float, optional
        the site elevation [m]
    lat: float, optional
        the site latitude [rad]
    n: pandas.Series/float, optional
        actual duration of sunshine [hour]
    nn: pandas.Series/float, optional
        maximum possible duration of sunshine or daylight hours [hour]
    rso: pandas.Series/float, optional
        clear-sky solar radiation [MJ m-2 day-1]
    a: float, optional
        empirical coefficient for Net Long-Wave radiation [-]
    b: float, optional
        empirical coefficient for Net Long-Wave radiation [-]
    albedo: float, optional
        surface albedo [-]
    ref: str, optional
        determines whether to use coefficients for grass or alfalfa reference crop (producing eto or etf, respectively)

    Returns
    -------
        pandas.Series containing the calculated evaporation, eto (or etf) in mm

    Examples
    --------
    # >>> et_fao56_ref = pm_fao56_ref(tmean, wind, rn=rn, rh=rh)

    Notes
    -----
    .. math:: PE = \\frac{0.408 \\Delta (R_{n}-G)+\\gamma \\frac{900}{T+273}
        (e_s-e_a) u_2}{\\Delta+\\gamma(1+0.34 u_2)}

    """
    if tmean is None:
        tmean = (tmax + tmin) / 2
    if pressure is None:
        pressure = calc_press(elevation)
    gamma = calc_psy(pressure)
    dlt = calc_vpc(tmean)

    # See ASCE-EWRI Task Committee Report, January, 2005
    if ref == 'grass' or ref == 'eto':
        cn = 900
        cd = 0.34
    elif ref == 'alfalfa' or ref == 'etr':
        cn = 1600
        cd = 0.38
    else:
        print("Unknown value for 'ref' variable, defaulting to grass")
        cn = 900
        cd = 0.34

    gamma1 = (gamma * (1 + cd * wind))

    if (rhmax is None) and (rhmin is None) and (rh is None):
        # from https://github.com/MTDNRC-WRD/pydlem/blob/main/prep/metdata.py#L194
        etmin = 0.6108 * np.exp((17.27 * tmin) / (tmin + 237.3))
        etmax = 0.6108 * np.exp((17.27 * tmax) / (tmax + 237.3))
        es = (etmax + etmin) / 2.0
        ea = etmin
    else:
        ea = calc_ea(tmean=tmean, tmax=tmax, tmin=tmin, rhmax=rhmax, rhmin=rhmin,
                     rh=rh)
        es = calc_es(tmean=tmean, tmax=tmax, tmin=tmin)

    if rn is None:
        rn = get_rn(tmean, rs, lat, n, nn, tmax, tmin, rhmax, rhmin, rh,
                    elevation, rso, a, b, ea, albedo)

    den = dlt + gamma1
    num1 = (0.408 * dlt * (rn - g)) / den
    num2 = (gamma * (es - ea) * cn * wind / (tmean + 273)) / den
    return num1 + num2


def agrimet_data(stn, save=False):
    """ Download AgriMet data for a single station, with option to save as csv or use as pd dataframe.
    Parameters
    ----------
    stn: desired agrimet station from which to pull data.
    save: bool, optional; determines whether to save data to csv file (True) or not (False)

    Returns
    -------
    none if save=True, pandas dataframe of station data if save=False.
    """
    # am_idx = pd.date_range("1984-01-01", end="2023-12-31", freq="D")
    am_idx = pd.date_range("2017-01-01", end="2023-12-31", freq="D")
    all_am = pd.DataFrame(index=am_idx)

    stations = load_stations()

    # Get elevation
    coords = stations[stn]['geometry']['coordinates']
    # geo = Point(coords)
    coord_rads = np.array(coords) * np.pi / 180
    elev = elevation_from_coordinate(coords[1], coords[0])

    # Get weather data
    stn = Agrimet(station=stn, region=stations[stn]['properties']['region'],
                  start_date='1984-01-01', end_date='2023-12-31')
    data = stn.fetch_met_data()
    # print(data.columns)
    data.columns = data.columns.droplevel([1, 2])
    all_am['ETkp'] = data['ET'] / 25.4  # Kimberly-Penman ET
    # all_am['ETkp'].mask(all_am['ETkp'] > 1)

    # print(data.index)
    tmean, tmax, tmin, wind, rs, rh = data['MM'], data['MX'], data['MN'], data['UA'], data['SR'], data['TA']
    ra = extraterrestrial_r(data.index, lat=coord_rads[1], shape=[data.shape[0]])
    rso = calc_rso(ra, elev)
    rn = get_rn(tmean, rs=rs, lat=coord_rads[1], tmax=tmax, tmin=tmin, rh=rh, elevation=elev, rso=rso)
    all_am['ETo'] = pm_fao56_ref(tmean, wind, rs=rs, tmax=tmax, tmin=tmin, rh=rh, elevation=elev, rn=rn, ref='grass')
    all_am['ETr'] = pm_fao56_ref(tmean, wind, rs=rs, tmax=tmax, tmin=tmin, rh=rh, elevation=elev, rn=rn, ref='alfalfa')
    all_am['ETo'] = all_am['ETo'] / 25.4  # mm to inches
    all_am['ETr'] = all_am['ETr'] / 25.4
    # df['ETRS'] = df['ETOS'] * 1.2  # What's the basis for this?

    all_am = all_am.mask(all_am > 1)

    # print("ra", ra)
    # print("elev", elev)
    # print("rn", rn)

    if save:
        all_am.to_csv('C:/Users/CND571/Documents/Data/all_agrimet_daily_etr_in_through2023.csv')
    else:
        return all_am


def era5_v_nldas():
    """ Stepping through each variable/calculation step to make nldas work."""
    am_list = pd.read_csv('C:/Users/CND571/Documents/Data/AgriMet_station_list.csv')

    # Loading ERA5 data
    # Temp and wind (daily dates)
    ds1 = xarray.open_dataset(
        "C:/Users/CND571/Downloads/adaptor.mars.internal-1720196759.3423746-29201-7-bd6c10fb-8f01-4350-a875-da1e6fdfa933.grib",
        engine="cfgrib")
    # Radiation (I think these could be added to e/et)
    ds2 = xarray.open_dataset(
        "C:/Users/CND571/Downloads/adaptor.mars.internal-1720194014.1757193-26708-10-a5d02e50-c9cc-4ac0-b020-047ddaf316ef.grib",
        engine="cfgrib")
    # Pre-calculated e/et (daily dates minus six hours, time mismatch w/ temp and wind)
    ds3 = xarray.open_dataset(
        "C:/Users/CND571/Downloads/adaptor.mars.internal-1719011409.8211007-13587-9-ab3d96eb-309e-4bfd-bafc-c6f71672405e.grib",
        engine="cfgrib")
    # print(ds1)
    # print(ds2)
    # print(ds3)
    # era5 = 0

    # Loading NLDAS data
    nldas_nc = Dataset("C:/Users/CND571/Downloads/NLDAS_FORA0125_M.2.0_Aggregation.ncml.nc")
    nldas_nc1 = Dataset("C:/Users/CND571/Downloads/NLDAS_NOAH0125_M.2.0_Aggregation.ncml.nc")
    print(nldas_nc.variables.keys())

    # Setting up dataframes
    era5 = pd.DataFrame(index=ds3.coords['valid_time'])
    era5.index = era5.index - dt.timedelta(hours=6)
    # print(era5.index)

    nldas = pd.DataFrame()
    nldas.index = num2pydate(nldas_nc.variables['time'][:-5], units=nldas_nc.variables['time'].units)

    # Location info
    i = 'COVM'
    lon = am_list[am_list['StationID'] == i]['StationLongitude'].values[0]
    lat = am_list[am_list['StationID'] == i]['StationLatitude'].values[0]
    elev = elevation_from_coordinate(lat, lon)
    # for NLDAS indexing
    lats = np.asarray(nldas_nc.variables['lat'][:])
    lons = np.asarray(nldas_nc.variables['lon'][:])
    lat_idx = np.argmin(np.abs(lats - lat))
    lon_idx = np.argmin(np.abs(lons - lon))

    # column names
    et = '{}_ET'.format(i)
    eta = '{}_ETa'.format(i)
    eto = '{}_ETo'.format(i)
    etr = '{}_ETr'.format(i)
    pet = '{}_PET'.format(i)

    # precomputed
    # rough agreement, NLDAS is higher
    era5[et] = ds3.pev.sel(longitude=lon, latitude=lat, method='nearest')
    era5[eta] = ds3.e.sel(longitude=lon, latitude=lat, method='nearest')
    era5[et] = era5.index.daysinmonth * np.abs(era5[et]) * (1000 / 25.4)  # m/day to in/month
    era5[eta] = era5.index.daysinmonth * np.abs(era5[eta]) * (1000 / 25.4)  # m/day to in/month
    nldas[pet] = pd.Series(nldas_nc.variables['PotEvap'][:-5, lat_idx, lon_idx],
                           index=nldas.index) / 25.4  # mm/month to in/month
    # plt.figure()
    # plt.plot(era5[et], label='ERA5')
    # plt.plot(nldas[pet], label='NLDAS')
    # plt.ylabel('PET')
    # plt.xlim(dt.datetime(year=2010, month=1, day=1), dt.datetime(year=2020, month=1, day=1))
    # plt.grid()
    # plt.legend()

    # temp
    # rough agreement, temps high in nldas, especially in winter
    tmean_e = ds1.t2m.sel(longitude=lon, latitude=lat, method='nearest').to_series() - 273.15  # K to C
    tmean_n = pd.Series(nldas_nc.variables['Tair'][:-5, lat_idx, lon_idx], index=nldas.index) - 273.15  # K to C
    # plt.figure()
    # plt.plot(tmean_e, label='ERA5')
    # plt.plot(tmean_n, label='NLDAS')
    # plt.ylabel('Temperature (C)')
    # plt.xlim(dt.datetime(year=2010, month=1, day=1), dt.datetime(year=2020, month=1, day=1))
    # plt.grid()
    # plt.legend()

    # wind
    # nldas has much higher wind speeds
    wind_e = ds1.si10.sel(longitude=lon, latitude=lat,
                          method='nearest').to_series()  # m/s at 10m
    wind_e = wind_2m(wind_e, 10)  # m/s at 2m
    winde_n = np.abs(nldas_nc.variables['Wind_E'][:-5, lat_idx, lon_idx])  # easting component, m/s at 10m
    windn_n = np.abs(nldas_nc.variables['Wind_N'][:-5, lat_idx, lon_idx])  # northing component, m/s at 10m
    wind_n = pd.Series(np.sqrt(winde_n ** 2 + windn_n ** 2), index=nldas.index)
    wind_n = wind_2m(wind_n, 10)  # m/s at 2m
    # plt.figure()
    # plt.plot(wind_e, label='ERA5')
    # plt.plot(wind_n, label='NLDAS')
    # plt.ylabel('Wind Speed @ 2m (m/s)')
    # plt.xlim(dt.datetime(year=2010, month=1, day=1), dt.datetime(year=2020, month=1, day=1))
    # plt.grid()
    # plt.legend()

    # relative humidity
    # rough agreement, NLDAS is higher
    dew_e = ds1.d2m.sel(longitude=lon, latitude=lat, method='nearest').to_series() - 273.15  # K to C
    ea = calc_e0(dew_e)
    es = calc_e0(tmean_e)
    rh_e = 100 * ea / es
    pres_n = pd.Series(nldas_nc.variables['PSurf'][:-5, lat_idx, lon_idx], index=nldas.index) / 1000  # kPa
    q = pd.Series(nldas_nc.variables['Qair'][:-5, lat_idx, lon_idx], index=nldas.index)  # kg/kg
    rh_n = 100 * q / calc_qsat(tmean_n, pres_n)  # Still a little confused about units here, but this looks much better
    # plt.figure()
    # plt.plot(q, label='q')
    # plt.plot(calc_qsat(tmean_n, pres_n), label='qs')
    # plt.ylabel('Specific Humidity')
    # plt.xlim(dt.datetime(year=2010, month=1, day=1), dt.datetime(year=2020, month=1, day=1))
    # plt.grid()
    # plt.legend()
    # plt.figure()
    # plt.plot(rh_e, label='ERA5')
    # plt.plot(rh_n, label='NLDAS')
    # plt.ylabel('Relative Humidity')
    # plt.xlim(dt.datetime(year=2010, month=1, day=1), dt.datetime(year=2020, month=1, day=1))
    # plt.grid()
    # plt.legend()

    # net radiation
    rnl_e = - ds2.str.sel(longitude=lon, latitude=lat,
                          method='nearest').to_series() / 1e6  # Jm-2d-1 to MJm-2d-1, correct for pos downwards convention
    rs_e = ds2.ssrd.sel(longitude=lon, latitude=lat,
                        method='nearest').to_series() / 1e6  # Jm-2d-1 to MJm-2d-1 Not sure this is the right variable
    rnl_e.index = rnl_e.index + dt.timedelta(hours=6)
    rs_e.index = rs_e.index + dt.timedelta(hours=6)
    rns_e = (1 - 0.23) * rs_e
    rn_e = rns_e - rnl_e

    rs_n = pd.Series(nldas_nc.variables['SWdown'][:-5, lat_idx, lon_idx] +
                     nldas_nc.variables['LWdown'][:-5, lat_idx, lon_idx], index=nldas.index) * 0.0864  # W/m2 to MJ/m2/day
    rns_n = pd.Series(nldas_nc1.variables['SWnet'][:-1, lat_idx, lon_idx], index=nldas.index) * 0.0864
    rnl_n = abs(pd.Series(nldas_nc1.variables['LWnet'][:-1, lat_idx, lon_idx], index=nldas.index) * 0.0864)  # correct sign
    rn_n = rns_n - rnl_n

    # plt.figure()
    # # plt.plot(rs_e, label='ERA5')
    # # plt.plot(rs_n, label='NLDAS')  # like twice as high?
    # # plt.plot(rnl_e, label='ERA5')
    # # plt.plot(rnl_n, label='NLDAS')  # very similar
    # # plt.plot(rns_e, label='ERA5')
    # # plt.plot(rns_n, label='NLDAS')  # very similar
    # plt.plot(rn_e, label='ERA5')
    # plt.plot(rn_n, label='NLDAS')  # very similar
    # plt.ylabel('Radiation')
    # plt.xlim(dt.datetime(year=2010, month=1, day=1), dt.datetime(year=2020, month=1, day=1))
    # plt.grid()
    # plt.legend()

    # ref ET
    # average daily ref ET per month
    era5[eto] = pm_fao56_ref(tmean_e, wind_e, rs=rs_e, rh=rh_e, elevation=elev, rn=rn_e, ref='grass') / 25.4  # mm to inches
    era5[etr] = pm_fao56_ref(tmean_e, wind_e, rs=rs_e, rh=rh_e, elevation=elev, rn=rn_e, ref='alfalfa') / 25.4
    nldas[eto] = pm_fao56_ref(tmean_n, wind_n, rs=rs_n, rh=rh_n, elevation=elev, rn=rn_n,
                              ref='grass') / 25.4  # mm to inches
    nldas[etr] = pm_fao56_ref(tmean_n, wind_n, rs=rs_n, rh=rh_n, elevation=elev, rn=rn_n,
                              ref='alfalfa') / 25.4  # 1 kg/m2 = 1 mm
    # average monthly total ET
    era5[eto] = era5.index.daysinmonth * era5[eto]
    era5[etr] = era5.index.daysinmonth * era5[etr]
    nldas[eto] = nldas.index.daysinmonth * nldas[eto]
    nldas[etr] = nldas.index.daysinmonth * nldas[etr]
    plt.figure()
    # plt.plot(era5[et], label='ERA5 PET')
    # plt.plot(nldas[pet], label='NLDAS PET')
    plt.plot(era5[eto], label='ERA5 ETo')
    plt.plot(nldas[eto], label='NLDAS ETo')  # Looks good now, I think.
    plt.ylabel('ET')
    plt.xlim(dt.datetime(year=2010, month=1, day=1), dt.datetime(year=2020, month=1, day=1))
    plt.grid()
    plt.legend()


def era5_data(save=False):
    """ Convert ERA5 data from local grib files to pandas dataframe, with option to save as csv or use as pd dataframe.

    Parameters
    ----------
    save: bool, optional; determines whether to save data to csv file (True) or not (False)

    Returns
    -------
    none if save=True, pandas dataframe of station data if save=False.
    """
    am_list = pd.read_csv('C:/Users/CND571/Documents/Data/AgriMet_station_list.csv')
    # Load in grib files
    # (from https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=form)
    # Temp and wind (daily dates)
    ds1 = xarray.open_dataset("C:/Users/CND571/Downloads/adaptor.mars.internal-1720196759.3423746-29201-7-bd6c10fb-8f01-4350-a875-da1e6fdfa933.grib", engine="cfgrib")
    # Radiation (I think these could be added to e/et)
    ds2 = xarray.open_dataset("C:/Users/CND571/Downloads/adaptor.mars.internal-1720194014.1757193-26708-10-a5d02e50-c9cc-4ac0-b020-047ddaf316ef.grib", engine="cfgrib")
    # Pre-calculated e/et (daily dates minus six hours, time mismatch w/ temp and wind)
    ds3 = xarray.open_dataset("C:/Users/CND571/Downloads/adaptor.mars.internal-1719011409.8211007-13587-9-ab3d96eb-309e-4bfd-bafc-c6f71672405e.grib", engine="cfgrib")
    print(ds1)
    # print(ds2)
    # print(ds3)
    # era5 = 0
    era5 = pd.DataFrame(index=ds3.coords['valid_time'])
    era5.index = era5.index - dt.timedelta(hours=6)
    # print(era5.index)

    # plt.figure()
    for i in ['COVM', 'CRSM', 'MWSM']:
        et = '{}_ET'.format(i)
        eta = '{}_ETa'.format(i)
        eto = '{}_ETo'.format(i)
        etr = '{}_ETr'.format(i)
        lon = am_list[am_list['StationID'] == i]['StationLongitude'].values[0]
        lat = am_list[am_list['StationID'] == i]['StationLatitude'].values[0]
        elev = elevation_from_coordinate(lat, lon)
        # precomputed, wrong methods
        era5[et] = ds3.pev.sel(longitude=lon, latitude=lat, method='nearest')
        print(i, ds3.pev.sel(longitude=lon, latitude=lat, method='nearest'))
        era5[eta] = ds3.e.sel(longitude=lon, latitude=lat, method='nearest')
        era5[et] = np.abs(era5[et]) * 1000 / 25.4
        era5[eta] = np.abs(era5[eta]) * 1000 / 25.4
        # Other era5 variables
        tmean = ds1.t2m.sel(longitude=lon, latitude=lat, method='nearest').to_series() - 273.15  # K to C
        dew = ds1.d2m.sel(longitude=lon, latitude=lat, method='nearest').to_series() - 273.15  # K to C
        wind = ds1.si10.sel(longitude=lon, latitude=lat, method='nearest').to_series()  # Does this need to be corrected?
        # plt.plot(wind, label='{} wind 10'.format(i))
        wind = wind_2m(wind, 10)
        # plt.plot(wind, label='{} wind 2'.format(i))
        rnl = - ds2.str.sel(longitude=lon, latitude=lat, method='nearest').to_series() / 1e6  # Jm-2d-1 to MJm-2d-1, correct for positive downwards convention
        rs = ds2.ssrd.sel(longitude=lon, latitude=lat, method='nearest').to_series() / 1e6  # Jm-2d-1 to MJm-2d-1 Not sure this is the right variable
        rnl.index = rnl.index + dt.timedelta(hours=6)
        rs.index = rs.index + dt.timedelta(hours=6)
        # Calculate relative humidity from dew point and temp:
        ea = calc_e0(dew)
        es = calc_e0(tmean)
        rh = 100 * ea / es

        # radiation calcs
        rns = (1 - 0.23) * rs
        rn = rns - rnl

        # print(len(tmean), len(wind), len(rs), len(rh), len(rn))
        # print(tmean.index[0], wind.index[0], rs.index[0], rh.index[0], rn.index[0])
        # print(tmean[0], wind[0], rs[0], rh[0], rn[0])
        # # Length and index is fine, but some units are wrong?

        # print(pm_fao56_ref(tmean, wind, rs=rs, rh=rh, elevation=elev, rn=rn, ref='grass'))

        # ref ET
        # average daily ref ET per month
        era5[eto] = pm_fao56_ref(tmean, wind, rs=rs, rh=rh, elevation=elev, rn=rn, ref='grass') / 25.4  # mm to inches
        era5[etr] = pm_fao56_ref(tmean, wind, rs=rs, rh=rh, elevation=elev, rn=rn, ref='alfalfa') / 25.4
        # average monthly total ET
        era5[eto] = era5.index.daysinmonth * era5[eto]
        era5[etr] = era5.index.daysinmonth * era5[etr]
        # plt.plot(rnl, label='{} rnl'.format(i))
        # plt.plot(rn, label='{} rn'.format(i))
        # plt.plot(rs, label='{} rs'.format(i))
        # plt.plot(era5[eto], label=eto)
        # plt.plot(era5[etr], label=etr)
    # plt.grid()
    # plt.legend()

    if save:
        era5.to_csv('C:/Users/CND571/Documents/Data/era5_at3agrimet.csv')
    else:
        return era5


def era5_all_data(all_data):
    # Load in grib files
    # (from https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=form)
    # Temp and wind (daily dates)
    ds1 = xarray.open_dataset("C:/Users/CND571/Downloads/adaptor.mars.internal-1720196759.3423746-29201-7-bd6c10fb-8f01-4350-a875-da1e6fdfa933.grib", engine="cfgrib")
    # Radiation (I think these could be added to e/et)
    ds2 = xarray.open_dataset("C:/Users/CND571/Downloads/adaptor.mars.internal-1720194014.1757193-26708-10-a5d02e50-c9cc-4ac0-b020-047ddaf316ef.grib", engine="cfgrib")
    # Pre-calculated e/et (daily dates minus six hours, time mismatch w/ temp and wind)
    ds3 = xarray.open_dataset("C:/Users/CND571/Downloads/adaptor.mars.internal-1719011409.8211007-13587-9-ab3d96eb-309e-4bfd-bafc-c6f71672405e.grib", engine="cfgrib")
    # print(ds1)
    # print(ds2)
    # print(ds3)
    # era5 = 0

    # print(era5.index)

    # r_index = pd.date_range('2021-01-01', '2023-12-31', freq='D')

    for i in tqdm(all_data.index, total=len(all_data)):
        era5 = pd.DataFrame(index=ds3.coords['valid_time'])
        era5.index = era5.index - dt.timedelta(hours=6)
        et = '{}_ET'.format(i)
        eto = '{}_ETo'.format(i)
        lon = all_data.at[i, 'lon']
        lat = all_data.at[i, 'lat']
        elev = elevation_from_coordinate(lat, lon)
        # precomputed, wrong methods
        era5[et] = ds3.pev.sel(longitude=lon, latitude=lat, method='nearest')
        # Other era5 variables
        tmean = ds1.t2m.sel(longitude=lon, latitude=lat, method='nearest').to_series() - 273.15  # K to C
        dew = ds1.d2m.sel(longitude=lon, latitude=lat, method='nearest').to_series() - 273.15  # K to C
        wind = ds1.si10.sel(longitude=lon, latitude=lat, method='nearest').to_series()  # Does this need to be corrected?
        # plt.plot(wind, label='{} wind 10'.format(i))
        wind = wind_2m(wind, 10)
        # plt.plot(wind, label='{} wind 2'.format(i))
        rnl = - ds2.str.sel(longitude=lon, latitude=lat, method='nearest').to_series() / 1e6  # Jm-2d-1 to MJm-2d-1, correct for positive downwards convention
        rs = ds2.ssrd.sel(longitude=lon, latitude=lat, method='nearest').to_series() / 1e6  # Jm-2d-1 to MJm-2d-1 Not sure this is the right variable
        rnl.index = rnl.index + dt.timedelta(hours=6)
        rs.index = rs.index + dt.timedelta(hours=6)
        # Calculate relative humidity from dew point and temp:
        ea = calc_e0(dew)
        es = calc_e0(tmean)
        rh = 100 * ea / es

        # radiation calcs
        rns = (1 - 0.23) * rs
        rn = rns - rnl

        # ref ET
        # average daily ref ET per month
        era5[eto] = pm_fao56_ref(tmean, wind, rs=rs, rh=rh, elevation=elev, rn=rn, ref='grass') / 25.4  # mm to inches
        # average monthly total ET
        era5[eto] = era5.index.daysinmonth * era5[eto]

        df = era5
        df = df.truncate(dt.date(year=2021, month=1, day=1), dt.date(year=2023, month=12, day=31))
        df['mask'] = [1 if d.month in range(4, 10) else 0 for d in df.index]
        df = df[df['mask'] == 1]
        df['year'] = [d.year for d in df.index]
        # print(df.groupby(['year']).sum())
        # print(df.groupby(['year']).sum().mean()[eto])
        all_data.at[i, 'era5'] = df.groupby(['year']).sum().mean()[eto]

    # return era5


def nldas_data():
    """ Convert NLDAS data from local nc files to pandas dataframe, calculates etr/eto.

    Returns
    -------
    nldas: pandas dataframe of gridded data collocated with each Agrimet/Mesonet station.
    """
    am_list = pd.read_csv('C:/Users/CND571/Documents/Data/AgriMet_station_list.csv')

    nldas_nc = Dataset("C:/Users/CND571/Downloads/NLDAS_FORA0125_M.2.0_Aggregation.ncml.nc")
    nldas_nc1 = Dataset("C:/Users/CND571/Downloads/NLDAS_NOAH0125_M.2.0_Aggregation.ncml.nc")
    print(nldas_nc)
    # print(nldas_nc.variables.keys())
    # print(nldas_nc1.variables.keys())

    lats = np.asarray(nldas_nc.variables['lat'][:])
    lons = np.asarray(nldas_nc.variables['lon'][:])
    # print("lats", lats)
    # print("lons", lons)

    nldas = pd.DataFrame()
    nldas.index = num2pydate(nldas_nc.variables['time'][:-5], units=nldas_nc.variables['time'].units)

    for i in ['COVM', 'CRSM', 'MWSM']:
        eto = '{}_ETo'.format(i)
        etr = '{}_ETr'.format(i)
        pet = '{}_PET'.format(i)
        lat = am_list[am_list['StationID'] == i]['StationLatitude'].values[0]
        lon = am_list[am_list['StationID'] == i]['StationLongitude'].values[0]
        lat_idx = np.argmin(np.abs(lats - lat))
        lon_idx = np.argmin(np.abs(lons - lon))
        # print(i, lats[lat_idx], lons[lon_idx])
        elev = elevation_from_coordinate(lat, lon)

        tmean = pd.Series(nldas_nc.variables['Tair'][:-5, lat_idx, lon_idx], index=nldas.index) - 273.15  # K to C
        winde = np.abs(nldas_nc.variables['Wind_E'][:-5, lat_idx, lon_idx])
        windn = np.abs(nldas_nc.variables['Wind_N'][:-5, lat_idx, lon_idx])
        wind = pd.Series(np.sqrt(winde**2 + windn**2), index=nldas.index)  # m/s at 10m
        wind = wind_2m(wind, 10)  # m/s at 2m
        pres = pd.Series(nldas_nc.variables['PSurf'][:-5, lat_idx, lon_idx], index=nldas.index) / 1000  # kPa
        q = pd.Series(nldas_nc.variables['Qair'][:-5, lat_idx, lon_idx], index=nldas.index)
        rh = 100 * q / calc_qsat(tmean, pres)
        rs = pd.Series(nldas_nc.variables['SWdown'][:-5, lat_idx, lon_idx] +
                       nldas_nc.variables['LWdown'][:-5, lat_idx, lon_idx], index=nldas.index) * 0.0864  # W/m2 to MJ/m2/day
        rns = pd.Series(nldas_nc1.variables['SWnet'][:-1, lat_idx, lon_idx], index=nldas.index) * 0.0864
        rnl = abs(pd.Series(nldas_nc1.variables['LWnet'][:-1, lat_idx, lon_idx], index=nldas.index) * 0.0864)
        rn = rns - rnl

        nldas[eto] = pm_fao56_ref(tmean, wind, rs=rs, rh=rh, elevation=elev, rn=rn, ref='grass')
        nldas[etr] = pm_fao56_ref(tmean, wind, rs=rs, rh=rh, elevation=elev, rn=rn, ref='alfalfa')
        nldas[eto] = nldas.index.daysinmonth * nldas[eto] / 25.4  # mm/day to inches/month
        nldas[etr] = nldas.index.daysinmonth * nldas[etr] / 25.4  # 1 kg/m2 = 1 mm
        nldas[pet] = pd.Series(nldas_nc.variables['PotEvap'][:-5, lat_idx, lon_idx], index=nldas.index) / 25.4

    return nldas


def nldas_all_data(all_data):
    # am_list = pd.read_csv('C:/Users/CND571/Documents/Data/AgriMet_station_list.csv')

    nldas_nc = Dataset("C:/Users/CND571/Downloads/NLDAS_FORA0125_M.2.0_Aggregation.ncml.nc")
    nldas_nc1 = Dataset("C:/Users/CND571/Downloads/NLDAS_NOAH0125_M.2.0_Aggregation.ncml.nc")
    print(nldas_nc)
    # print(nldas_nc.variables.keys())
    # print(nldas_nc1.variables.keys())

    lats = np.asarray(nldas_nc.variables['lat'][:])
    lons = np.asarray(nldas_nc.variables['lon'][:])
    # print("lats", lats)
    # print("lons", lons)

    # for i in ['COVM', 'CRSM', 'MWSM']:
    for i in tqdm(all_data.index, total=len(all_data)):
        nldas = pd.DataFrame()
        nldas.index = num2pydate(nldas_nc.variables['time'][:-5], units=nldas_nc.variables['time'].units)
        eto = '{}_ETo'.format(i)
        etr = '{}_ETr'.format(i)
        # pet = '{}_PET'.format(i)
        lon = all_data.at[i, 'lon']
        lat = all_data.at[i, 'lat']
        lat_idx = np.argmin(np.abs(lats - lat))
        lon_idx = np.argmin(np.abs(lons - lon))
        # print(i, lats[lat_idx], lons[lon_idx])
        elev = elevation_from_coordinate(lat, lon)

        tmean = pd.Series(nldas_nc.variables['Tair'][:-5, lat_idx, lon_idx], index=nldas.index) - 273.15  # K to C
        winde = np.abs(nldas_nc.variables['Wind_E'][:-5, lat_idx, lon_idx])
        windn = np.abs(nldas_nc.variables['Wind_N'][:-5, lat_idx, lon_idx])
        wind = pd.Series(np.sqrt(winde**2 + windn**2), index=nldas.index)  # m/s at 10m
        wind = wind_2m(wind, 10)  # m/s at 2m
        pres = pd.Series(nldas_nc.variables['PSurf'][:-5, lat_idx, lon_idx], index=nldas.index) / 1000  # kPa
        q = pd.Series(nldas_nc.variables['Qair'][:-5, lat_idx, lon_idx], index=nldas.index)
        rh = 100 * q / calc_qsat(tmean, pres)
        rs = pd.Series(nldas_nc.variables['SWdown'][:-5, lat_idx, lon_idx] +
                       nldas_nc.variables['LWdown'][:-5, lat_idx, lon_idx], index=nldas.index) * 0.0864  # W/m2 to MJ/m2/day
        rns = pd.Series(nldas_nc1.variables['SWnet'][:-1, lat_idx, lon_idx], index=nldas.index) * 0.0864
        rnl = abs(pd.Series(nldas_nc1.variables['LWnet'][:-1, lat_idx, lon_idx], index=nldas.index) * 0.0864)
        rn = rns - rnl

        nldas[eto] = pm_fao56_ref(tmean, wind, rs=rs, rh=rh, elevation=elev, rn=rn, ref='grass')
        nldas[etr] = pm_fao56_ref(tmean, wind, rs=rs, rh=rh, elevation=elev, rn=rn, ref='alfalfa')
        nldas[eto] = nldas.index.daysinmonth * nldas[eto] / 25.4  # mm/day to inches/month
        nldas[etr] = nldas.index.daysinmonth * nldas[etr] / 25.4  # 1 kg/m2 = 1 mm
        # nldas[pet] = pd.Series(nldas_nc.variables['PotEvap'][:-5, lat_idx, lon_idx], index=nldas.index) / 25.4

        df = nldas
        df = df.truncate(dt.date(year=2021, month=1, day=1), dt.date(year=2023, month=12, day=31))
        df['mask'] = [1 if d.month in range(4, 10) else 0 for d in df.index]
        df = df[df['mask'] == 1]
        df['year'] = [d.year for d in df.index]
        # print(df.groupby(['year']).sum())
        # print(df.groupby(['year']).sum().mean()[eto])
        all_data.at[i, 'nldas'] = df.groupby(['year']).sum().mean()[eto]

    # return nldas


def stns_metadata(active=True):
    """ Retrieve Mesonet station metadata.

    Parameters
    ----------
    active: bool, optional; if True, retieve only currently active stations, if False, retrieve info for all stations.

    Returns
    -------
    A dictionary of station metadata.
    """

    if active:
        active = 'true'  # currently 155
    else:
        active = 'false'  # currently 220

    # Error in retrieving elements from inactive stations, so just deal with active ones for now.
    # How to handle this, especially when it is noted by changing the install date to 'None'?

    url = 'https://mesonet.climate.umt.edu/api/v2/stations/?public=true&active={}&type=json'.format(active)
    r = requests.get(url)
    stations = json.loads(r.text)

    stns_dict = {}
    for i in range(len(stations)):
        # retrieve station identifier
        temp = stations[i]['station']
        # remove station identifier from existing dictionary
        stations[i].pop('station')
        # store info as values in dictionary by station identifier keys
        stns_dict[temp] = stations[i]

    return stns_dict


def mesonet_data(stn, save=False):
    """ Download Mesonet data, with option to save as csv or use as pd dataframe.

    Parameters
    ----------
    stn: desired Mesonet station from which to pull data.
    save: bool, optional; determines whether to save data to csv file (True) or not (False)

    Returns
    -------
    none if save=True, pandas dataframe of station data if save=False.
    """

    mn_idx = pd.date_range("2021-01-01", end="2023-12-31", freq="D")
    all_mn = pd.DataFrame(index=mn_idx)

    mn_url = (
        'https://mesonet.climate.umt.edu/api/v2/derived/daily/?crop=corn&high=86&low=50&alpha=0.23&'
        'na_info=false&rm_na=false&premade=true&wide=true&keep=false&units=us&type=csv&tz=America%2FDenver&'
        'simple_datetime=false&end_time=2024-01-01T00%3A00%3A00&start_time=2021-01-01T00%3A00%3A00&'
        'level=1&stations={}&elements=etr'.format(stn))
    mn_station1 = pd.read_csv(mn_url, index_col='datetime')
    mn_station1.index = [j[:10] for j in mn_station1.index]
    mn_station1.index = pd.to_datetime(mn_station1.index)
    all_mn['ETo'] = mn_station1['Reference ET (a=0.23) [in]']  # Grass reference

    if save:
        all_mn.to_csv('C:/Users/CND571/Documents/Data/all_mesonet_daily_etr_in_through2023.csv')
    else:
        return all_mn


def am_mn_comp_plots(am_d, mn_d, am_m, mn_m, era5_m, gm_m, nldas_m, am_err=None, mn_err=None):
    """ Comparison plots for all different meteorological data sources.

    Parameters
    ----------
    am_d: pd df, daily Agrimet data
    mn_d: pd df, daily Mesonet data
    am_m: pd df, monthly Agrimet data
    mn_m: pd df, monthly Mesonet data
    era5_m: pd df, monthly ERA5 data
    gm_m: pd df, monthly GridMET data
    nldas_m: pd df, monthly NLDAS data
    am_err: pd df, optional; Agrimet number of days of missing data per month
    mn_err: pd df, optional; Mesonet number of days of missing data per month
    """
    # Time series comparison of 3 pairs of "collocated" Arigmet/Mesonet stations,
    # along with 3 sources of gridded met data
    plt.figure(figsize=(15, 10), dpi=200)

    plt.subplot(311)
    plt.title('Corvallis')
    # plt.plot(am_d['COVM'].index, am_d['COVM']['ETkp'], label='AgriMet ETkp')
    # plt.plot(am_d['COVM'].index, am_d['COVM']['ETo'], label='AgriMet ETo')
    # plt.plot(mn_d['corvalli'].index, mn_d['corvalli'], label='Mesonet ETo')
    # plt.plot(am_m['COVM'].index - dt.timedelta(days=15), am_m['COVM']['ETkp'], label='AgriMet ETkp (Monthly)')

    plt.plot(am_m['COVM'].index - dt.timedelta(days=15),
             am_m['COVM'].index.daysinmonth * am_m['COVM']['ETo'], label='AgriMet')
    plt.plot(mn_m['corvalli'].index - dt.timedelta(days=15),
             mn_m['corvalli'].index.daysinmonth * mn_m['corvalli']['ETo'], label='Mesonet')
    # plt.plot(am_m['COVM'].index - dt.timedelta(days=15), am_m['COVM']['ETo'], label='AgriMet')
    # plt.plot(mn_m['corvalli'].index - dt.timedelta(days=15), mn_m['corvalli']['ETo'], label='Mesonet')

    # plt.plot(era5_m.index + dt.timedelta(days=15), era5_m['COVM_ET'], label='ERA5 ET (Monthly)')
    plt.plot(gm_m.index, gm_m['COVM_ETo'], label='GridMET')
    plt.plot(era5_m.index + dt.timedelta(days=15), era5_m['COVM_ETo'], label='ERA5')
    plt.plot(nldas_m.index, nldas_m['COVM_ETo'], label='NLDAS ETo')
    # plt.plot(nldas_m.index, nldas_m['COVM_PET'], label='NLDAS PET')

    # plt.scatter(am_err['COVM'].index - dt.timedelta(days=15), am_err['COVM'], label='AgriMet missing days')
    # plt.scatter(mn_err['corvalli'].index - dt.timedelta(days=15), mn_err['corvalli'],
    #             label='Mesonet missing days')

    plt.ylabel('Monthly Average ETo (in)')
    # plt.xlabel('Date')
    plt.grid()
    plt.xlim(dt.date(year=2017, month=1, day=1), dt.date(year=2024, month=1, day=1))
    plt.ylim(0, 10)
    # plt.ylim(0, 10)
    # plt.legend()

    plt.subplot(312)
    plt.title('Kalispell')
    # plt.plot(am_d['CRSM'].index, am_d['CRSM']['ETkp'], label='AgriMet ETkp')
    # plt.plot(am_d['CRSM'].index, am_d['CRSM']['ETo'], label='AgriMet ETo')
    # plt.plot(mn_d['kalispel'].index, mn_d['kalispel'], label='Mesonet ETo')
    # plt.plot(am_m['CRSM'].index - dt.timedelta(days=15), am_m['CRSM']['ETkp'], label='AgriMet ETkp (Monthly)')

    plt.plot(am_m['CRSM'].index - dt.timedelta(days=15),
             am_m['CRSM'].index.daysinmonth * am_m['CRSM']['ETo'], label='AgriMet ETo (Monthly)')
    plt.plot(mn_m['kalispel'].index - dt.timedelta(days=15),
             mn_m['kalispel'].index.daysinmonth * mn_m['kalispel']['ETo'], label='Mesonet ETo (Monthly)')
    # plt.plot(am_m['CRSM'].index - dt.timedelta(days=15),
    #          am_m['CRSM'].index.daysinmonth * am_m['CRSM']['ETo'], label='AgriMet ETo (Monthly)')
    # plt.plot(mn_m['kalispel'].index - dt.timedelta(days=15),
    #          mn_m['kalispel'].index.daysinmonth * mn_m['kalispel']['ETo'], label='Mesonet ETo (Monthly)')

    # plt.plot(era5_m.index + dt.timedelta(days=15), era5_m['CRSM_ET'], label='ERA5 ET (Monthly)')
    plt.plot(gm_m.index, gm_m['CRSM_ETo'], label='GridMET ETo (Monthly)')
    plt.plot(era5_m.index + dt.timedelta(days=15), era5_m['CRSM_ETo'], label='ERA5 ETo (Monthly)')
    plt.plot(nldas_m.index, nldas_m['CRSM_ETo'], label='NLDAS ETo (Monthly)')
    # plt.plot(nldas_m.index, nldas_m['CRSM_PET'], label='NLDAS PET')

    # plt.scatter(am_err['CRSM'].index - dt.timedelta(days=15), am_err['CRSM'], label='AgriMet missing days')
    # plt.scatter(mn_err['kalispel'].index - dt.timedelta(days=15), mn_err['kalispel'],
    #             label='Mesonet missing days')

    plt.ylabel('Monthly Average ETo (in)')
    # plt.xlabel('Date')
    plt.grid()
    plt.xlim(dt.date(year=2017, month=1, day=1), dt.date(year=2024, month=1, day=1))
    plt.ylim(0, 8)
    # plt.ylim(0, 8)
    # plt.legend()

    plt.subplot(313)
    plt.title('Moccasin')
    # plt.plot(am_d['MWSM'].index, am_d['MWSM']['ETkp'], label='AgriMet ETkp')
    # plt.plot(am_d['MWSM'].index, am_d['MWSM']['ETo'], label='AgriMet ETo')
    # plt.plot(mn_d['moccasin'].index, mn_d['moccasin'], label='Mesonet ETo')
    # plt.plot(am_m['MWSM'].index - dt.timedelta(days=15), am_m['MWSM']['ETkp'], label='AgriMet ETkp (Monthly)')

    plt.plot(am_m['MWSM'].index - dt.timedelta(days=15),
             am_m['MWSM'].index.daysinmonth * am_m['MWSM']['ETo'], label='AgriMet')
    plt.plot(mn_m['moccasin'].index - dt.timedelta(days=15),
             mn_m['moccasin'].index.daysinmonth * mn_m['moccasin']['ETo'], label='Mesonet')
    # plt.plot(am_m['MWSM'].index - dt.timedelta(days=15), am_m['MWSM']['ETo'], label='AgriMet')
    # plt.plot(mn_m['moccasin'].index - dt.timedelta(days=15), mn_m['moccasin']['ETo'], label='Mesonet')

    # plt.plot(era5_m.index + dt.timedelta(days=15), era5_m['MWSM_ET'], label='ERA5 ET (Monthly)')
    plt.plot(gm_m.index, gm_m['MWSM_ETo'], label='GridMET')
    plt.plot(era5_m.index + dt.timedelta(days=15), era5_m['MWSM_ETo'], label='ERA5')
    plt.plot(nldas_m.index, nldas_m['MWSM_ETo'], label='NLDAS')
    # plt.plot(nldas_m.index, nldas_m['MWSM_PET'], label='NLDAS PET')

    # plt.scatter(am_err['MWSM'].index - dt.timedelta(days=15), am_err['MWSM'], label='AgriMet missing days')
    # plt.scatter(mn_err['moccasin'].index - dt.timedelta(days=15), mn_err['moccasin'],
    #             label='Mesonet missing days')

    plt.ylabel('Monthly Average ETo (in)')
    # plt.xlabel('Date')
    plt.grid()
    plt.xlim(dt.date(year=2017, month=1, day=1), dt.date(year=2024, month=1, day=1))
    plt.ylim(0, 12)
    # plt.ylim(0, 12)
    plt.legend(ncols=5)

    plt.tight_layout()

    # print((am_m['COVM']['ETkp']))
    # print(mn_m['corvalli']['ETo'])

    # # Difference between AgriMet and Mesonet monthly averages.
    # plt.figure()
    #
    # plt.subplot(211)
    # plt.title('ETkp vs ETo')
    # plt.hlines(1, dt.date(year=2019, month=1, day=1), dt.date(year=2024, month=1, day=1), 'k')
    # plt.plot(am_m['COVM'].index - dt.timedelta(days=15),
    #          (am_m['COVM']['ETkp']) / mn_m['corvalli']['ETo'], label='Corvallis')
    # plt.plot(am_m['CRSM'].index - dt.timedelta(days=15),
    #          (am_m['CRSM']['ETkp']) / mn_m['kalispel']['ETo'], label='Kalispell')
    # plt.plot(am_m['MWSM'].index - dt.timedelta(days=15),
    #          (am_m['MWSM']['ETkp']) / mn_m['moccasin']['ETo'], label='Moccasin')
    # plt.xlim(dt.date(year=2019, month=1, day=1), dt.date(year=2024, month=1, day=1))
    # plt.ylim(0, 2)
    # plt.ylabel('AgriMet/Mesonet')
    # plt.grid()
    #
    # plt.subplot(212)
    # plt.hlines(0, dt.date(year=2019, month=1, day=1), dt.date(year=2024, month=1, day=1), 'k')
    # plt.plot(am_m['COVM'].index - dt.timedelta(days=15),
    #          (am_m['COVM']['ETkp']) - mn_m['corvalli']['ETo'], label='Corvallis')
    # plt.plot(am_m['CRSM'].index - dt.timedelta(days=15),
    #          (am_m['CRSM']['ETkp']) - mn_m['kalispel']['ETo'], label='Kalispell')
    # plt.plot(am_m['MWSM'].index - dt.timedelta(days=15),
    #          (am_m['MWSM']['ETkp']) - mn_m['moccasin']['ETo'], label='Moccasin')
    # plt.xlim(dt.date(year=2019, month=1, day=1), dt.date(year=2024, month=1, day=1))
    # plt.ylim(-0.1, 0.2)
    # plt.ylabel('AgriMet minus Mesonet (in)')
    # plt.legend()
    # plt.grid()
    #
    # plt.figure()
    #
    # plt.subplot(211)
    # plt.title('ETo')
    # plt.hlines(1, dt.date(year=2019, month=1, day=1), dt.date(year=2024, month=1, day=1), 'k')
    # plt.plot(am_m['COVM'].index - dt.timedelta(days=15), (am_m['COVM']['ETo']) / mn_m['corvalli']['ETo'],
    #          label='Corvallis')
    # plt.plot(am_m['CRSM'].index - dt.timedelta(days=15), (am_m['CRSM']['ETo']) / mn_m['kalispel']['ETo'],
    #          label='Kalispell')
    # plt.plot(am_m['MWSM'].index - dt.timedelta(days=15), (am_m['MWSM']['ETo']) / mn_m['moccasin']['ETo'],
    #          label='Moccasin')
    # plt.xlim(dt.date(year=2019, month=1, day=1), dt.date(year=2024, month=1, day=1))
    # plt.ylim(0, 2)
    # plt.ylabel('AgriMet/Mesonet')
    # plt.grid()
    #
    # plt.subplot(212)
    # plt.hlines(0, dt.date(year=2019, month=1, day=1), dt.date(year=2024, month=1, day=1), 'k')
    # plt.plot(am_m['COVM'].index - dt.timedelta(days=15), (am_m['COVM']['ETo']) - mn_m['corvalli']['ETo'],
    #          label='Corvallis')
    # plt.plot(am_m['CRSM'].index - dt.timedelta(days=15), (am_m['CRSM']['ETo']) - mn_m['kalispel']['ETo'],
    #          label='Kalispell')
    # plt.plot(am_m['MWSM'].index - dt.timedelta(days=15), (am_m['MWSM']['ETo']) - mn_m['moccasin']['ETo'],
    #          label='Moccasin')
    # plt.xlim(dt.date(year=2019, month=1, day=1), dt.date(year=2024, month=1, day=1))
    # plt.ylim(-0.1, 0.2)
    # plt.ylabel('AgriMet minus Mesonet (in)')
    # plt.legend()
    # plt.grid()

    # plt.show()


def footprint_plots():
    """ Attempt to show relative size of different gridded meteorological data products. QGIS is better. """
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(3, 1, figsize=(3, 10), dpi=200)

    ax[0].set_title('Corvallis')
    ax[0].scatter(-114.083, 46.333, label='AgriMET')
    ax[0].scatter(-114.09, 46.33, label='Mesonet')
    ax[0].add_patch(Rectangle((-114.121, 46.296), 0.042, 0.042, ec='tab:green', fill=0, label='GridMET'))
    ax[0].add_patch(Rectangle((-114.125, 46.125), 0.25, 0.25, ec='tab:red', fill=0, label='ERA5'))
    ax[0].add_patch(Rectangle((-114.125, 46.25), 0.125, 0.125, ec='tab:purple', fill=0, label='NLDAS'))
    ax[0].legend(loc='lower right')
    ax[0].set_xlim(-114.15, -113.85)
    ax[0].set_ylim(46.1, 46.4)

    ax[1].set_title('Kalispell')
    ax[1].scatter(-114.128, 48.1875, label='AgriMET')
    ax[1].scatter(-114.14, 48.19, label='Mesonet')
    ax[1].add_patch(Rectangle((-114.163, 48.171), 0.042, 0.042, ec='tab:green', fill=0, label='GridMET'))
    ax[1].add_patch(Rectangle((-114.325, 48.125), 0.25, 0.25, ec='tab:red', fill=0, label='ERA5'))
    ax[1].add_patch(Rectangle((-114.25, 48.125), 0.125, 0.125, ec='tab:purple', fill=0, label='NLDAS'))
    ax[1].set_xlim(-114.35, -114.05)
    ax[1].set_ylim(48.1, 48.4)
    # ax[1].legend()

    ax[2].set_title('Moccasin')
    ax[2].scatter(-109.951, 47.0589, label='AgriMET')
    ax[2].scatter(-109.95, 47.06, label='Mesonet')
    ax[2].add_patch(Rectangle((-109.996, 47.004), 0.042, 0.042, ec='tab:green', fill=0, label='GridMET'))
    ax[2].add_patch(Rectangle((-110.125, 46.875), 0.25, 0.25, ec='tab:red', fill=0, label='ERA5'))
    ax[2].add_patch(Rectangle((-110, 47), 0.125, 0.125, ec='tab:purple', fill=0, label='NLDAS'))
    ax[2].set_xlim(-110.15, -109.85)
    ax[2].set_ylim(46.85, 47.15)
    # ax[2].legend()

    plt.tight_layout()

    # plt.figure(figsize=(3, 10), dpi=200)
    #
    # plt.subplot(311)
    # plt.title('Corvallis')
    # plt.scatter(46.333, -114.083, label='AgriMET')
    # plt.scatter(46.33, -114.09, label='Mesonet')
    #
    # plt.subplot(312)
    # plt.title('Kalispell')
    #
    # plt.subplot(313)
    # plt.title('Moccasin')
    #
    # plt.tight_layout()


def plot_combos(all_data, fit=True):

    # TODO: figure out which stations have zeros?

    xs = np.asarray([18.0, 42.0])

    plt.figure(figsize=(12, 12))
    plt.suptitle('Average Total Seasonal ETo (2021-2023, in)')

    # plt.subplot(334)
    # plt.hist(all_data[all_data['which'] == 'am']['stn'])

    plt.subplot(331)
    plt.xlabel('GridMET')
    plt.ylabel('Weather Station')
    plt.scatter(all_data[all_data['which'] == 'am']['gridmet'], all_data[all_data['which'] == 'am']['stn'],
                label='AgriMet, n={}'.format(all_data[all_data['which'] == 'am']['gridmet'].count()), zorder=3)
    plt.scatter(all_data[all_data['which'] == 'mn']['gridmet'], all_data[all_data['which'] == 'mn']['stn'],
                label='Mesonet, n={}'.format(all_data[all_data['which'] == 'mn']['gridmet'].count()), zorder=4)
    if fit:
        res1 = linregress(all_data['gridmet'], all_data['stn'])
        plt.plot(xs, res1.intercept + res1.slope*xs,
                 label='m:{:.2f} r^2:{:.2f}'.format(res1.slope, res1.rvalue**2), color='tab:green')
        # res2 = linregress(all_data[all_data['which'] == 'am']['gridmet'], all_data[all_data['which'] == 'am']['stn'])
        # plt.plot(xs, res2.intercept + res2.slope*xs,
        #          label='m:{:.2f} r^2:{:.2f}'.format(res2.slope, res2.rvalue**2), color='tab:blue')
        # res3 = linregress(all_data[all_data['which'] == 'mn']['gridmet'], all_data[all_data['which'] == 'mn']['stn'])
        # plt.plot(xs, res3.intercept + res3.slope * xs,
        #          label='m:{:.2f} r^2:{:.2f}'.format(res3.slope, res3.rvalue ** 2), color='tab:orange')
    plt.plot([18, 42], [18, 42], 'k', zorder=2, label='1:1 line')
    plt.legend()
    plt.grid(zorder=1)
    plt.xlim(18, 42)
    plt.ylim(18, 42)

    plt.subplot(332)
    plt.xlabel('NLDAS')
    plt.ylabel('Weather Station')
    plt.scatter(all_data[all_data['which'] == 'am']['nldas'], all_data[all_data['which'] == 'am']['stn'],
                label='n={}'.format(all_data[all_data['which'] == 'am']['nldas'].count()), zorder=3)
    plt.scatter(all_data[all_data['which'] == 'mn']['nldas'], all_data[all_data['which'] == 'mn']['stn'],
                label='n={}'.format(all_data[all_data['which'] == 'mn']['nldas'].count()), zorder=4)
    if fit:
        res1 = linregress(all_data['nldas'], all_data['stn'])
        plt.plot(xs, res1.intercept + res1.slope * xs,
                 label='m:{:.2f} r^2:{:.2f}'.format(res1.slope, res1.rvalue ** 2), color='tab:green')
        # res2 = linregress(all_data[all_data['which'] == 'am']['nldas'], all_data[all_data['which'] == 'am']['stn'])
        # plt.plot(xs, res2.intercept + res2.slope * xs,
        #          label='m:{:.2f} r^2:{:.2f}'.format(res2.slope, res2.rvalue ** 2), color='tab:blue')
        # res3 = linregress(all_data[all_data['which'] == 'mn']['nldas'], all_data[all_data['which'] == 'mn']['stn'])
        # plt.plot(xs, res3.intercept + res3.slope * xs,
        #          label='m:{:.2f} r^2:{:.2f}'.format(res3.slope, res3.rvalue ** 2), color='tab:orange')
    plt.plot([18, 42], [18, 42], 'k', zorder=2)
    plt.grid(zorder=1)
    plt.legend()
    plt.xlim(18, 42)
    plt.ylim(18, 42)

    plt.subplot(333)
    plt.xlabel('ERA5')
    plt.ylabel('Weather Station')
    plt.scatter(all_data[all_data['which'] == 'am']['era5'], all_data[all_data['which'] == 'am']['stn'],
                label='n={}'.format(all_data[all_data['which'] == 'am']['era5'].count()), zorder=2)
    plt.scatter(all_data[all_data['which'] == 'mn']['era5'], all_data[all_data['which'] == 'mn']['stn'],
                label='n={}'.format(all_data[all_data['which'] == 'mn']['era5'].count()), zorder=3)
    if fit:
        res1 = linregress(all_data['era5'], all_data['stn'])
        plt.plot(xs, res1.intercept + res1.slope * xs,
                 label='m:{:.2f} r^2:{:.2f}'.format(res1.slope, res1.rvalue ** 2), color='tab:green')
        # res2 = linregress(all_data[all_data['which'] == 'am']['era5'], all_data[all_data['which'] == 'am']['stn'])
        # plt.plot(xs, res2.intercept + res2.slope * xs,
        #          label='m:{:.2f} r^2:{:.2f}'.format(res2.slope, res2.rvalue ** 2), color='tab:blue')
        # res3 = linregress(all_data[all_data['which'] == 'mn']['era5'], all_data[all_data['which'] == 'mn']['stn'])
        # plt.plot(xs, res3.intercept + res3.slope * xs,
        #          label='m:{:.2f} r^2:{:.2f}'.format(res3.slope, res3.rvalue ** 2), color='tab:orange')
    plt.plot([18, 42], [18, 42], 'k', zorder=4)
    plt.grid(zorder=1)
    plt.legend()
    plt.xlim(18, 42)
    plt.ylim(18, 42)

    plt.subplot(335)
    plt.xlabel('NLDAS')
    plt.ylabel('GridMET')
    plt.scatter(all_data['nldas'], all_data['gridmet'], zorder=3, label='n={}'.format(all_data['gridmet'].count()))
    if fit:
        res1 = linregress(all_data['nldas'], all_data['gridmet'])
        plt.plot(xs, res1.intercept + res1.slope * xs,
                 label='m:{:.2f} r^2:{:.2f}'.format(res1.slope, res1.rvalue ** 2), color='tab:green')
    plt.plot([18, 42], [18, 42], 'k', zorder=2)
    plt.grid(zorder=1)
    plt.legend()
    plt.xlim(18, 42)
    plt.ylim(18, 42)

    plt.subplot(336)
    plt.xlabel('ERA5')
    plt.ylabel('GridMET')
    plt.scatter(all_data['era5'], all_data['gridmet'], zorder=3, label='n={}'.format(all_data['gridmet'].count()))
    if fit:
        res1 = linregress(all_data['era5'], all_data['gridmet'])
        plt.plot(xs, res1.intercept + res1.slope * xs,
                 label='m:{:.2f} r^2:{:.2f}'.format(res1.slope, res1.rvalue ** 2), color='tab:green')
    plt.plot([18, 42], [18, 42], 'k', zorder=2)
    plt.grid(zorder=1)
    plt.legend()
    plt.xlim(18, 42)
    plt.ylim(18, 42)

    plt.subplot(339)
    plt.xlabel('ERA5')
    plt.ylabel('NLDAS')
    plt.scatter(all_data['era5'], all_data['nldas'], zorder=3, label='n={}'.format(all_data['era5'].count()))
    if fit:
        res1 = linregress(all_data['era5'], all_data['nldas'])
        plt.plot(xs, res1.intercept + res1.slope * xs,
                 label='m:{:.2f} r^2:{:.2f}'.format(res1.slope, res1.rvalue ** 2), color='tab:green')
    plt.plot([18, 42], [18, 42], 'k', zorder=2)
    plt.grid(zorder=1)
    plt.legend()
    plt.xlim(18, 42)
    plt.ylim(18, 42)

    plt.tight_layout()


def download_all_data():
    am_md = load_stations()
    mn_md = stns_metadata()

    all_data = pd.DataFrame(columns=['lat', 'lon', 'stn', 'gridmet', 'nldas', 'era5'])
    r_index = pd.date_range('2021-01-01', '2023-12-31', freq='D')

    for stn, val in tqdm(am_md.items(), total=len(am_md)):
        # if val['properties']['state'] == 'MT':
        # if stn == 'bomt':
        if stn == 'bfam':
            all_data.at[stn, 'lat'] = val['geometry']['coordinates'][1]
            all_data.at[stn, 'lon'] = val['geometry']['coordinates'][0]
            # Average yearly Agrimet ETo value, 2021-2023
            df = agrimet_data(stn)
            df = df.truncate(dt.date(year=2021, month=1, day=1), dt.date(year=2023, month=12, day=31))
            df = df.reindex(r_index)
            df = df.interpolate()
            df['mask'] = [1 if d.month in range(4, 10) else 0 for d in df.index]
            df = df[df['mask'] == 1]
            df['year'] = [d.year for d in df.index]
            plt.figure()
            plt.plot(df['ETo'])
            print(stn)
            print(df.groupby(['year']).sum())
            print(df.groupby(['year']).sum().mean()['ETo'])
            # all_data.at[stn, 'stn'] = df.groupby(['year']).sum().mean()['ETo']

    for stn, val in tqdm(mn_md.items(), total=len(mn_md)):
        # if (dt.datetime.strptime(val['date_installed'], '%Y-%m-%d').year < 2021) and (stn != 'blmglsou'):
        # if stn == 'acebozem':
        if stn == 'wsrreeds':
            all_data.at[stn, 'lat'] = val['latitude']
            all_data.at[stn, 'lon'] = val['longitude']
            # Average yearly mesonet ETo value, 2021-2023
            df = mesonet_data(stn)
            df = df.truncate(dt.date(year=2021, month=1, day=1), dt.date(year=2023, month=12, day=31))
            df = df.reindex(r_index)
            df = df.interpolate()
            df['mask'] = [1 if d.month in range(4, 10) else 0 for d in df.index]
            df = df[df['mask'] == 1]
            df['year'] = [d.year for d in df.index]
            plt.figure()
            plt.plot(df['ETo'])
            print(stn)
            print(df.groupby(['year']).sum())
            print(df.groupby(['year']).sum().mean()['ETo'])
            # all_data.at[stn, 'stn'] = df.groupby(['year']).sum().mean()['ETo']

    # all_data.to_csv('C:/Users/CND571/Documents/Data/ETcompdata_20240820.csv')
    print(all_data)


def gridmet_match(con, fields, gridmet_points, fields_join):
    """
    Match each input field centroid with closest gridmet pixel centroid and update sqlite database table.

    This depends on running 'Raster Pixels to Points' on a WGS Gridmet raster,
     attributing GFID, lat, and lon in the attribute table, and saving to project crs: 5071.
     GFID is an arbitrary identifier e.g., @row_number. It further depends on projecting the
     rasters to EPSG:5071, using the project.sh bash script (or gis)

     The reason we're not just doing a zonal stat on correction surface for every object is that
     there may be many fields that only need data from one gridmet cell. This prevents us from downloading
     many redundant data sets. Looks like it often works out to roughly 1 gridmet data pull for every 10 fields.

     :con: sqlite database connection
     :fields: filepath, shapefile containing field geometries and unique field ids with county identifier
     :gridmet_points: filepath, shapefile of gridmet pixel centroids
     :fields_join: str, name of table in sqlite database containing field/gridmet lookup
     """

    def convert_to_wgs84(x, y):
        return pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:5071').transform(x, y)

    # fields = gpd.read_file(fields)
    # print('Finding field-gridmet joins for {} fields'.format(len(fields)))

    gridmet_pts = gpd.read_file(gridmet_points)
    # print(gridmet_pts.crs)

    existing_fields = pd.read_sql("SELECT DISTINCT gfid FROM {}".format(fields_join), con)
    # print(existing_fields)
    fields['gfid'] = np.nan
    fields['in_db'] = False
    # and run matching algorithm
    for i, field in fields.iterrows():
        geometry = Point(convert_to_wgs84(field['lat'], field['lon']))
        close_points = gridmet_pts.sindex.nearest(geometry)
        closest_fid = gridmet_pts.iloc[close_points[1]]['GFID'].iloc[0]
        fields.at[i, 'gfid'] = closest_fid
        # print('Matched {} to {}'.format(field['id'], closest_fid))
        if closest_fid in existing_fields.values:
            fields.at[i, 'in_db'] = True
        # g = GridMet('elev', lat=lat, lon=lon)
        # elev = g.get_point_elevation()
        # fields.at[i, 'elev_gm'] = elev
    print('Found {} gridmet target points for {} new fields'.format(len(fields['gfid'].unique()),
                                                                    fields.shape[0]))
    print()
    return fields


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


def livneh_stuff(stns, am_metadata=None):

    # TODO: add station data!
    # I've only been looking at GridMET and Livneh.

    livneh = pd.read_csv('C:/Users/CND571/Documents/Data/livneh_25_agrimet_allyears_20241122.csv',
                         index_col='Unnamed: 0')
    livneh.index = pd.to_datetime(livneh.index)
    # livneh = livneh.reindex(sorted(livneh.columns), axis=1)
    # print(livneh.filter(regex='^bfam').columns)
    # print(livneh)

    # # Analyzing daily swings in temperature (magnitude and timing, livneh data)
    # bigswings = []
    # smallswings = []
    # for i in old_stns:
    #     livneh['{}_flux'.format(i)] = livneh['{}_Tmax'.format(i)] - livneh['{}_Tmin'.format(i)]
    #     print("{} {:.2f} {:.2f}".format(i, livneh['{}_flux'.format(i)].max(),
    #                                     livneh['{}_flux'.format(i)].min()))
    #     print("{} {} {}".format(i, livneh.index[livneh['{}_flux'.format(i)].argmax()],
    #                             livneh.index[livneh['{}_flux'.format(i)].argmin()]))
    #     bigswings.append(pd.to_datetime(livneh.index[livneh['{}_flux'.format(i)].argmax()]))
    #     smallswings.append(pd.to_datetime(livneh.index[livneh['{}_flux'.format(i)].argmin()]))
    # bigswings_d = [x.month for x in bigswings]
    # smallswings_d = [x.month for x in smallswings]
    # plt.figure()
    # plt.subplot(121)
    # plt.title("Big Swings")
    # plt.hist(bigswings_d, bins=np.arange(14))
    # plt.subplot(122)
    # plt.title("Small Swings")
    # plt.hist(smallswings_d, bins=np.arange(14))

    # controlling plot features for different variables
    thing = 2  # <--- change this, the rest will follow.
    names = ['eto', 'Tmax', 'Tmin']
    units = ['mm', 'C', 'C']
    lims = [[0, 10], [-30, 40], [-40, 25]]
    bar_lims = [4500, 4250, 6500]
    steps = [0.5, 5, 5]
    clrs = ['tab:green', 'tab:red', 'tab:blue']
    clrmps = ['viridis', 'plasma', 'ocean']

    livneh1 = livneh.dropna()

    # plt.figure()
    # for i in range(24):
    #     plt.subplot(4, 6, i+1)
    #     plt.title(old_stns[i])
    #     plt.scatter(livneh['{}_{}_gm'.format(old_stns[i], names[thing])],
    #                 livneh['{}_{}'.format(old_stns[i], names[thing])], zorder=3, s=2, color=clrs[thing])
    #     # # Max and min
    #     # print(old_stns[i], 'gm', livneh['{}_{}_gm'.format(old_stns[i], thing)].max(),
    #     #       livneh['{}_{}_gm'.format(old_stns[i], thing)].min())
    #     # print(old_stns[i], 'ln', livneh['{}_{}'.format(old_stns[i], thing)].max(),
    #     #       livneh['{}_{}'.format(old_stns[i], thing)].min())
    #     # # Range
    #     # print(old_stns[i], 'gm', livneh['{}_{}_gm'.format(old_stns[i], thing)].max()
    #     #       - livneh['{}_{}_gm'.format(old_stns[i], thing)].min())
    #     # print(old_stns[i], 'ln', livneh['{}_{}'.format(old_stns[i], thing)].max()
    #     #       - livneh['{}_{}'.format(old_stns[i], thing)].min())
    #
    #     # Linear regression for slope and r^2
    #     res1 = linregress(livneh1['{}_{}_gm'.format(old_stns[i], names[thing])],
    #                       livneh1['{}_{}'.format(old_stns[i], names[thing])])
    #     plt.plot(lims[thing], res1.intercept + res1.slope * np.asarray(lims[thing]),
    #              label='m:{:.2f} r^2:{:.2f}'.format(res1.slope, res1.rvalue ** 2), color='tab:gray', zorder=5)
    #
    #     plt.grid(zorder=1)
    #     plt.legend()
    #     plt.xlim(lims[thing][0], lims[thing][1])
    #     plt.ylim(lims[thing][0], lims[thing][1])
    #     plt.plot(lims[thing], lims[thing], 'k', zorder=4)
    #
    #     if i % 6 == 0:
    #         plt.ylabel('livneh {} [{}]'.format(names[thing], units[thing]))
    #     if i//6 == 3:
    #         plt.xlabel('gridmet {} [{}]'.format(names[thing], units[thing]))

    # # Histograms of livneh data
    # plt.figure()
    # for i in range(24):
    #     plt.subplot(4, 6, i+1)
    #     plt.title(old_stns[i])
    #     plt.hist(livneh['{}_{}'.format(old_stns[i], names[thing])], zorder=3, color=clrs[thing],
    #              bins=np.arange(lims[thing][0], lims[thing][1] + 2*steps[thing], steps[thing]) - steps[thing]/2)
    #
    #     plt.grid(zorder=1)
    #     # plt.legend()
    #     plt.ylim(0, bar_lims[thing])
    #     # plt.ylim(lims[thing][0], lims[thing][1])
    #     # plt.plot(lims[thing], lims[thing], 'k', zorder=4)
    #
    #     # if i % 6 == 0:
    #     #     plt.ylabel('livneh {} [{}]'.format(names[thing], units[thing]))
    #     if i//6 == 3:
    #         plt.xlabel('Livneh {} [{}]'.format(names[thing], units[thing]))

    # # Histograms of livneh temp data
    # plt.figure()
    # for i in range(24):
    #     plt.subplot(4, 6, i + 1)
    #     plt.title(old_stns[i])
    #     plt.hist([livneh['{}_Tmin'.format(old_stns[i])], livneh['{}_Tmax'.format(old_stns[i])]],
    #              color=['tab:blue', 'tab:red'], label=['Tmin', 'Tmax'], zorder=3, histtype='stepfilled', alpha=0.3,
    #              bins=np.arange(lims[thing][0], lims[thing][1] + 2 * steps[thing], steps[thing]) - steps[
    #                  thing] / 2, density=True)
    #     plt.hist([livneh['{}_Tmin'.format(old_stns[i])], livneh['{}_Tmax'.format(old_stns[i])]],
    #              color=['tab:blue', 'tab:red'], label=['Tmin', 'Tmax'], zorder=4, histtype='step',
    #              bins=np.arange(lims[thing][0], lims[thing][1] + 2 * steps[thing], steps[thing]) - steps[
    #                  thing] / 2, density=True)
    #     plt.grid(zorder=1)
    #     plt.legend(loc='upper left')
    #     # plt.ylim(0, 6500)
    #     plt.ylim(0, 0.055)
    #
    #     if i % 6 == 0:
    #         plt.ylabel('Days')
    #     if i // 6 == 3:
    #         plt.xlabel('Livneh Temp [C]')

    # # Histograms of livneh temp data
    # plt.figure()
    # for i in range(24):
    #     plt.subplot(4, 6, i + 1)
    #     plt.title("{} ({})".format(am_metadata[old_stns[i]]['properties']['title'], old_stns[i].upper()),
    #               color=mpl.colormaps['viridis'](old_stns_d[old_stns[i]]/24))
    #     plt.hist([livneh['{}_Tmin'.format(old_stns[i])], livneh['{}_Tmax'.format(old_stns[i])]],
    #              color=['tab:blue', 'tab:red'], zorder=3, histtype='stepfilled', alpha=0.3,
    #              bins=np.arange(lims[thing][0], lims[thing][1] + 2 * steps[thing], steps[thing]) - steps[
    #                  thing] / 2, density=True)
    #     plt.hist([livneh['{}_Tmin'.format(old_stns[i])], livneh['{}_Tmax'.format(old_stns[i])]],
    #              color=['tab:blue', 'tab:red'], zorder=4, histtype='step',
    #              bins=np.arange(lims[thing][0], lims[thing][1] + 2 * steps[thing], steps[thing]) - steps[
    #                  thing] / 2, density=True)
    #     plt.grid(zorder=1)
    #     # plt.legend(loc='upper left')
    #     # plt.ylim(0, 6500)
    #     plt.ylim(0, 0.055)
    #
    #     if i == 5:
    #         from matplotlib.patches import Patch
    #         from matplotlib.colors import to_rgba
    #         legend_elements = [Patch(facecolor=to_rgba('tab:blue', 0.3), edgecolor='tab:blue', label='Tmin'),
    #                            Patch(facecolor=to_rgba('tab:red', 0.3), edgecolor='tab:red', label='Tmax')]
    #         plt.legend(handles=legend_elements, loc='upper left')
    #
    #     if i % 6 == 0:
    #         plt.ylabel('Frequency')
    #     if i // 6 == 3:
    #         plt.xlabel('Livneh Temp [C]')

    # Histograms of livneh temp data
    plt.figure()
    plt.suptitle("Climate Distributions, 1950-2013")
    for i in range(24):
        plt.subplot(4, 6, i + 1)
        stn = inv_stns_d[i]
        bins = np.arange(-40, 40 + 2 * 5, 5) - 5/5
        # plt.title("{} ({})".format(am_metadata[stn]['properties']['title'], stn.upper()),
        #           color=mpl.colormaps['viridis'](old_stns_d[old_stns[i]] / 24))
        plt.title("{} ({})".format(am_metadata[stn]['properties']['title'], stn.upper()))
        plt.hist([livneh['{}_Tmin'.format(stn)], livneh['{}_Tmax'.format(stn)]],
                 color=['tab:blue', 'tab:red'], zorder=3, histtype='stepfilled', alpha=0.3,
                 bins=bins, density=True)
        plt.hist([livneh['{}_Tmin'.format(stn)], livneh['{}_Tmax'.format(stn)]],
                 color=['tab:blue', 'tab:red'], zorder=4, histtype='step',
                 bins=bins, density=True)
        # plt.hist([livneh['{}_Tmin'.format(stn)], livneh['{}_Tmax'.format(stn)]],
        #          color=['tab:blue', 'tab:red'], zorder=3, histtype='stepfilled', alpha=0.3,
        #          bins=np.arange(lims[thing][0], lims[thing][1] + 2 * steps[thing], steps[thing]) - steps[
        #              thing] / 2, density=True)
        # plt.hist([livneh['{}_Tmin'.format(stn)], livneh['{}_Tmax'.format(stn)]],
        #          color=['tab:blue', 'tab:red'], zorder=4, histtype='step',
        #          bins=np.arange(lims[thing][0], lims[thing][1] + 2 * steps[thing], steps[thing]) - steps[
        #              thing] / 2, density=True)
        plt.grid(zorder=1)
        plt.ylim(0, 0.055)
        if i == 5:
            from matplotlib.patches import Patch
            from matplotlib.colors import to_rgba
            legend_elements = [Patch(facecolor=to_rgba('tab:blue', 0.3), edgecolor='tab:blue', label='Daily Minimum'),
                               Patch(facecolor=to_rgba('tab:red', 0.3), edgecolor='tab:red', label='Daily Maximum')]
            plt.legend(handles=legend_elements, loc='upper left')
        if i % 6 == 0:
            plt.ylabel('Frequency')
        if i // 6 == 3:
            plt.xlabel('Livneh Temperature [C]')

    # # Density plots
    # # plt.figure()
    # fig, axs = plt.subplots(4, 6)
    # print(np.shape(axs))
    # for i in range(24):
    #     # plt.subplot(4, 6, i + 1)
    #     x, y = i // 6, i % 6
    #     axs[x, y].set_title(old_stns[i])
    #     density_scatter(livneh1['{}_{}_gm'.format(old_stns[i], names[thing])],
    #                     livneh1['{}_{}'.format(old_stns[i], names[thing])], ax=axs[x, y],
    #                     zorder=3, s=2)
    #
    #     # # Linear regression for slope and r^2
    #     # res1 = linregress(livneh1['{}_{}_gm'.format(old_stns[i], names[thing])],
    #     #                   livneh1['{}_{}'.format(old_stns[i], names[thing])])
    #     # plt.plot(lims[thing], res1.intercept + res1.slope * np.asarray(lims[thing]),
    #     #          label='m:{:.2f} r^2:{:.2f}'.format(res1.slope, res1.rvalue ** 2), color='tab:gray', zorder=5)
    #
    #     axs[x, y].grid(zorder=1)
    #     # axs[x, y].legend()
    #     axs[x, y].set_xlim(lims[thing][0], lims[thing][1])
    #     axs[x, y].set_ylim(lims[thing][0], lims[thing][1])
    #     axs[x, y].plot(lims[thing], lims[thing], 'k', zorder=4)
    #
    #     if i % 6 == 0:
    #         axs[x, y].set_ylabel('livneh {} [{}]'.format(names[thing], units[thing]))
    #     if i // 6 == 3:
    #         axs[x, y].set_xlabel('gridmet {} [{}]'.format(names[thing], units[thing]))

    # # I'm not sure what this is telling me.
    # plt.figure(figsize=(10, 2))
    # plt.plot(livneh1['{}_{}'.format(stns[0], names[thing])], label='Livneh')
    # plt.plot(livneh1['{}_{}_gm'.format(stns[0], names[thing])], label='GridMET')
    # plt.grid()
    # plt.legend()

    # plt.figure()
    # for i in range(6):
    #     stn = stns[i + 0]  # Add a 0, 6, 12, or 18 to plot other sets of 6 stations
    #     plt.subplot(6, 1, i+1)
    #     plt.title(stn)
    #     plt.plot(livneh1['{}_{}'.format(stn, names[thing])], label='Livneh')
    #     plt.plot(livneh1['{}_{}_gm'.format(stn, names[thing])], label='GridMET')
    #     plt.grid()
    #     if i == 6:
    #         plt.legend()

    # # Looking at extreme temps
    # for i in old_stns:
    #     print("{} gm: max {:.2f} min {:.2f} dif {:.2f}"
    #           .format(i, livneh['{}_Tmax_gm'.format(i)].max(), livneh['{}_Tmin_gm'.format(i)].min(),
    #                   livneh['{}_Tmax_gm'.format(i)].max() - livneh['{}_Tmin_gm'.format(i)].min()))
    #     print("{} ln: max {:.2f} min {:.2f} dif {:.2f}"
    #           .format(i, livneh['{}_Tmax'.format(i)].max(), livneh['{}_Tmin'.format(i)].min(),
    #                   livneh['{}_Tmax'.format(i)].max() - livneh['{}_Tmin'.format(i)].min()))

    # if os.path.exists('F:/FileShare'):
    #     main_dir = 'F:/FileShare/openet_pilot'
    # else:
    #     main_dir = 'F:/openet_pilot'
    # conec = sqlite3.connect(os.path.join(main_dir, "opnt_analysis_03042024_Copy.db"))  # full project
    # gm_d = os.path.join(main_dir, 'gridmet')  # location of general gridmet files
    # gridmet_cent = os.path.join(gm_d, 'gridmet_centroids_MT.shp')
    # fields_db = 'field_data'
    #
    # twentyfive = pd.DataFrame()
    # twentyfive['id'] = old_stns
    # twentyfive['lat'] = [am_md[x]['geometry']['coordinates'][1] for x in old_stns]
    # twentyfive['lon'] = [am_md[x]['geometry']['coordinates'][0] for x in old_stns]
    #
    # twentyfive = gridmet_match(conec, twentyfive, gridmet_cent, fields_db)

    # print(twentyfive)
    # print()
    # print(livneh)

    # # Adding GridMET information to Livneh data
    # for i in tqdm(old_stns, total=len(old_stns)):
    #     gfid = twentyfive[twentyfive['id'] == i]['gfid'].values[0]
    #     grd = pd.read_sql("SELECT date, eto_mm, tmax_c, tmin_c FROM gridmet_ts WHERE gfid={}".format(gfid), conec)
    #     grd.index = grd['date']
    #     grd = grd[grd.index < '2014-01-01']
    #     livneh['{}_Tmax_gm'.format(i)] = grd['tmax_c']
    #     livneh['{}_Tmin_gm'.format(i)] = grd['tmin_c']
    #     livneh['{}_eto_gm'.format(i)] = grd['eto_mm']
    # livneh = livneh.reindex(sorted(livneh.columns), axis=1)
    # livneh.to_csv('C:/Users/CND571/Documents/Data/livneh_25_agrimet_allyears_20241122.csv')

    # # Adding ETo information for Livneh data
    # for i in tqdm(old_stns, total=len(old_stns)):
    #     # tmean = (livneh['{}_Tmax'.format(i)] + livneh['{}_Tmin'.format(i)]) / 2
    #     # vpd = calculate_vpd_temponly()
    #     lat = am_md[i]['geometry']['coordinates'][1]
    #     lon = am_md[i]['geometry']['coordinates'][0]
    #     elev = elevation_from_coordinate(lat, lon)
    #     # print('elev', elev)
    #     # print('lat', lat)
    #     livneh['{}_eto'.format(i)] = pm_fao56_ref(tmean=None, wind=livneh['{}_wind'.format(i)],
    #                                               tmax=livneh['{}_Tmax'.format(i)], tmin=livneh['{}_Tmin'.format(i)],
    #                                               elevation=elev, lat=(lat * np.pi/180))
    # print()
    # print(livneh)
    # livneh.to_csv('C:/Users/CND571/Documents/Data/livneh_25_agrimet_allyears_20241122.csv')


if __name__ == '__main__':
    """ Focus on 3 "collocated" Agrimet/Gridmet stations. """
    # am_stns = ['COVM', 'CRSM', 'MWSM']
    # mn_stns = ['corvalli', 'kalispel', 'moccasin']

    # old_stns = ['bfam', 'bftm', 'bozm', 'brgm', 'brtm', 'covm', 'crsm', 'dlnm', 'drlm', 'gfmt', 'glgm', 'hrlm',
    #             'hvmt', 'jvwm', 'lmmm', 'matm', 'mwsm', 'rbym', 'rdbm', 'sigm', 'svwm', 'tosm', 'trfm', 'umhm',
    #             'wssm']  # am
    # mwsm dropped because it does not have gridmet point in SID.
    old_stns = ['bfam', 'bftm', 'bozm', 'brgm', 'brtm', 'covm', 'crsm', 'dlnm', 'drlm', 'gfmt', 'glgm', 'hrlm', 'hvmt',
                'jvwm', 'lmmm', 'matm', 'rbym', 'rdbm', 'sigm', 'svwm', 'tosm', 'trfm', 'umhm', 'wssm']  # am
    old_stns_d = {'bfam': 1, 'bftm': 3, 'bozm': 22, 'brgm': 11, 'brtm': 10, 'covm': 12,
                  'crsm': 0, 'dlnm': 18, 'drlm': 13, 'gfmt': 9, 'glgm': 5, 'hrlm': 2,
                  'hvmt': 14, 'jvwm': 19, 'lmmm': 17, 'matm': 4, 'rbym': 20, 'rdbm': 6,
                  'sigm': 7, 'svwm': 23, 'tosm': 21, 'trfm': 8, 'umhm': 16, 'wssm': 15}  # am
    inv_stns_d = {v: k for k, v in old_stns_d.items()}

    old_stns_d1 = {'bfam': 1, 'bftm': 4, 'bozm': 21, 'brgm': 17, 'brtm': 23, 'covm': 18,
                   'crsm': 0, 'dlnm': 20, 'drlm': 19, 'gfmt': 13, 'glgm': 11, 'hrlm': 10,
                   'hvmt': 2, 'jvwm': 8, 'lmmm': 22, 'matm': 5, 'rbym': 14, 'rdbm': 6,
                   'sigm': 12, 'svwm': 15, 'tosm': 9, 'trfm': 7, 'umhm': 16, 'wssm': 3}  # am
    inv_stns_d1 = {v: k for k, v in old_stns_d1.items()}

    am_md = load_stations()
    # mn_md = stns_metadata()

    print(am_md[old_stns[0]])
    # print(am_md['abei'])
    # print(mn_md['aceabsar'])

    # plt.figure()
    # for i in old_stns:
    #     print(am_md[i]['properties']['title'])
    #     plt.text(am_md[i]['geometry']['coordinates'][0], am_md[i]['geometry']['coordinates'][1],
    #              am_md[i]['properties']['siteid'].upper())
    # plt.xlim(-116, -104)
    # plt.ylim(44, 49)
    #
    # # Testing station organization
    # plt.figure(figsize=(10, 4))
    # plt.subplot(121)
    # for i in old_stns:
    #     # print(am_md[i]['geometry']['coordinates'][0], am_md[i]['geometry']['coordinates'][1],
    #     #       am_md[i]['properties']['siteid'].upper())
    #     plt.text(am_md[i]['geometry']['coordinates'][0], am_md[i]['geometry']['coordinates'][1],
    #              am_md[i]['properties']['siteid'].upper(), color=mpl.colormaps['viridis'](old_stns_d[i]/24))
    # plt.xlim(-116, -104)
    # plt.ylim(44, 49)
    # plt.subplot(122)
    # for i in range(24):
    #     # print(i % 6, i // 6, inv_stns_d[i])
    #     plt.text(i % 6, i // 6, "{} {}".format(i, inv_stns_d[i].upper()), color=mpl.colormaps['viridis'](i/24))
    # plt.xlim(0, 5)
    # plt.ylim(3, 0)
    #
    # plt.figure(figsize=(10, 4))
    # plt.subplot(121)
    # for i in old_stns:
    #     # print(am_md[i]['geometry']['coordinates'][0], am_md[i]['geometry']['coordinates'][1],
    #     #       am_md[i]['properties']['siteid'].upper())
    #     plt.text(am_md[i]['geometry']['coordinates'][0], am_md[i]['geometry']['coordinates'][1],
    #              am_md[i]['properties']['siteid'].upper(), color=mpl.colormaps['viridis'](old_stns_d1[i] / 24))
    # plt.xlim(-116, -104)
    # plt.ylim(44, 49)
    # plt.subplot(122)
    # for i in range(24):
    #     # print(i % 6, i // 6, inv_stns_d1[i])
    #     plt.text(i % 6, i // 6, "{} {}".format(i, inv_stns_d1[i].upper()), color=mpl.colormaps['viridis'](i / 24))
    # plt.xlim(0, 5)
    # plt.ylim(3, 0)

    livneh_stuff(old_stns, am_md)

    # # This still seems like a good set to use for initial livneh comparisons...
    # print(am_md['covm'])  # 1984
    # print(am_md['crsm'])  # 1988
    # print(am_md['mwsm'])  # 2001

    # for i in am_md.keys():
    #     if am_md[i]['properties']['state'] == 'MT':
    #         print(am_md[i]['properties']['install'])  # A lot of data missing?

    # download_all_data()
    # 2 stations are missing some data.

    # all_data.to_csv('C:/Users/CND571/Documents/Data/ETcompdata_20240820.csv')

    # all_data = pd.read_csv('C:/Users/CND571/Documents/Data/ETcompdata_20240822.csv', index_col='Unnamed: 0')
    #
    # which = []
    # for i in all_data.index:
    #     if len(i) > 4:
    #         which.append('mn')
    #     else:
    #         which.append('am')
    #
    # all_data['which'] = which
    #
    # r_index = pd.date_range('2021-01-01', '2023-12-31', freq='D')
    #
    # for stn, val in tqdm(mn_md.items(), total=len(mn_md)):
    #     if (dt.datetime.strptime(val['date_installed'], '%Y-%m-%d').year < 2021) and (stn != 'blmglsou'):
    #         all_data.at[stn, 'lat'] = val['latitude']
    #         all_data.at[stn, 'lon'] = val['longitude']
    #         # Average yearly mesonet ETo value, 2021-2023
    #         df = mesonet_data(stn)
    #         df = df.truncate(dt.date(year=2021, month=1, day=1), dt.date(year=2023, month=12, day=31))
    #         df = df.reindex(r_index)
    #         df = df.interpolate()
    #         df['mask'] = [1 if d.month in range(4, 10) else 0 for d in df.index]
    #         df = df[df['mask'] == 1]
    #         df['year'] = [d.year for d in df.index]
    #         # print(df.groupby(['year']).sum())
    #         # print(df.groupby(['year']).sum().mean()['ETo'])
    #         all_data.at[stn, 'stn'] = df.groupby(['year']).sum().mean()['ETo']
    #
    # all_data['stn'] = pd.to_numeric(all_data['stn'])
    # print(all_data['stn'])
    # print(all_data.columns)
    #
    # print(all_data[all_data['stn'] == 0])

    # era5_all_data(all_data)
    # nldas_all_data(all_data)
    # all_data.to_csv('C:/Users/CND571/Documents/Data/ETcompdata_20240820.csv')

    # print(all_data)

    # nldas_data = nldas_data()
    # print(nldas_data)
    #
    # era5_data = era5_data()
    # print(era5_data)

    # am_data = 0
    # mn_data = 0
    # am_data_m = 0
    # mn_data_m = 0

    # # Station data
    # am_data = {'COVM': '', 'CRSM': '', 'MWSM': ''}
    # mn_data = {'corvalli': '', 'kalispel': '', 'moccasin': ''}
    # am_miss = {'COVM': '', 'CRSM': '', 'MWSM': ''}
    # mn_miss = {'corvalli': '', 'kalispel': '', 'moccasin': ''}
    # am_data_m = {'COVM': '', 'CRSM': '', 'MWSM': ''}
    # mn_data_m = {'corvalli': '', 'kalispel': '', 'moccasin': ''}
    # for i in range(3):
    #     am_data[am_stns[i]] = agrimet_data(am_stns[i])
    #     mn_data[mn_stns[i]] = mesonet_data(mn_stns[i])
    #     # Average daily ET for each month
    #     am_data_m[am_stns[i]] = am_data[am_stns[i]].resample('ME').mean()
    #     mn_data_m[mn_stns[i]] = mn_data[mn_stns[i]].resample('ME').mean()
    #     # Total monthly ET
    #     # am_num_days = am_data[am_stns[i]].resample('ME').index.daysinmonth
    #     # mn_num_days = mn_data[mn_stns[i]].resample('ME').index.daysinmonth
    #     # am_data_m[am_stns[i]] = am_num_days * am_data_m[am_stns[i]]
    #     # mn_data_m[am_stns[i]] = mn_num_days * mn_data_m[am_stns[i]]
    #     # am_data_m[am_stns[i]] = am_data_m[am_stns[i]].index.daysinmonth * am_data_m[am_stns[i]]
    #     # mn_data_m[am_stns[i]] = mn_data_m[am_stns[i]].index.daysinmonth * mn_data_m[am_stns[i]]
    #     # Missing data counts
    #     am_miss[am_stns[i]] = am_data[am_stns[i]].resample('ME')['ETo'].count()
    #     mn_miss[mn_stns[i]] = mn_data[mn_stns[i]].resample('ME').count()
    #     # Mask results with less than 25 days of data in a month
    #     am_data_m[am_stns[i]] = am_data_m[am_stns[i]].mask(am_miss[am_stns[i]] < 25)
    #     mn_data_m[mn_stns[i]] = mn_data_m[mn_stns[i]].mask(mn_miss[mn_stns[i]] < 25)
    # # print(am_data)
    # # print(am_data['COVM'])
    # # print(mn_data['corvalli'])

    # # GridMET!
    # if os.path.exists('F:/FileShare'):
    #     main_dir = 'F:/FileShare/openet_pilot'
    # else:
    #     main_dir = 'F:/openet_pilot'
    # conec = sqlite3.connect(os.path.join(main_dir, "opnt_analysis_03042024_Copy.db"))  # full project
    #
    # # am_gm = [102821, 40450, 79358]  # GFIDs associated with 3 AgriMet stations
    #
    # def convert_to_wgs84(x, y):
    #     return pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:5071').transform(x, y)
    #
    # gridmet_cent = 'C:/Users/CND571/Documents/Data/gridmet/gridmet_centroids_MT.shp'
    # gridmet_pts = gpd.read_file(gridmet_cent)
    # gfids = pd.read_sql("SELECT DISTINCT gfid FROM gridmet_ts", conec)
    # # print(gfids.values.flatten())
    #
    # inc = 0
    # target_gfids = []
    # to_fix = []
    # for i in tqdm(all_data.index, total=len(all_data)):
    #     grd_m = pd.DataFrame()
    #     eto = '{}_ETo'.format(i)
    #     etr = '{}_ETr'.format(i)
    #     lon = all_data.at[i, 'lon']
    #     lat = all_data.at[i, 'lat']
    #
    #     # lat, lon = convert_to_wgs84(lat, lon)
    #     # print(Point(lat, lon))
    #     # close_points = gridmet_pts.sindex.nearest(Point(lon, lat))
    #     close_points = gridmet_pts.to_crs('EPSG:4326').sindex.nearest(Point(lon, lat))
    #     # print(close_points)
    #     closest_fid = gridmet_pts.iloc[close_points[1]]['GFID'].iloc[0]
    #     gfid = closest_fid
    #     # print(gfid)
    #
    #     if gfid in gfids.values.flatten():
    #         print("got it!")
    #     else:
    #         target_gfids.append(gfid)
    #         to_fix.append(i)
    #         print(i, "Sorry, this data hasn't been downloaded yet")
    #         inc += 1
    #
    #     # if gfid in gfids.values.flatten():
    #     #     grd = pd.read_sql("SELECT date, eto_mm, etr_mm FROM gridmet_ts WHERE gfid={}".format(gfid), conec)
    #     #     grd.index = pd.to_datetime(grd['date'])
    #     #     grd_m[eto] = grd['eto_mm'].resample('ME').sum() / 25.4  # mm to in
    #     #     grd_m[etr] = grd['etr_mm'].resample('ME').sum() / 25.4  # GridMET always has data, so this is fine.
    #     #
    #     #     df = grd_m
    #     #     df = df.truncate(dt.date(year=2021, month=1, day=1), dt.date(year=2023, month=12, day=31))
    #     #     df['mask'] = [1 if d.month in range(4, 10) else 0 for d in df.index]
    #     #     df = df[df['mask'] == 1]
    #     #     df['year'] = [d.year for d in df.index]
    #     #     print(df.groupby(['year']).sum())
    #     #     print(df.groupby(['year']).sum().mean()[eto])
    #     #     all_data.at[i, 'gridmet'] = df.groupby(['year']).sum().mean()[eto]
    #     # else:
    #     #     print(i, "Sorry, this data hasn't been downloaded yet")
    #     #     inc += 1
    # # print("{}/109 stations ({:.1f}%) missing gridmet data".format(inc, 100*inc/109))
    #
    # # print('target_gfids')
    # # print(target_gfids)
    #
    # # target_gfids = [77973.0, 29391.0, 100245.0, 103016.0, 57146.0, 138886.0, 137524.0, 127805.0, 144645.0, 140368.0,
    # #                 136324.0, 83641.0, 29536.0, 141873.0, 18396.0, 50292.0, 98895.0, 107181.0, 94634.0, 64137.0,
    # #                 85015.0, 145923.0, 123845.0, 54398.0, 30852.0, 50239.0, 112560.0, 76586.0, 71003.0, 25441.0,
    # #                 35138.0, 23911.0, 21180.0, 34962.0, 21107.0, 17097.0, 77973.0, 119573.0, 57073.0, 55686.0,
    # #                 112640.0, 119565.0, 91898.0, 66880.0, 94548.0, 30741.0, 130657.0, 126492.0]
    #
    # # If there is any new data to fetch,
    # if len(target_gfids) > 0:
    #     # run data fetching algorithm
    #     gridmet_pts.index = gridmet_pts['GFID']
    #
    #     # Loading correction surfaces
    #     print('Loading correction factors...')
    #     rasters = []
    #     for v in ['eto', 'etr']:
    #         [rasters.append('C:/Users/CND571/Documents/Data/gridmet/correction_surfaces_aea/gridmet_corrected_{}_{}.tif'
    #                         .format(v, m)) for m in range(1, 13)]
    #
    #     # Getting correction factors for the required gridmet locations
    #     gridmet_targets = {}
    #     for j in target_gfids:
    #         gridmet_targets[j] = {str(m): {} for m in range(1, 13)}
    #         geo = gridmet_pts.at[j, 'geometry']
    #         gdf = gpd.GeoDataFrame({'geometry': [geo]})
    #         for r in rasters:
    #             splt = r.split('_')
    #             _var, month = splt[-2], splt[-1].replace('.tif', '')
    #             stats = zonal_stats(gdf, r, stats=['mean'])[0]['mean']
    #             gridmet_targets[j][month].update({_var: stats})
    #
    #     len_ = len(gridmet_targets)
    #     print('Get gridmet for {} target points'.format(len_))
    #     gridmet_pts.index = gridmet_pts['GFID']
    #
    #     # Getting the gridmet data
    #     i = 0
    #     for j in tqdm(target_gfids, total=len(target_gfids)):
    #         df = pd.DataFrame()
    #         r = gridmet_pts.loc[j]
    #         thredds_var = 'pet'
    #         cols = {'nc': 'agg_met_pet_1979_CurrentYear_CONUS',
    #                 'var': 'daily_mean_reference_evapotranspiration_grass',
    #                 'col': 'eto_mm'}
    #         variable = cols['col']
    #         lat, lon = r['lat'], r['lon']
    #         g = GridMet(thredds_var, start='2021-01-01', end='2023-12-31', lat=lat, lon=lon)
    #         s = g.get_point_timeseries()
    #         df[variable] = s[thredds_var]
    #
    #         corr_val = gridmet_targets[j]
    #         variable = 'eto_mm'
    #         for month in range(1, 13):
    #             corr_factor = corr_val[str(month)]['eto']
    #             idx = [i for i in df.index if i.month == month]
    #             df.loc[idx, '{}_uncorr'.format(variable)] = df.loc[idx, variable]
    #             df.loc[idx, variable] = df.loc[idx, '{}_uncorr'.format(variable)] * corr_factor
    #
    #         df = df.truncate(dt.date(year=2021, month=1, day=1), dt.date(year=2023, month=12, day=31))
    #         df['mask'] = [1 if d.month in range(4, 10) else 0 for d in df.index]
    #         df = df[df['mask'] == 1]
    #         df['year'] = [d.year for d in df.index]
    #         # print(df.groupby(['year']).sum())
    #         # print(df.groupby(['year']).sum().mean()['eto_mm'] / 25.4)
    #         all_data.at[to_fix[i], 'gridmet'] = df.groupby(['year']).sum().mean()['eto_mm'] / 25.4
    #         i += 1
    #
    # print(all_data['gridmet'])
    #
    # all_data.to_csv('C:/Users/CND571/Documents/Data/ETcompdata_20240822.csv')
    # # print(all_data[['gridmet', 'era5']])

    # am_mn_comp_plots(am_data, mn_data, am_data_m, mn_data_m, era5_data, grd_m, nldas_data)

    # # Calculate total growing season averages
    # r_index = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    # for i in [am_data_m, mn_data_m]:
    #     for j in i.keys():
    #         df = i[j]
    #         df = df.truncate(dt.date(year=2020, month=1, day=1), dt.date(year=2023, month=12, day=31))
    #         df = df.reindex(r_index)
    #         df = df.interpolate()
    #         df['mask'] = [1 if d.month in range(4, 10) else 0 for d in df.index]
    #         df = df[df['mask'] == 1]
    #         df['year'] = [d.year for d in df.index]
    #         print(j)
    #         print(df.groupby(['year']).sum())
    #         print(df.groupby(['year']).sum().mean())
    #         # print(df.mean())
    # for i in [era5_data, grd_m, nldas_data]:
    #     i = i.truncate(dt.date(year=2020, month=1, day=1), dt.date(year=2023, month=12, day=31))
    #     # i = i.reindex(r_index)
    #     # i = i.interpolate()
    #     i['mask'] = [1 if d.month in range(4, 10) else 0 for d in i.index]
    #     i = i[i['mask'] == 1]
    #     i['year'] = [d.year for d in i.index]
    #     print(i.groupby(['year']).sum())
    #     print(i.groupby(['year']).sum().mean())
    #     # print(i.mean())

    # all_data = all_data.drop(['bomt', 'acebozem'])  # data just not available. I have no clue why.
    # # Most of it will plot on the online dashboard, but it's just zeros when I try to call it from the API.
    # # It's apparently not passing QA/QC checks. Why?

    # plot_combos(all_data, fit=False)

    # print(all_data.index)

    plt.show()
# ========================= EOF ====================================================================
