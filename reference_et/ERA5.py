
import xarray
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
# import cdsapi
#
# c = cdsapi.Client()
#
# c.retrieve(
#     'reanalysis-era5-single-levels-monthly-means',
#     {
#         'format': 'grib',
#         'product_type': 'monthly_averaged_reanalysis',
#         'variable': [
#             'evaporation', 'potential_evaporation',
#         ],
#         'year': [
#             '1985', '1986', '1987',
#             '1988', '1989', '1990',
#             '1991', '1992', '1993',
#             '1994', '1995', '1996',
#             '1997', '1998', '1999',
#             '2000', '2001', '2002',
#             '2003', '2004', '2005',
#             '2006', '2007', '2008',
#             '2009', '2010', '2011',
#             '2012', '2013', '2014',
#             '2015', '2016', '2017',
#             '2018', '2019', '2020',
#             '2021', '2022', '2023',
#         ],
#         'month': [
#             '01', '02', '03',
#             '04', '05', '06',
#             '07', '08', '09',
#             '10', '11', '12',
#         ],
#         'time': '00:00',
#         'area': [
#             50, -117, 44,
#             -104,
#         ],
#     },
#     'download.grib')


if __name__ == '__main__':
    ds = xarray.load_dataset("C:/Users/CND571/Downloads/adaptor.mars.internal-1719011409.8211007-13587-9-ab3d96eb-309e-4bfd-bafc-c6f71672405e.grib", engine="cfgrib")

    print(ds)

    # # This appears to be working.
    # with open('C:/Users/CND571/Downloads/subset_NLDAS_NOAH0125_M_002_20240626_202619_.txt') as file:
    #     lines = [line.rstrip() for line in file]
    # # with open('C:/Users/CND571/Downloads/subset_NLDAS_NOAH0125_M_2.0_20240626_201918_.txt') as file:
    # #     lines = [line.rstrip() for line in file]
    # print(lines[1])

    # import requests
    # # Set the URL string to point to a specific data URL. Some generic examples are:
    # #   https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/path/to/granule.nc4
    # URL = lines[1]
    # # Set the FILENAME string to the data file name, the LABEL keyword value, or any customized name.
    # FILENAME = 'test'
    # result = requests.get(URL)
    # try:
    #     result.raise_for_status()
    #     f = open(FILENAME, 'wb')
    #     f.write(result.content)
    #     f.close()
    #     print('contents of URL written to ' + FILENAME)
    # except:
    #     print('requests.get() returned an error code ' + str(result.status_code))

    # # This appears to not be working.
    # ds = xarray.open_dataset(lines[1])
    # print(ds)

    # # Plotting states not working.

    fig = plt.figure(figsize=(12, 10))
    # ax = plt.axes(projection=ccrs.Robinson())
    # ax.borders(resolution="10m")
    # plot = ds.e[6].plot(
    #     cmap=plt.cm.plasma.r, transform=ccrs.Robinson(), cbar_kwargs={"shrink": 0.6}
    # )
    # ax.add_feature(cartopy.feature.BORDERS)
    # ax.add_feature(cartopy.feature.STATES)
    plot = ds.e[0].plot()
    # plt.plot()

    # plt.title("ERA5 - 2m temperature British Isles March 2019")
    plt.show()

# ========================= EOF ====================================================================
