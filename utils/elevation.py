import urllib
import requests
# import ee

url = r'https://epqs.nationalmap.gov/v1/json?'


# Where should this function live? - Added it to chmdata.met_utils.py
def elevation_from_coordinate(lat, lon):
    """ Returns elevation in meters from USGS National Map services given decimal degree coordinates. """
    params = {
        'output': 'json',
        'x': lon,
        'y': lat,
        'units': 'Meters'
    }

    result = requests.get((url + urllib.parse.urlencode(params)))
    elev = float(result.json()['value'])
    return elev


# # backup function, above was not working for a short period.
# def elevation_from_coordinate(lat, lon, project='ee-hehaugen'):
#     """ Use Earth Engine API to get elevation data (in meters) from decimal degree coordinates.
#     Dataset referenced is NASA SRTM Digital Elevation 30m. """
#
#     ee.Authenticate()
#     ee.Initialize(project=project)
#
#     img = ee.Image("USGS/SRTMGL1_003")
#     point = ee.Geometry.Point(lon, lat)
#     sample = img.sample(point).getInfo()
#     elev = sample['features'][0]['properties']['elevation']
#     # print(lon, lat, elev)
#     return elev


if __name__ == '__main__':
    # print(elevation_from_coordinate(lat=46.5889579, lon=-112.0152353))
    pass
# ========================= EOF ====================================================================
