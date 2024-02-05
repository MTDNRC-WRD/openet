import requests
import urllib

import ee

url = r'https://epqs.nationalmap.gov/v1/json?'

# # This one is giving problems 02/02/2024
# def elevation_from_coordinate(lat, lon):
#     params = {
#         'output': 'json',
#         'x': lon,
#         'y': lat,
#         'units': 'Meters'
#     }
#
#     result = requests.get((url + urllib.parse.urlencode(params)))
#     elev = float(result.json()['value'])
#     return elev
#


def elevation_from_coordinate(lat, lon):
    """ Use Earth Engine API to get elevation data (in meters) from decimal degree coordinates.
    Dataset referenced is NASA SRTM Digital Elevation 30m. """

    # should I initialize ee here, or somewhere else?
    ee.Authenticate()
    ee.Initialize(project='ee-hehaugen')

    img = ee.Image("USGS/SRTMGL1_003")
    point = ee.Geometry.Point(lon, lat)
    sample = img.sample(point).getInfo()
    elev = sample['features'][0]['properties']['elevation']
    # print(lon, lat, elev)
    return elev


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
