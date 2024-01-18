import requests
import urllib

import ee

# url = r'https://nationalmap.gov/epqs/pqs.php?' ## raises error 301, "moved permanently"
# url = r'https://apps.nationalmap.gov/epqs/' ## this one is 403 Forbidden
url = r'https://apps.nationalmap.gov/epqs/pqs.php?' ## this one is also 403 Forbidden

def elevation_from_coordinate(lat, lon):
    params = {
        'output': 'json',
        'x': lon,
        'y': lat,
        'units': 'Meters'
    }

    result = requests.get((url + urllib.parse.urlencode(params)))
    print(result.status_code)
    elev = result.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation']
    return elev

def elevation_from_coordinate_ee(lat, lon):
    """ Use Earth Engine API to get elevation data (in meters) from decimal degree coordinates.
    Dataset referenced is NASA SRTM Digital Elevation 30m. """

    ## should I initialize ee here, or somewhere else?
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
