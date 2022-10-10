#!/usr/bin/env python
import cdsapi
c = cdsapi.Client()
c.retrieve('reanalysis-era5-complete', {
     'class': 'ea',
     'date': '2000-01-01/to/2000-01-10',
     'expver': '1',
     'levelist': '1/to/137',
     'levtype': 'ml',
     'param': '155',
     'stream': 'oper',
     'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
     'type': 'an',
     'format': 'grib'
}, 'data_train.grib')