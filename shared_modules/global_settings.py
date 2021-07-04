import datetime as dt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
import shared_modules.columns_info as columnsInfo       # pylint: disable=import-error


datasets_folder = 'datasets/'

ci = columnsInfo.ColumnsInfo()

decomp_method = 'simple_seasonal' # None  OR  'simple_seasonal'
SS_1YEAR    = 'SS-1Y'
SS_ResTrend = 'SS-RT'
SS_YEAR     = 'SS-Y'

str_timedelta = {'0':None,                      '30m':dt.timedelta(minutes=30),     '45m':dt.timedelta(minutes=45),
                 '1h':dt.timedelta(hours=1),    '2h':dt.timedelta(hours=2),         '3h':dt.timedelta(hours=3),
                 '6h':dt.timedelta(hours=6),    '8h':dt.timedelta(hours=8),         '12h':dt.timedelta(hours=12),
                 '1d':dt.timedelta(days=1),     '2d':dt.timedelta(days=2),          '3d':dt.timedelta(days=3),
                 '4d':dt.timedelta(days=4),     '5d':dt.timedelta(days=5),
                 '1w':dt.timedelta(weeks=1),    '2w':dt.timedelta(weeks=2),         '3w':dt.timedelta(weeks=3),
                 '4w':dt.timedelta(weeks=4),
                 '1iM':dt.timedelta(seconds=2629800),   # 12*idealM = real_year (not 365 days, but one precise circulation around sun)
                 '5w':dt.timedelta(weeks=5),
                 '6w':dt.timedelta(weeks=6),
                 '8w':dt.timedelta(weeks=8),
                 '2iM':dt.timedelta(seconds=2*2629800),
                 '4iM':dt.timedelta(seconds=4*2629800),
                 '5iM':dt.timedelta(seconds=5*2629800),
                 '6iM':dt.timedelta(seconds=6*2629800),
                 '7iM':dt.timedelta(seconds=7*2629800),
                 '9iM':dt.timedelta(seconds=9*2629800),
                 '1y':dt.timedelta(days=365),   '2y':dt.timedelta(days=730),        '3y':dt.timedelta(days=1095),
                 '4y':dt.timedelta(days=1460)
                 }

