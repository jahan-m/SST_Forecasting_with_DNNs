import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
import shared_modules.global_settings as gs     # pylint: disable=import-error


# location_name = 'Dalian'
# years_from_to = [1970, 2020]

# location_name = 'Naze'
# years_from_to = [1973, 2020]

location_name = 'Blair'
years_from_to = [1973, 2020]



dataset_folder_path = 'C:/Users/MLaptop/Desktop/Raw Datasets/Downloaded/+ Air Temp Daily 1957-2020 from NCDC NOAA'
dataset_file_name = location_name + ', ' + str(years_from_to[0]) + ' to ' + str(years_from_to[1]) + '.csv'
dataset_file_path = os.path.join(dataset_folder_path, dataset_file_name)

dataset_column_types = {'STATION':str, 'NAME':str, 'DATE':str, 'TAVG':float}
dataset_columns = ['STATION', 'NAME', 'DATE', 'TAVG']
dataset_date = ['DATE']

dataset_sampling_period_str = '1d'
dataset_sampling_period_dt = gs.str_timedelta[dataset_sampling_period_str]
