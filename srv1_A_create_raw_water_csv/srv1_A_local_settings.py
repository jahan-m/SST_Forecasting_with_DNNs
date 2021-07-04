import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
import shared_modules.global_settings as gs     # pylint: disable=import-error


# location_name = 'Bohai'           # Bohai Sea
# # years_from_to = [2007, 2020]
# years_from_to = [1998, 2020]

# location_name = 'PHL'             # Philippine Sea
# years_from_to = [2007, 2020]
# years_from_to = [1998, 2020]

location_name = 'BoB'             # Bay of Bengal
# years_from_to = [2007, 2020]
years_from_to = [1998, 2020]



nc_folder_path = 'PATH/TO/THE/DOWNLOADED/Water Temp Daily 1900-2020 from PSL NOAA'

area_aggregation = 'mean'

dataset_sampling_period_str = '1d'
dataset_sampling_period_dt = gs.str_timedelta[dataset_sampling_period_str]
dataset_spatial_res = 0.25
dataset_lat_from_to = np.array([-89.875, 89.875])
# (89.875 - -89.875)/0.25 + 1 = 720
dataset_lon_from_to = np.array([0.125, 359.875])
# (359.875 - 0.125)/0.25 + 1 = 1440
if location_name == 'Bohai':
    lat_from_to = np.array([36.5, 40.5])     # Should be North (0 to +90)
    lon_from_to = np.array([117.5, 121.5])  # Should be East (0 to +360)
elif location_name == 'PHL':
    lat_from_to = np.array([20, 30])     # Should be North (0 to +90)
    lon_from_to = np.array([120, 131.5])  # Should be East (0 to +360)
elif location_name == 'BoB':
    lat_from_to = np.array([12, 18])     # Should be North (0 to +90)
    lon_from_to = np.array([87, 93])  # Should be East (0 to +360)
else:
    raise Exception(f'"{location_name}" is not a known location name!')
lat_ind_from_to = (  (lat_from_to - dataset_lat_from_to[0]) / dataset_spatial_res  ).astype(int)
lon_ind_from_to = (  (lon_from_to - dataset_lon_from_to[0]) / dataset_spatial_res  ).astype(int)
del(lat_from_to)
del(lon_from_to)

def nc_file_name_with_ext(year):
    return 'sst.day.mean.' + str(year) + '.nc'
