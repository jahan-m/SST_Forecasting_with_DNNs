import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
import shared_modules.global_settings as gs     # pylint: disable=import-error


# merging_files = {'water': 'Bohai_2007to2020_1d.csv', 'air': 'Dalian_1970to2020_1d.csv'}
# dataset_name = 'Boh-Dal_2007to2020'
# merging_files = {'water': 'Bohai_1998to2020_1d.csv', 'air': 'Dalian_1970to2020_1d.csv'}
# dataset_name = 'Boh-Dal_1998to2020'

# merging_files = {'water': 'PHL_2007to2020_1d.csv', 'air': 'Naze_1973to2020_1d.csv'}
# dataset_name = 'PHL-Naz_2007to2020'
# merging_files = {'water': 'PHL_1998to2020_1d.csv', 'air': 'Naze_1973to2020_1d.csv'}
# dataset_name = 'PHL-Naz_1998to2020'

# merging_files = {'water': 'BoB_2007to2020_1d.csv', 'air': 'Blair_1973to2020_1d.csv'}
# dataset_name = 'BoB-BLR_2007to2020'
merging_files = {'water': 'BoB_1998to2020_1d.csv', 'air': 'Blair_1973to2020_1d.csv'}
dataset_name = 'BoB-BLR_1998to2020'



dataset_sampling_period_str = '1d'
other_requested_sampling_period_str = ['1w', '1iM']
dataset_sampling_period_dt = gs.str_timedelta[dataset_sampling_period_str]

nan_strategy = 'linear_interp'
outlier_win_size = 3
outlier_std_factor = 1.5
outlier_IQR_factor = 1.5

downsampling_aggregation = 'mean'
