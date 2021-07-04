import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
import shared_modules.global_settings as gs     # pylint: disable=import-error


# dataset_name = 'Boh-Dal_2007to2020'
# dataset_name = 'Boh-Dal_1998to2020'

# dataset_name = 'PHL-Naz_2007to2020'
# dataset_name = 'PHL-Naz_1998to2020'

# dataset_name = 'BoB-BLR_2007to2020'
dataset_name = 'BoB-BLR_1998to2020'



dataset_sampling_periods_str = ['1d', '1w', '1iM']

dataset_sampling_periods_dt = [gs.str_timedelta[sp_str] for sp_str in dataset_sampling_periods_str]
test_data_duration_str = '1y'
