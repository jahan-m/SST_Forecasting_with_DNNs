import numpy as np
import datetime as dt
from tensorflow import keras                        # pylint: disable=import-error
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
import shared_modules.global_settings as gs         # pylint: disable=import-error
import shared_modules.shared_functions as shf       # pylint: disable=import-error


layer_types =   ['L', 'L', 'D']       #  'D': Dense     'R': SimpleRNN     'G': GRU     'L': LSTM
neurons_in_layers =  [200, 300, 100]
activation_fcns =    [keras.activations.linear,  keras.activations.linear,  keras.activations.linear]
pre_dropout_layers = [1, 2]
dropout_ratio = 0.1

# ML_model_type = 'ens'
ML_model_type = 'ensSide'

# settings_dict = shf.adjust_ensmble_model_settings(ML_model_type, 'Boh-Dal', '1d',  '1d', neurons_in_layers, neurons_in_layers)
# settings_dict = shf.adjust_ensmble_model_settings(ML_model_type, 'Boh-Dal', '1d',  '3d', neurons_in_layers, neurons_in_layers)
# settings_dict = shf.adjust_ensmble_model_settings(ML_model_type, 'Boh-Dal', '1d',  '1w', neurons_in_layers, neurons_in_layers)
# settings_dict = shf.adjust_ensmble_model_settings(ML_model_type, 'Boh-Dal', '1w',  '1w', neurons_in_layers, neurons_in_layers)
# settings_dict = shf.adjust_ensmble_model_settings(ML_model_type, 'Boh-Dal', '1w',  '3w', neurons_in_layers, neurons_in_layers)
# settings_dict = shf.adjust_ensmble_model_settings(ML_model_type, 'Boh-Dal', '1iM',  '1iM', neurons_in_layers, neurons_in_layers)

# settings_dict = shf.adjust_ensmble_model_settings(ML_model_type, 'PHL-Naz', '1d',  '1d', neurons_in_layers, neurons_in_layers)
# settings_dict = shf.adjust_ensmble_model_settings(ML_model_type, 'PHL-Naz', '1d',  '3d', neurons_in_layers, neurons_in_layers)
# settings_dict = shf.adjust_ensmble_model_settings(ML_model_type, 'PHL-Naz', '1d',  '1w', neurons_in_layers, neurons_in_layers)
# settings_dict = shf.adjust_ensmble_model_settings(ML_model_type, 'PHL-Naz', '1w',  '1w', neurons_in_layers, neurons_in_layers)
# settings_dict = shf.adjust_ensmble_model_settings(ML_model_type, 'PHL-Naz', '1w',  '3w', neurons_in_layers, neurons_in_layers)
# settings_dict = shf.adjust_ensmble_model_settings(ML_model_type, 'PHL-Naz', '1iM',  '1iM', neurons_in_layers, neurons_in_layers)

# settings_dict = shf.adjust_ensmble_model_settings(ML_model_type, 'BoB-BLR', '1d',  '1d', neurons_in_layers, neurons_in_layers)
# settings_dict = shf.adjust_ensmble_model_settings(ML_model_type, 'BoB-BLR', '1d',  '2d', neurons_in_layers, neurons_in_layers)
# settings_dict = shf.adjust_ensmble_model_settings(ML_model_type, 'BoB-BLR', '1d',  '3d', neurons_in_layers, neurons_in_layers)
# settings_dict = shf.adjust_ensmble_model_settings(ML_model_type, 'BoB-BLR', '1d',  '4d', neurons_in_layers, neurons_in_layers)
settings_dict = shf.adjust_ensmble_model_settings(ML_model_type, 'BoB-BLR', '1d',  '5d', neurons_in_layers, neurons_in_layers)



dataset_name = settings_dict['dataset_name']
dataset_sampling_period_str = settings_dict['dataset_sampling_period_str']
NN_input_data_length_str = settings_dict['NN_input_data_length_str']
NN_output_data_length_str = settings_dict['NN_output_data_length_str']
fore_model_path = settings_dict['fore_model_path']
est_model_path = settings_dict['est_model_path']

n_td_jump = 1
# if dataset_sampling_period_str == '1d':
#     n_td_jump = 1#30
# elif dataset_sampling_period_str == '1w':
#     n_td_jump = 1#3
# elif dataset_sampling_period_str == '1iM':
#     n_td_jump = 1
# else:
#     raise Exception(f'The dataset_sampling_period_str value ({dataset_sampling_period_str}) is not supported!')

dataset_sampling_period_dt = gs.str_timedelta[dataset_sampling_period_str]
LPF_window_duration_on_air = '3w'
LPF_window_size_on_air = max(int(  gs.str_timedelta[LPF_window_duration_on_air] / dataset_sampling_period_dt  ), 1)

test_data_duration_str = '1y'

if gs.str_timedelta[NN_input_data_length_str] % gs.str_timedelta[dataset_sampling_period_str] != dt.timedelta(0):
    raise Exception(f'NN_input_data_length_str ({NN_input_data_length_str}) should be a factor of dataset_sampling_period_str ({dataset_sampling_period_str})!')
if gs.str_timedelta[NN_output_data_length_str] % gs.str_timedelta[dataset_sampling_period_str] != dt.timedelta(0):
    raise Exception(f'NN_output_data_length_str ({NN_output_data_length_str}) should be a factor of dataset_sampling_period_str ({dataset_sampling_period_str})!')
if gs.str_timedelta[NN_output_data_length_str] > gs.str_timedelta[NN_input_data_length_str]:
    raise Exception(f'NN_output_data_length_str ({NN_output_data_length_str}) should be smaller than or equal to the NN_input_data_length_str ({NN_input_data_length_str})!')
