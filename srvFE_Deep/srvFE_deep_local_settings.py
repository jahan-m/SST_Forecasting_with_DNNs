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

ML_model_type = 'fore'
# ML_model_type = 'est'
# ML_model_type = 'estSide'

# settings_dict = shf.adjust_deep_model_settings('Boh-Dal', '1d',  '1d')
# settings_dict = shf.adjust_deep_model_settings('Boh-Dal', '1d',  '3d')
# settings_dict = shf.adjust_deep_model_settings('Boh-Dal', '1d',  '1w')
# settings_dict = shf.adjust_deep_model_settings('Boh-Dal', '1w',  '1w')
# settings_dict = shf.adjust_deep_model_settings('Boh-Dal', '1w',  '3w')
# settings_dict = shf.adjust_deep_model_settings('Boh-Dal', '1iM', '1iM')

# settings_dict = shf.adjust_deep_model_settings('PHL-Naz', '1d',  '1d')
# settings_dict = shf.adjust_deep_model_settings('PHL-Naz', '1d',  '3d')
# settings_dict = shf.adjust_deep_model_settings('PHL-Naz', '1d',  '1w')
# settings_dict = shf.adjust_deep_model_settings('PHL-Naz', '1w',  '1w')
# settings_dict = shf.adjust_deep_model_settings('PHL-Naz', '1w',  '3w')
# settings_dict = shf.adjust_deep_model_settings('PHL-Naz', '1iM', '1iM')

# settings_dict = shf.adjust_deep_model_settings('BoB-BLR', '1d',  '1d')
# settings_dict = shf.adjust_deep_model_settings('BoB-BLR', '1d',  '2d')
# settings_dict = shf.adjust_deep_model_settings('BoB-BLR', '1d',  '3d')
# settings_dict = shf.adjust_deep_model_settings('BoB-BLR', '1d',  '4d')
settings_dict = shf.adjust_deep_model_settings('BoB-BLR', '1d',  '5d')



dataset_name = settings_dict['dataset_name']
dataset_sampling_period_str = settings_dict['dataset_sampling_period_str']
NN_input_data_length_str = settings_dict['NN_input_data_length_str']
NN_output_data_length_str = settings_dict['NN_output_data_length_str']

dataset_sampling_period_dt = gs.str_timedelta[dataset_sampling_period_str]
LPF_window_duration_on_air = '3w'
LPF_window_size_on_air = max(int(  gs.str_timedelta[LPF_window_duration_on_air] / dataset_sampling_period_dt  ), 1)

if ML_model_type == 'fore':
    causing_column = gs.ci.water_column(sz=True)
elif ML_model_type in ['est', 'estSide']:
    causing_column = gs.ci.air_column(sz=True)
else:
    raise Exception(f'ML_model_type ({ML_model_type}) is unknown!')
test_data_duration_str = '1y'
val_data_duration_str = '1y'
dropout_ratio = 0.1
learning_rate = 0.001
batch_size = 10
n_epochs = 1000
n_train_batch_per_epoch = 10
n_val_batch_per_epoch = 20
consecutive_val_loss_steadiness = 60 #20

if gs.str_timedelta[NN_input_data_length_str] % gs.str_timedelta[dataset_sampling_period_str] != dt.timedelta(0):
    raise Exception(f'NN_input_data_length_str ({NN_input_data_length_str}) should be a factor of dataset_sampling_period_str ({dataset_sampling_period_str})!')
if gs.str_timedelta[NN_output_data_length_str] % gs.str_timedelta[dataset_sampling_period_str] != dt.timedelta(0):
    raise Exception(f'NN_output_data_length_str ({NN_output_data_length_str}) should be a factor of dataset_sampling_period_str ({dataset_sampling_period_str})!')
if gs.str_timedelta[NN_output_data_length_str] > gs.str_timedelta[NN_input_data_length_str]:
    raise Exception(f'NN_output_data_length_str ({NN_output_data_length_str}) should be smaller than or equal to the NN_input_data_length_str ({NN_input_data_length_str})!')
