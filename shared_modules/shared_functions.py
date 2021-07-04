import datetime as dt
from tensorflow import keras
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
import shared_modules.shared_views as shv           # pylint: disable=import-error
import shared_modules.global_settings as gs         # pylint: disable=import-error
import shared_modules.shared_functions as shf       # pylint: disable=import-error
import shared_modules.file_operations as fo         # pylint: disable=import-error


def list_processing_devices():
    from tensorflow.python.client import device_lib # pylint: disable=import-error, no-name-in-module
    local_devices = device_lib.list_local_devices() # List CPU and GPU devices
    shv.dict_print(f'List of Local CPUs and GPUs', {'local_devices': local_devices})

def hyper_parameters_str(hyper_str_of='', parameters_dict=None):
    if hyper_str_of == 'raw_csv':
        return parameters_dict['location_name'] + '_' +\
            str(parameters_dict['years_from_to'][0]) + 'to' + str(parameters_dict['years_from_to'][1]) + '_' +\
            parameters_dict['sampling_period_str']
    elif hyper_str_of == 'unified_clean_csv':
        return parameters_dict['dataset_name'] + '_' + parameters_dict['sampling_period_str']
    elif hyper_str_of == 'decomposed_csv':
        return parameters_dict['dataset_name'] + '_' + parameters_dict['sampling_period_str'] + '_Test' + parameters_dict['test_data_duration_str']
    elif hyper_str_of == 'deep_network':
        sz = parameters_dict['dataset_name'] + '_' + parameters_dict['sampling_period_str']
        sz = sz + '_N'
        for n in parameters_dict['neurons_in_layers']:
            sz = sz + str(n)
        sz = sz + '_' + parameters_dict['NN_input_data_length_str'] + '-' + parameters_dict['NN_output_data_length_str']
        sz = sz + '_W' + str(parameters_dict['n_weight_snapshot'])
        return sz
    elif hyper_str_of == 'vote_ens_network':
        sz = parameters_dict['dataset_name'] + '_' + parameters_dict['sampling_period_str']
        sz = sz + '_' + parameters_dict['NN_input_data_length_str'] + '-' + parameters_dict['NN_output_data_length_str']
        return sz
    else:
        raise Exception(f'The hyper_str_of value ({hyper_str_of}) is not recognized!')

def create_deep_network_structure(layer_types=[], neurons_in_layers=[], activation_fcns=[], pre_dropout_layers=[], dropout_ratio=0.1,
                                  n_input_neurons=None, n_output_neurons=None):
    n_layers = len(neurons_in_layers)
    deep_model = keras.models.Sequential()
    if layer_types[0] == 'D':
        deep_model.add(keras.layers.Dense( neurons_in_layers[0], activation_fcns[0], input_shape=(int(n_input_neurons),) ))
    elif layer_types[0] == 'R':
        deep_model.add(keras.layers.SimpleRNN(neurons_in_layers[0], return_sequences=True, return_state=False,            # go_backwards=True,
                                                    activation=activation_fcns[0], input_shape=(1, int(n_input_neurons),)) )
    elif layer_types[0] == 'G':
        deep_model.add(keras.layers.GRU(neurons_in_layers[0], return_sequences=True, return_state=False,                  # go_backwards=True,
                                                    activation=activation_fcns[0], input_shape=(1, int(n_input_neurons),)) )
    elif layer_types[0] == 'L':
        deep_model.add(keras.layers.LSTM(neurons_in_layers[0], return_sequences=True, return_state=False,                 # go_backwards=True,
                                                    activation=activation_fcns[0], input_shape=(1, int(n_input_neurons),)) )
    else:
        raise Exception(f'Unrecognized layer_types of "{layer_types[0]}"')
    for ii in range(1, n_layers):
        if ii in pre_dropout_layers:
            deep_model.add(keras.layers.Dropout(dropout_ratio))
        if layer_types[ii] == 'D':
            deep_model.add(keras.layers.Dense(neurons_in_layers[ii], activation_fcns[ii]))
        elif layer_types[ii] == 'R':
            deep_model.add( keras.layers.SimpleRNN(neurons_in_layers[ii], return_sequences=True, return_state=False,      # go_backwards=True,
                                                        activation=activation_fcns[ii]) )
        elif layer_types[ii] == 'G':
            deep_model.add( keras.layers.GRU(neurons_in_layers[ii], return_sequences=True, return_state=False,            # go_backwards=True,
                                                        activation=activation_fcns[ii]) )
        elif layer_types[ii] == 'L':
            deep_model.add( keras.layers.LSTM(neurons_in_layers[ii], return_sequences=True, return_state=False,           # go_backwards=True,
                                                        activation=activation_fcns[ii]) )
        else:
            raise Exception(f'Unrecognized layer_types of "{layer_types[ii]}"')
    if neurons_in_layers[-1] != n_output_neurons:
        deep_model.add(keras.layers.Dense(n_output_neurons, keras.activations.linear))
    
    return deep_model

def adjust_deep_model_settings(dataset_abbr, dataset_sampling_period_str, NN_output_data_length_str):
    # dataset_sampling_period_str = '1d'
    # dataset_sampling_period_str = '1w'
    # dataset_sampling_period_str = '1iM'       <<< For this one, use       1998
    settings_dict = {
        'Boh-Dal': {                             ############################## Boh-Dal
            '1d': {
                '1d': {
                    'dataset_name': 'Boh-Dal_2007to2020',
                    'NN_input_data_length_str': '1w'
                },
                '3d': {
                    'dataset_name': 'Boh-Dal_2007to2020',
                    'NN_input_data_length_str': '2w'
                },
                '1w': {
                    'dataset_name': 'Boh-Dal_2007to2020',
                    'NN_input_data_length_str': '3w'
                }
            },
            '1w': {
                '1w': {
                    'dataset_name': 'Boh-Dal_2007to2020',
                    'NN_input_data_length_str': '3w'
                },
                '3w': {
                    'dataset_name': 'Boh-Dal_2007to2020',
                    'NN_input_data_length_str': '3w'
                }
            },
            '1iM': {
                '1iM': {
                    'dataset_name': 'Boh-Dal_1998to2020',
                    'NN_input_data_length_str': '5iM'
                }
            }
        },
        'PHL-Naz': {                             ############################## PHL-Naz
            '1d': {
                '1d': {
                    'dataset_name': 'PHL-Naz_2007to2020',
                    'NN_input_data_length_str': '1w'
                },
                '3d': {
                    'dataset_name': 'PHL-Naz_2007to2020',
                    'NN_input_data_length_str': '2w'
                },
                '1w': {
                    'dataset_name': 'PHL-Naz_2007to2020',
                    'NN_input_data_length_str': '3w'
                }
            },
            '1w': {
                '1w': {
                    'dataset_name': 'PHL-Naz_2007to2020',
                    'NN_input_data_length_str': '3w'
                },
                '3w': {
                    'dataset_name': 'PHL-Naz_2007to2020',
                    'NN_input_data_length_str': '3w'
                }
            },
            '1iM': {
                '1iM': {
                    'dataset_name': 'PHL-Naz_1998to2020',
                    'NN_input_data_length_str': '5iM'
                }
            }
        },
        'BoB-BLR': {                             ############################## BoB-BLR
            '1d': {
                '1d': {
                    'dataset_name': 'BoB-BLR_2007to2020',
                    'NN_input_data_length_str': '1w'
                },
                '2d': {
                    'dataset_name': 'BoB-BLR_2007to2020',
                    'NN_input_data_length_str': '1w'
                },
                '3d': {
                    'dataset_name': 'BoB-BLR_2007to2020',
                    'NN_input_data_length_str': '2w'
                },
                '4d': {
                    'dataset_name': 'BoB-BLR_2007to2020',
                    'NN_input_data_length_str': '2w'
                },
                '5d': {
                    'dataset_name': 'BoB-BLR_2007to2020',
                    'NN_input_data_length_str': '3w'
                }
            },
            '1w': {
                '1w': {
                    'dataset_name': 'BoB-BLR_2007to2020',
                    'NN_input_data_length_str': '3w'
                }
            },
            '1iM': {
                '1iM': {
                    'dataset_name': 'BoB-BLR_1998to2020',
                    'NN_input_data_length_str': '5iM'
                }
            }
        }
    }
    
    try:
        settings_dict = settings_dict[dataset_abbr][dataset_sampling_period_str][NN_output_data_length_str]
    except:
        raise Exception(f'The specified dataset parameters are not match!')
    
    settings_dict.update({'dataset_sampling_period_str': dataset_sampling_period_str,
                     'NN_output_data_length_str': NN_output_data_length_str})
    
    return settings_dict

def adjust_ensmble_model_settings(ML_model_type, dataset_abbr, dataset_sampling_period_str, NN_output_data_length_str, neurons_in_fore_layers, neurons_in_est_layers):
    # dataset_sampling_period_str = '1d'
    # dataset_sampling_period_str = '1w'
    # dataset_sampling_period_str = '1iM'       <<< For this one, use       1998
    settings_dict = adjust_deep_model_settings(dataset_abbr, dataset_sampling_period_str,  NN_output_data_length_str)

    dataset_name = settings_dict['dataset_name']
    dataset_sampling_period_str = settings_dict['dataset_sampling_period_str']
    NN_input_data_length_str = settings_dict['NN_input_data_length_str']
    NN_output_data_length_str = settings_dict['NN_output_data_length_str']
    
    ML_model_types = ['fore', 'est'] if ML_model_type == 'ens' else ['fore', 'estSide']
    neurons_in_layers_of_models = [neurons_in_fore_layers, neurons_in_est_layers]
    model_file_paths = []
    for ML_model_type, neurons_in_layers in zip(ML_model_types, neurons_in_layers_of_models):
        hp_dict = {'dataset_name': dataset_name, 'sampling_period_str': dataset_sampling_period_str,
                   'neurons_in_layers': neurons_in_layers, 'NN_input_data_length_str': NN_input_data_length_str,
                   'NN_output_data_length_str': NN_output_data_length_str, 'ML_model_type': ML_model_type,
                   'n_weight_snapshot': ''}
        
        hp_dict['n_weight_snapshot'] = ''
        hp_str = shf.hyper_parameters_str(hyper_str_of='deep_network', parameters_dict=hp_dict)
        model_folder_path, _, _, model_file_name_without_ext = fo.file_paths(path_of=ML_model_type, str_hyper_parameters=hp_str)
        files = os.listdir(model_folder_path)
        NN = []
        for the_file in files:
            if the_file.endswith('.h5'):
                n_W = the_file.find('_W')
                nExt = the_file.find('.h5')
                NN.append(int(the_file[n_W+2 : nExt]))
        NN = list( dict.fromkeys(NN) )
        NN.sort()
        for nn in NN:
            if (model_file_name_without_ext + str(nn) + '.h5') in files:
                hp_dict['n_weight_snapshot'] = nn
        hp_str = shf.hyper_parameters_str(hyper_str_of='deep_network', parameters_dict=hp_dict)
        _, model_file_path, _, _ = fo.file_paths(path_of=ML_model_type, str_hyper_parameters=hp_str)
        model_file_paths.append(model_file_path)
    
    settings_dict.update({'fore_model_path': model_file_paths[0]})
    settings_dict.update({'est_model_path': model_file_paths[1]})
    return settings_dict
