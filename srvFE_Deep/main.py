import numpy as np
from tensorflow import keras                                    # pylint: disable=import-error
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
import shared_modules.global_settings as gs                     # pylint: disable=import-error
import shared_modules.file_operations as fo                     # pylint: disable=import-error
import shared_modules.shared_functions as shf                   # pylint: disable=import-error
import shared_modules.shared_views as shv                       # pylint: disable=import-error
import srv2_create_unified_clean_csv.data_model as unifiedCSV   # pylint: disable=import-error
import srv3_create_decomposed_csv.data_model as decompCSV       # pylint: disable=import-error
import srvFE_Deep.srvFE_deep_local_settings as ls               # pylint: disable=import-error
import srvFE_Deep.deep_controller as deep_controller            # pylint: disable=import-error


# shfcn.list_processing_devices()

deepController = deep_controller.DeepController(seed_value=60, LPF_window_size_on_air=ls.LPF_window_size_on_air)

# shv.dict_print(f'{ls.ML_model_type} Model Summary', {'summary': ''}); deepController.deep_model.summary()

by_pass_training = False
continue_training = not by_pass_training
tf_file_path = deepController.load_latest_weight_snapshot()
if tf_file_path and not by_pass_training:
    shv.print_separator('Continue Previous Training')
    print(f'Previous weights are loaded from "{tf_file_path}" snapshot.')
    print(f'Train Loss: {deepController.temp_train_loss:.3f}, Val Loss: {deepController.temp_val_loss:.3f}')
    continue_training = True if input('Do you wish to continue with training (y/n)? ').lower() in ['y', 'yes', 'ok'] else False
if continue_training:
    deepController.train_ML_model()

y_hat = deepController.predict_by_ML_model('test')

if ls.ML_model_type == 'fore':
    df_hat = y_hat + deepController.decomp_DFs[gs.SS_YEAR][deepController.test_strt:deepController.test_end]
elif ls.ML_model_type in ['est', 'estSide']:
    df_hat = y_hat
else:
    raise Exception(f'The "{ls.ML_model_type}" model name is not supported!')

SSE_for_test_data = (df_hat - deepController.df.loc[deepController.test_strt:deepController.test_end]).pow(2).sum()[gs.ci.water_column(sz=True)]
MSE_for_test_data = SSE_for_test_data / len(df_hat)
RMSE_for_test_data = np.sqrt(MSE_for_test_data)

deepController.hp_dict['n_weight_snapshot'] -= 1
hp_str = shf.hyper_parameters_str(hyper_str_of='deep_network', parameters_dict=deepController.hp_dict)
model_folder_path, _, _, model_file_name_without_ext = fo.file_paths(path_of=ls.ML_model_type, str_hyper_parameters=hp_str)
deepController.hp_dict['n_weight_snapshot'] += 1
precision_file_path = os.path.join(model_folder_path, model_file_name_without_ext+'_'+f'testMSE-{MSE_for_test_data:.3f}'+'.txt')
with open(precision_file_path, 'w') as ff:
    ff.write('< Test Data Precision Metrics >\n')
    ff.write('SSE: '+str(SSE_for_test_data)+'\n')
    ff.write('MSE: '+str(MSE_for_test_data)+'\n')
    ff.write('RMSE: '+str(RMSE_for_test_data)+'\n')
shv.dict_print(f'Test-data prediction accuracy in "{ls.ML_model_type}"', {'SSE': SSE_for_test_data, 'MSE': MSE_for_test_data, 'RMSE': RMSE_for_test_data})
shv.rect_plot(f'{ls.ML_model_type} Prediction', {'real': deepController.df, 'predicted': df_hat}, gs.ci.water_column(), subplotting=False)
shv.show_plot()
