import re
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
tf.keras.backend.set_floatx('float64')
import random
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
import shared_modules.global_settings as gs                     # pylint: disable=import-error
import shared_modules.file_operations as fo                     # pylint: disable=import-error
import shared_modules.shared_functions as shf                   # pylint: disable=import-error
import shared_modules.shared_views as shv                       # pylint: disable=import-error
import srv2_create_unified_clean_csv.data_model as unifiedCSV   # pylint: disable=import-error
import srv3_create_decomposed_csv.data_model as decompCSV       # pylint: disable=import-error
import srvFE_Ensemble.srvFE_ensemble_local_settings as ls       # pylint: disable=import-error


class EnsembleController:
    def __init__(self, seed_value=None, LPF_window_size_on_air=1):
        self.df = unifiedCSV.DataModel.read_unified_csv(ls.dataset_name, ls.dataset_sampling_period_str)
        
        self.df[gs.ci.air_column(sz=True)] = self.df[gs.ci.air_column(sz=True)].rolling(window=LPF_window_size_on_air, min_periods=0).mean()
        
        self.decomp_DFs = decompCSV.DataModel.read_decomposed_CSVs(ls.dataset_name, ls.dataset_sampling_period_str, ls.test_data_duration_str)
        self.foreDf = self.decomp_DFs[gs.SS_ResTrend]
        
        ind1 = self.df.index[1]
        ind0 = self.df.index[0]
        indM1 = self.df.index[-1]
        self.delta_t = ind1 - ind0
        self.input_td = gs.str_timedelta[ls.NN_input_data_length_str]
        self.outpu_td = gs.str_timedelta[ls.NN_output_data_length_str]
        self.n_input_neurons = self.input_td / self.delta_t
        self.n_output_neurons = self.outpu_td / self.delta_t
        
        
        self.test_end = indM1
        
        self.test_strt = self.test_end - gs.str_timedelta[ls.test_data_duration_str] + self.delta_t
        indx = self.df.index.searchsorted(self.test_strt)
        self.test_strt = self.df.index[indx]
        
        self.train_end = self.test_strt - self.delta_t
        
        self.train_strt = ind0
        
        self.hp_dict = {'dataset_name': ls.dataset_name, 'sampling_period_str': ls.dataset_sampling_period_str,
                        'ML_model_type': ls.ML_model_type, 'NN_input_data_length_str': ls.NN_input_data_length_str,
                        'NN_output_data_length_str': ls.NN_output_data_length_str}
        
        self.fore_model = shf.create_deep_network_structure(layer_types=ls.layer_types, neurons_in_layers=ls.neurons_in_layers,
                                                            activation_fcns=ls.activation_fcns, pre_dropout_layers=ls.pre_dropout_layers,
                                                            dropout_ratio=ls.dropout_ratio,
                                                            n_input_neurons=int(self.n_input_neurons), n_output_neurons=self.n_output_neurons)
        self.fore_model.load_weights(ls.fore_model_path)
        self.est_model = shf.create_deep_network_structure(layer_types=ls.layer_types, neurons_in_layers=ls.neurons_in_layers,
                                                           activation_fcns=ls.activation_fcns, pre_dropout_layers=ls.pre_dropout_layers,
                                                           dropout_ratio=ls.dropout_ratio,
                                                           n_input_neurons=int(self.n_input_neurons), n_output_neurons=self.n_output_neurons)
        self.est_model.load_weights(ls.est_model_path)
        
        if seed_value:
            random.seed(seed_value)
        
    def load_latest_voting_coefs(self):
        hp_str = shf.hyper_parameters_str(hyper_str_of='vote_ens_network', parameters_dict=self.hp_dict)
        model_folder_path, _, txt_file_path, _ = fo.file_paths(path_of=ls.ML_model_type, str_hyper_parameters=hp_str)
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        txt_file_exists = os.path.isfile(txt_file_path)
        if txt_file_exists:
            with open(txt_file_path, 'r') as ff:
                fore_coef = float(ff.readline().rstrip())
                est_coef = float(ff.readline().rstrip())
            return (fore_coef, est_coef)
        else:
            return (None, None)
        
    def train_voting_model(self):
        print('Calculating the output of both the fore and the est models, in response to train df...')
        
        (train_fore_df, train_est_df, train_true_df, _, _, _) =\
            self.create_and_load_model_response_CSVs(do_for_train_dataset=True, do_for_test_dataset=False)
        
        print('Calculating the optimum coefficients...')
        c0 = np.array([1.0, 0.0]) if ls.ML_model_type == 'ens' else np.array([0.0, 1.0]) # c0 is (fore_coef, est_coef)
        resl = 0.01 # Search resolution
        
        (opt_train_fore_coef, opt_train_est_coef, opt_train_MSE) = self.find_optimum_fore_and_est_coef(c0, resl, train_fore_df, train_est_df, train_true_df)
        sz_train = f'Optimum values, based on the train dataset are      fore_coef: {opt_train_fore_coef:.5f},   est_coef: {opt_train_est_coef:.5f},   train_MSE: {opt_train_MSE[0]:.5f}'
        print('  • '+sz_train)
        
        hp_str = shf.hyper_parameters_str(hyper_str_of='vote_ens_network', parameters_dict=self.hp_dict)
        _, _, txt_file_path, _ = fo.file_paths(path_of=ls.ML_model_type, str_hyper_parameters=hp_str)
        with open(txt_file_path, 'w') as ff:
            ff.write(str(opt_train_fore_coef)+'\n')
            ff.write(str(opt_train_est_coef)+'\n')
            # ff.write('\n\n--------------------------\n')
            # ff.write(sz_train+'\n')
            
        return (opt_train_fore_coef, opt_train_est_coef)
    
    def predict_voting_model(self, fore_vote_coef, est_vote_coef, time_periode='test'):
        if time_periode == 'test':
            
            (_, _, _, test_fore_df, test_est_df, test_true_df) =\
                self.create_and_load_model_response_CSVs(do_for_train_dataset=False, do_for_test_dataset=True)
            
            df_hat = fore_vote_coef * test_fore_df + est_vote_coef * test_est_df
            mse_value = ( df_hat - test_true_df ).pow(2).sum() / len(df_hat)
            
            hp_str = shf.hyper_parameters_str(hyper_str_of='vote_ens_network', parameters_dict=self.hp_dict)
            model_folder_path, _, _, model_file_name_without_ext = fo.file_paths(path_of=ls.ML_model_type, str_hyper_parameters=hp_str)
            test_mse_file_path = os.path.join(model_folder_path, model_file_name_without_ext+f'__testMSE-{mse_value[0]:.3f}.txt')
            with open(test_mse_file_path, 'w') as ff:
                ff.write('MSE value for the test dataset is putted in the filename.')
            
            return df_hat, mse_value, test_fore_df, test_est_df, test_true_df
        else:
            raise Exception(f'The time_periode value ({time_periode}) is not suported!')
    
    def create_and_load_model_response_CSVs(self, do_for_train_dataset=True, do_for_test_dataset=True):
        train_fore_df, train_est_df, train_true_df, test_fore_df, test_est_df, test_true_df = (None, None, None, None, None, None)
        
        hp_str = shf.hyper_parameters_str(hyper_str_of='vote_ens_network', parameters_dict=self.hp_dict)
        model_folder_path, _, _, model_file_name_without_ext = fo.file_paths(path_of=ls.ML_model_type, str_hyper_parameters=hp_str)
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        train_csv_file_exists = test_csv_file_exists = False
        if do_for_train_dataset:
            train_fore_file_path = os.path.join(model_folder_path, model_file_name_without_ext + '_train_fore.csv')
            train_est_file_path = os.path.join(model_folder_path, model_file_name_without_ext + '_train_est.csv')
            train_true_file_path = os.path.join(model_folder_path, model_file_name_without_ext + '_train_true.csv')
            train_csv_file_exists = os.path.isfile(train_fore_file_path)
        if do_for_test_dataset:
            test_fore_file_path = os.path.join(model_folder_path, model_file_name_without_ext + '_test_fore.csv')
            test_est_file_path = os.path.join(model_folder_path, model_file_name_without_ext + '_test_est.csv')
            test_true_file_path = os.path.join(model_folder_path, model_file_name_without_ext + '_test_true.csv')
            test_csv_file_exists = os.path.isfile(test_fore_file_path)
        
        if train_csv_file_exists or test_csv_file_exists:
            if do_for_train_dataset:
                train_fore_df = fo.read_df_from_csv(train_fore_file_path, header=0, names=gs.ci.date_water_columns(), dtype=gs.ci.date_water_dtypes(), parse_dates=gs.ci.date_column(), index_col=gs.ci.date_column())
                train_est_df = fo.read_df_from_csv(train_est_file_path, header=0, names=gs.ci.date_water_columns(), dtype=gs.ci.date_water_dtypes(), parse_dates=gs.ci.date_column(), index_col=gs.ci.date_column())
                train_true_df = fo.read_df_from_csv(train_true_file_path, header=0, names=gs.ci.date_water_columns(), dtype=gs.ci.date_water_dtypes(), parse_dates=gs.ci.date_column(), index_col=gs.ci.date_column())
            if do_for_test_dataset:
                test_fore_df = fo.read_df_from_csv(test_fore_file_path, header=0, names=gs.ci.date_water_columns(), dtype=gs.ci.date_water_dtypes(), parse_dates=gs.ci.date_column(), index_col=gs.ci.date_column())
                test_est_df = fo.read_df_from_csv(test_est_file_path, header=0, names=gs.ci.date_water_columns(), dtype=gs.ci.date_water_dtypes(), parse_dates=gs.ci.date_column(), index_col=gs.ci.date_column())
                test_true_df = fo.read_df_from_csv(test_true_file_path, header=0, names=gs.ci.date_water_columns(), dtype=gs.ci.date_water_dtypes(), parse_dates=gs.ci.date_column(), index_col=gs.ci.date_column())
        else:
            if do_for_train_dataset:
                (train_index_array, train_fore_out_array, train_est_out_array, train_true_out_array) =\
                    self.compile_model_responses_into_df('Train', n_td_jump=ls.n_td_jump)
                
                train_fore_df = pd.DataFrame(train_fore_out_array, index=train_index_array, columns=gs.ci.water_column())
                train_fore_df.index.name = gs.ci.date_column(sz=True)
                fo.write_df_to_csv(train_fore_file_path, train_fore_df, write_index=True, var_columns=gs.ci.water_column())
                
                train_est_df = pd.DataFrame(train_est_out_array, index=train_index_array, columns=gs.ci.water_column())
                train_est_df.index.name = gs.ci.date_column(sz=True)
                fo.write_df_to_csv(train_est_file_path, train_est_df, write_index=True, var_columns=gs.ci.water_column())
                
                train_true_df = pd.DataFrame(train_true_out_array, index=train_index_array, columns=gs.ci.water_column())
                train_true_df.index.name = gs.ci.date_column(sz=True)
                fo.write_df_to_csv(train_true_file_path, train_true_df, write_index=True, var_columns=gs.ci.water_column())
            if do_for_test_dataset:
                (test_index_array, test_fore_out_array, test_est_out_array, test_true_out_array) =\
                    self.compile_model_responses_into_df('Test', n_td_jump=1)
                    
                test_fore_df = pd.DataFrame(test_fore_out_array, index=test_index_array, columns=gs.ci.water_column())
                test_fore_df.index.name = gs.ci.date_column(sz=True)
                fo.write_df_to_csv(test_fore_file_path, test_fore_df, write_index=True, var_columns=gs.ci.water_column())
                
                test_est_df = pd.DataFrame(test_est_out_array, index=test_index_array, columns=gs.ci.water_column())
                test_est_df.index.name = gs.ci.date_column(sz=True)
                fo.write_df_to_csv(test_est_file_path, test_est_df, write_index=True, var_columns=gs.ci.water_column())
                
                test_true_df = pd.DataFrame(test_true_out_array, index=test_index_array, columns=gs.ci.water_column())
                test_true_df.index.name = gs.ci.date_column(sz=True)
                fo.write_df_to_csv(test_true_file_path, test_true_df, write_index=True, var_columns=gs.ci.water_column())
        return (train_fore_df, train_est_df, train_true_df, test_fore_df, test_est_df, test_true_df)
    
    def find_optimum_fore_and_est_coef(self, c0, resl, fore_df, est_df, true_df):
        global_mse_min = np.inf
        while(True):
            coef_M = np.array( [[c0+np.array([-resl,-resl]), c0+np.array([-resl,0]), c0+np.array([-resl,resl])],
                                [c0+np.array([0,-resl]), c0+np.array([0,0]), c0+np.array([0,resl])],
                                [c0+np.array([resl,-resl]), c0+np.array([resl,0]), c0+np.array([resl,resl])]] )
            local_mse_min = np.inf
            for row_ind, row_arr in enumerate(coef_M):
                for col_ind, coef_pair in enumerate(row_arr):
                    fore_coef = coef_pair[0]
                    est_coef = coef_pair[1]
                    df_hat = fore_coef * fore_df + est_coef * est_df
                    mse_values = ( df_hat - true_df ).pow(2).sum() / len(df_hat)
                    if local_mse_min > mse_values.values:
                        local_mse_min = mse_values.values
                        c0 = coef_M[row_ind, col_ind]
            if global_mse_min > local_mse_min:
                global_mse_min = local_mse_min
            else:
                break
        opt_fore_coef = c0[0]
        opt_est_coef = c0[1]
        opt_MSE = global_mse_min
        return (opt_fore_coef, opt_est_coef, opt_MSE)
        
    def compile_model_responses_into_df(self, trn_tst, n_td_jump=1):
        if trn_tst not in ['Train', 'Test']:
            raise Exception(f'"{trn_tst}" is Unknown!')
        (fore_start_range, end_range) = (self.train_strt, self.train_end) if trn_tst=='Train' else (self.test_strt, self.test_end)
        est_start_range = fore_start_range
        
        if trn_tst in ['Test']: # out_from_beginning
            fore_start_range = fore_start_range - self.input_td
            if ls.ML_model_type.startswith('ensSide'):
                est_start_range = est_start_range - self.input_td + self.outpu_td
            else:
                est_start_range = est_start_range - self.input_td
        start_point = 0
        
        index_array = []
        fore_out_array = []
        est_out_array = []
        true_out_array = []
        while True:
            fore_t_in_start = fore_start_range + start_point*self.delta_t
            fore_t_in_end = fore_t_in_start + self.input_td - self.delta_t
            t_out_start = fore_t_in_end + self.delta_t
            t_out_end = t_out_start + self.outpu_td -self.delta_t
            
            if ls.ML_model_type.startswith('ensSide'):
                est_t_in_end = t_out_end
                est_t_in_start = est_t_in_end - self.input_td + self.delta_t
            else:
                est_t_in_start = est_start_range + start_point*self.delta_t
                est_t_in_end = est_t_in_start + self.input_td - self.delta_t
            
            if t_out_end > end_range:
                break
            start_point+=n_td_jump
            
            fore_in = self.foreDf.loc[fore_t_in_start:fore_t_in_end, gs.ci.water_column(sz=True)].values.reshape(1, 1, -1)
            # print('==================')
            # print(fore_in)
            # print('++++++++++++++++++')
            # print(self.one_batch_forecasting(fore_in))
            # print('==================')
            fore_out = self.one_batch_forecasting(fore_in) +\
                       self.decomp_DFs[gs.SS_YEAR].loc[t_out_start:t_out_end, gs.ci.water_column(sz=True)].values
            est_in = self.df.loc[est_t_in_start:est_t_in_end, gs.ci.air_column(sz=True)].values.reshape(1, 1, -1)
            est_out = self.one_batch_estimation(est_in)
            true_out = self.df.loc[t_out_start:t_out_end, gs.ci.water_column(sz=True)].values
            # print('==================')
            # print(fore_out)
            # print('++++++++++++++++++')
            # print(true_out)
            # print('==================')
            index_array.append(t_out_end)
            fore_out_array.append(fore_out.numpy().reshape(1,-1)[0][-1])
            est_out_array.append(est_out.numpy().reshape(1,-1)[0][-1])
            true_out_array.append(true_out[-1])
            
        return (index_array, fore_out_array, est_out_array, true_out_array)

    @tf.function
    def one_batch_forecasting(self, x):
        return self.fore_model(x, training=False)
    
    @tf.function
    def one_batch_estimation(self, x):
        return self.est_model(x, training=False)
    

    