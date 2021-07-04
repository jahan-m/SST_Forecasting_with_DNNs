import matplotlib.pyplot as plt
import tensorflow as tf                                         # pylint: disable=import-error
from tensorflow import keras                                    # pylint: disable=import-error
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
import srvFE_Deep.srvFE_deep_local_settings as ls               # pylint: disable=import-error

class DeepController:
    def __init__(self, seed_value=None, LPF_window_size_on_air=1):
        self.df = unifiedCSV.DataModel.read_unified_csv(ls.dataset_name, ls.dataset_sampling_period_str)
        
        self.df[gs.ci.air_column(sz=True)] = self.df[gs.ci.air_column(sz=True)].rolling(window=LPF_window_size_on_air, min_periods=0).mean()
        
        if ls.ML_model_type == 'fore':
            self.decomp_DFs = decompCSV.DataModel.read_decomposed_CSVs(ls.dataset_name, ls.dataset_sampling_period_str, ls.test_data_duration_str)
            self.working_df = self.decomp_DFs[gs.SS_ResTrend]
        elif ls.ML_model_type.startswith('est') or ls.ML_model_type.startswith('estSide'):
            self.working_df = self.df#.copy()
        else:
            raise Exception(f'The "{ls.ML_model_type}" model is not suported!')
        
        ind0 = self.working_df.index[0]
        ind1 = self.working_df.index[1]
        indM1 = self.working_df.index[-1]
        
        self.delta_t = ind1 - ind0
        if self.delta_t != ls.dataset_sampling_period_dt:
            raise Exception(f'Existing sampling period ({self.delta_t}) is not equal to the expection period ({ls.dataset_sampling_period_dt})!')
        self.in_td = gs.str_timedelta[ls.NN_input_data_length_str]
        self.out_td = gs.str_timedelta[ls.NN_output_data_length_str]
        
        self.n_input_neurons = self.in_td / self.delta_t
        self.n_output_neurons = self.out_td / self.delta_t
        
        self.test_end = indM1
        
        self.test_strt = self.test_end - gs.str_timedelta[ls.test_data_duration_str] + self.delta_t
        indx = self.working_df.index.searchsorted(self.test_strt)
        self.test_strt = self.working_df.index[indx]
        
        self.val_end = self.test_strt - self.delta_t
        
        self.val_strt = self.val_end - gs.str_timedelta[ls.val_data_duration_str] + self.delta_t
        indx = self.working_df.index.searchsorted(self.val_strt)
        self.val_strt = self.working_df.index[indx]
        
        self.train_end = self.val_strt - self.delta_t
        
        self.train_strt = ind0
        
        self.temp_train_loss = None
        self.temp_val_loss = None
        
        self.hp_dict = {'dataset_name': ls.dataset_name, 'sampling_period_str': ls.dataset_sampling_period_str,
                        'ML_model_type': ls.ML_model_type, 'neurons_in_layers': ls.neurons_in_layers,
                        'NN_input_data_length_str': ls.NN_input_data_length_str, 'NN_output_data_length_str': ls.NN_output_data_length_str,
                        'n_weight_snapshot': ''}
        
        self.deep_model = shf.create_deep_network_structure(layer_types=ls.layer_types, neurons_in_layers=ls.neurons_in_layers,
                                                            activation_fcns=ls.activation_fcns, pre_dropout_layers=ls.pre_dropout_layers,
                                                            dropout_ratio=ls.dropout_ratio,
                                                            n_input_neurons=self.n_input_neurons, n_output_neurons=self.n_output_neurons)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=ls.learning_rate)

        self.train_mean_metric = tf.keras.metrics.Mean(name='train_mean_of_loss_metric')
        self.val_mean_metric = tf.keras.metrics.Mean(name='val_mean_of_loss_metric')
        self.test_mean_metric = tf.keras.metrics.Mean(name='test_mean_of_loss_metric')
        
        self.train_ds_batch_itr = iter( tf.data.Dataset.from_generator(lambda: self.in_out_data_generator(
            'Train', shuffle=False, overlap_prev_out=True, repeat=True, out_from_beginning=False),
            output_types=(tf.float32, tf.float32)).batch(ls.batch_size) )
        
        self.val_ds_batch_itr = iter( tf.data.Dataset.from_generator(lambda: self.in_out_data_generator(
            'Val', shuffle=True, overlap_prev_out=True, repeat=True, out_from_beginning=True),
            output_types=(tf.float32, tf.float32)).batch(ls.batch_size) )
        
        self.test_ds_batch_itr = iter( tf.data.Dataset.from_generator(lambda: self.in_out_data_generator(
            'Test', shuffle=True, overlap_prev_out=True, repeat=True, out_from_beginning=True),
            output_types=(tf.float32, tf.float32)).batch(ls.batch_size) )
        
        self.test_ds_itr = iter( tf.data.Dataset.from_generator(lambda: self.in_out_data_generator(
            'Test', shuffle=False, overlap_prev_out=True, repeat=False, out_from_beginning=True),
            output_types=(tf.float32, tf.float32)).batch(1) )
        
        if seed_value:
            random.seed(seed_value)
        
    def load_latest_weight_snapshot(self):
        self.hp_dict['n_weight_snapshot'] = '' # Shoul be an 'int' number. This is an intentional exception here.
        hp_str = shf.hyper_parameters_str(hyper_str_of='deep_network', parameters_dict=self.hp_dict)
        model_folder_path, _, _, model_file_name_without_ext = fo.file_paths(path_of=ls.ML_model_type, str_hyper_parameters=hp_str)
        self.hp_dict['n_weight_snapshot'] = 0
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
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
                self.hp_dict['n_weight_snapshot'] = nn
        
        hp_str = shf.hyper_parameters_str(hyper_str_of='deep_network', parameters_dict=self.hp_dict)
        _, model_file_path, txt_file_path, _ = fo.file_paths(path_of=ls.ML_model_type, str_hyper_parameters=hp_str)
        model_file_exists = os.path.isfile(model_file_path)
        txt_file_exists = os.path.isfile(txt_file_path)
        if model_file_exists and txt_file_exists:
            self.deep_model.load_weights(model_file_path)
            self.hp_dict['n_weight_snapshot'] += 1
            with open(txt_file_path, 'r') as ff:
                self.temp_train_loss = float(ff.readline().split(':')[1])
                self.temp_val_loss = float(ff.readline().split(':')[1])
            return model_file_path
        else:
            return False
    
    def save_current_weights(self, epoch, train_loss, val_loss):
        hp_str = shf.hyper_parameters_str(hyper_str_of='deep_network', parameters_dict=self.hp_dict)
        _, model_file_path, txt_file_path, _ = fo.file_paths(path_of=ls.ML_model_type, str_hyper_parameters=hp_str)
        self.deep_model.save_weights(model_file_path)
        self.hp_dict['n_weight_snapshot'] += 1
        with open(txt_file_path, 'w') as ff:
            ff.write('Train Loss: '+str(train_loss)+'\n')
            ff.write('Val Loss: '+str(val_loss)+'\n')
        print(f'Model is saved to "{model_file_path}" at:')
        print(f'Epoch: {epoch},   Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}')
    
    def train_ML_model(self):
        epoch_of_the_best_val_loss = 0
        train_loss_array = []
        val_loss_array = []
        weights = dict()
        if self.temp_val_loss is not None:
            epoch = 0
            train_loss_array.append(self.temp_train_loss)
            val_loss_array.append(self.temp_val_loss)
            weights.update({epoch: {'train_loss': self.temp_train_loss, 'val_loss': self.temp_val_loss,
                                    'weights': self.deep_model.get_weights()}})
        else:
            epoch = -1
        while(True):
            epoch += 1
            self.train_mean_metric.reset_states()
            self.val_mean_metric.reset_states()
            for _ in range(ls.n_train_batch_per_epoch):
                x, y_true = next(self.train_ds_batch_itr)
                if x is None: return False
                self.One_batch_train_step(x, y_true)
            for _ in range(ls.n_val_batch_per_epoch):
                x, y_true = next(self.val_ds_batch_itr)
                if x is None: return False
                self.one_batch_val_step(x, y_true)
            train_loss_array.append(self.train_mean_metric.result().numpy())
            val_loss_array.append(self.val_mean_metric.result().numpy())
            print(f'Epoch: {epoch},   Train Loss: {train_loss_array[-1]:.3f}, Val Loss: {val_loss_array[-1]:.3f}')
            
            flag1 = True
            for ee in weights.keys():
                if val_loss_array[-1] > weights[ee]['val_loss']:
                    flag1 = False
                    break
            if flag1:
                weights.update({epoch: {'train_loss': train_loss_array[-1], 'val_loss': val_loss_array[-1],
                                        'weights': self.deep_model.get_weights()}})
                epoch_of_the_best_val_loss = epoch
            if epoch >= ls.n_epochs or (epoch - epoch_of_the_best_val_loss) > ls.consecutive_val_loss_steadiness:
                markers = {kk:vv['val_loss'] for kk,vv in weights.items()}
                shv.rect_plot('Train and Validation Loss Plot', {'train_loss': train_loss_array, 'val_loss': val_loss_array}, 'Mean_Square_Error', markers=markers, subplotting=False)
                shv.show_plot()
                weights_reformated = dict()
                for ee in weights.keys():
                    ttt = weights[ee]['train_loss']
                    vvv = weights[ee]['val_loss']
                    weights_reformated.update({ee: f'Epoch: {ee},  Train Loss: {ttt:.3f},  Val Loss: {vvv:.3f}'})
                shv.dict_print(f'Selective Epoch Numbers with Lower Validation Errors', weights_reformated)
                print('\n\nPlease check the above validation errors, and then select from bellow options.')
                print(f'We are currently on our {epoch}th epoch.')
                while(True):
                    ans = input('Continue training (c) or load the N-th epoch and quit training (N): ').lower()
                    if ans == 'c':
                        ls.n_epochs = 2*ls.n_epochs if epoch >= ls.n_epochs else ls.n_epochs
                        epoch_of_the_best_val_loss = epoch
                        break
                    else:
                        try:
                            ans = int(ans)
                            self.deep_model.set_weights(weights[ans]['weights'])
                            self.save_current_weights(ans, weights[ans]['train_loss'], weights[ans]['val_loss'])
                            return True
                        except:
                            print('\n\n')
    
    # @tf.function
    def predict_by_ML_model(self, time_periode='test'):
        y = []
        flag = True
        while True:
            x, _ = next(self.test_ds_itr) if time_periode=='test' else (None, None)
            if np.isnan(x.numpy()).sum() != 0:
                break
            v = self.one_batch_step_without_loss(x)
            if flag:
                for lll in v.numpy().reshape(1,-1)[0]:
                    y.append(lll)
                flag = False
            else:
                y.append( v.numpy().reshape(1,-1)[0][-1] )
        if time_periode is 'test':
            y_df = pd.DataFrame(y, index=self.working_df.loc[self.test_strt:self.test_end].index, columns=gs.ci.water_column())
        else:
            raise Exception(f'The time_periode value ({time_periode}) is not suported!')
        # self.hp_dict['n_weight_snapshot'] -= 1
        # hp_str = shf.hyper_parameters_str(hyper_str_of='deep_network', parameters_dict=self.hp_dict)
        # model_folder_path, _, _, model_file_name_without_ext = fo.file_paths(path_of=ls.ML_model_type, str_hyper_parameters=hp_str)
        # self.hp_dict['n_weight_snapshot'] += 1
        # fo.write_df_to_csv(os.path.join(model_folder_path, model_file_name_without_ext+'_'+time_periode+'.csv'), y_df, write_index=True, columns=gs.ci.water_column())
        return y_df
        
    # @tf.function
    def in_out_data_generator(self, trn_val_tst, shuffle=True, overlap_prev_out=True, repeat=True, out_from_beginning=True):
        if trn_val_tst not in ['Train', 'Val', 'Test']:
            raise Exception(f'"{trn_val_tst}" is Unknown!')
        (start_range, end_range) = (self.train_strt, self.train_end) if trn_val_tst=='Train'\
                              else (self.val_strt, self.val_end) if trn_val_tst=='Val'\
                              else (self.test_strt, self.test_end)
        if out_from_beginning and trn_val_tst in ['Val', 'Test'] and ls.ML_model_type.startswith('estSide'):
            start_range = start_range - self.in_td + self.out_td
        elif out_from_beginning and trn_val_tst in ['Val', 'Test']:
            start_range = start_range - self.in_td
        if shuffle:
            if ls.ML_model_type.startswith('estSide'):
                # n_possible_start_points = ( end_range-start_range+self.delta_t - (self.in_td)+self.delta_t ) / self.delta_t
                n_possible_start_points = int(np.floor( ( end_range-start_range+self.delta_t - (self.in_td)+self.delta_t ) / self.delta_t ))
            else:
                # n_possible_start_points = ( end_range-start_range+self.delta_t - (self.in_td + self.out_td)+self.delta_t ) / self.delta_t
                n_possible_start_points = int(np.floor( ( end_range-start_range+self.delta_t - (self.in_td + self.out_td)+self.delta_t ) / self.delta_t ))
            while True:
                start_point = random.randrange(0, n_possible_start_points)
                t_in_start = start_range + start_point*self.delta_t
                t_in_end = t_in_start + self.in_td - self.delta_t
                if ls.ML_model_type.startswith('estSide'):
                    t_out_start = t_in_end - self.out_td + self.delta_t
                    t_out_end = t_in_end
                else:
                    t_out_start = t_in_end + self.delta_t
                    t_out_end = t_out_start + self.out_td - self.delta_t
                iiin = self.working_df.loc[t_in_start:t_in_end, ls.causing_column].values
                ooout = self.working_df.loc[t_out_start:t_out_end, gs.ci.water_column(sz=True)].values
                if ls.layer_types[0] == 'D':
                    iiin = iiin.reshape(-1)
                    ooout = ooout.reshape(-1)
                elif ls.layer_types[0] in ['R', 'G', 'L']:
                    iiin = iiin.reshape(1, -1)
                    ooout = ooout.reshape(1, -1)
                else:
                    raise Exception(f'"{ls.layer_types[0]}" in unknown!')
                yield (iiin, ooout)
        else:
            if ls.ML_model_type.startswith('estSide'):
                n_td_jump = 1 if overlap_prev_out else (self.in_td)/self.delta_t
            else:
                n_td_jump = 1 if overlap_prev_out else (self.in_td + self.out_td)/self.delta_t
            start_point = 0
            while True:
                t_in_start = start_range + start_point*self.delta_t
                t_in_end = t_in_start + self.in_td - self.delta_t
                if ls.ML_model_type.startswith('estSide'):
                    t_out_start = t_in_end - self.out_td + self.delta_t
                    t_out_end = t_in_end
                else:
                    t_out_start = t_in_end + self.delta_t
                    t_out_end = t_out_start + self.out_td - self.delta_t
                if t_out_end > end_range:
                    if repeat:
                        start_point = 0
                        continue
                    else:
                        break
                start_point+=n_td_jump
                iiin = self.working_df.loc[t_in_start:t_in_end, ls.causing_column].values
                ooout = self.working_df.loc[t_out_start:t_out_end, gs.ci.water_column(sz=True)].values
                if ls.layer_types[0] == 'D':
                    iiin = iiin.reshape(-1)
                    ooout = ooout.reshape(-1)
                elif ls.layer_types[0] in ['R', 'G', 'L']:
                    iiin = iiin.reshape(1, -1)
                    ooout = ooout.reshape(1, -1)
                else:
                    raise Exception(f'"{ls.layer_types[0]}" in unknown!')
                yield (iiin, ooout)
            yield (None, None)
    
    @tf.function
    def One_batch_train_step(self, x, y_true):
        with tf.GradientTape() as tape:
            y_pred = self.deep_model(x, training=True)
            losses = tf.keras.losses.mse(y_true, y_pred)
            gradients = tape.gradient(losses, self.deep_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.deep_model.trainable_variables))
        self.train_mean_metric(losses)

    @tf.function
    def one_batch_val_step(self, x, y_true):
        y_pred = self.deep_model(x, training=False)
        losses = tf.keras.losses.mse(y_true, y_pred)
        self.val_mean_metric(losses)
    
    @tf.function
    def one_batch_test_step(self, x, y_true):
        y_pred = self.deep_model(x, training=False)
        losses = tf.keras.losses.mse(y_true, y_pred)
        self.test_mean_metric(losses)
    
    @tf.function
    def one_batch_step_without_loss(self, x):
        return self.deep_model(x, training=False)
    

    