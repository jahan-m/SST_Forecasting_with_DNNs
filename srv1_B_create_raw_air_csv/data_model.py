import pandas as pd
import numpy as np
import xarray as xr
import datetime as dt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
import shared_modules.file_operations as fo                     # pylint: disable=import-error
import shared_modules.global_settings as gs                     # pylint: disable=import-error
import shared_modules.shared_views as shv                       # pylint: disable=import-error
import shared_modules.shared_functions as shf                   # pylint: disable=import-error
import srv1_B_create_raw_air_csv.srv1_B_local_settings as ls    # pylint: disable=import-error

class DataModel:
    def __init__(self):
        self.df = pd.DataFrame([], columns=gs.ci.date_air_columns())
        self.df.set_index(gs.ci.date_column(sz=True), inplace=True)
        
    def convert_dataset_air_to_raw_csv(self):
        shv.print_separator('Converting Dataset Air Files to Raw CSV')
        parameters_dict = {'location_name': ls.location_name,
                           'years_from_to': ls.years_from_to,
                           'sampling_period_str': ls.dataset_sampling_period_str}
        hyper_params_str = shf.hyper_parameters_str(hyper_str_of='raw_csv', parameters_dict=parameters_dict)
        
        raw_csv_file_path = fo.file_paths(path_of='raw_csv', str_hyper_parameters=hyper_params_str)
        
        if not os.path.isdir(gs.datasets_folder):
            os.mkdir('./'+gs.datasets_folder)
        
        if not os.path.isfile(raw_csv_file_path):
            df_temp = fo.read_df_from_csv(ls.dataset_file_path, header=0, names=ls.dataset_columns, dtype=ls.dataset_column_types,
                                          parse_dates=ls.dataset_date, index_col=ls.dataset_date)
            df_temp.index.name = 'date'
            df_temp.rename(columns={'TAVG':'air_temp'}, inplace=True)
            # aaa = pd.DataFrame(df_temp['air_temp'])
            self.df = self.df.append(pd.DataFrame(df_temp['air_temp']))
            self.df = self.regulate_sampling_intervals(self.df, ls.dataset_sampling_period_dt, 'linear')
            if self.df.index[-1] - self.df.index[-2] != ls.dataset_sampling_period_dt:
                raise Exception(f'Final dataset temporal resolution ({self.df.index[-1] - self.df.index[-2]}) does not match to expectes interval ({ls.dataset_sampling_period_dt})!')
            fo.write_df_to_csv(raw_csv_file_path, self.df, write_index=True, var_columns=gs.ci.air_column())
        else:
            self.df = fo.read_df_from_csv(raw_csv_file_path, header=0, names=gs.ci.date_air_columns(),
                                          dtype=gs.ci.date_air_dtypes(), parse_dates=gs.ci.date_column(),
                                          index_col=gs.ci.date_column())
            print(f'The raw CSV file ("{raw_csv_file_path}") already exists!')
    
    def regulate_sampling_intervals(self, df, delta_t, inter_p_method):
        if len(df.columns) != 1:
            raise Exception(f'df should have 1 column, not {len(df.columns)}!')
        col = df.columns[0]
        rs = pd.DataFrame(index=df.resample(delta_t).mean().iloc[1:].index)
        # array of indexes corresponding with closest timestamp after resample
        idx_after = np.searchsorted(df.index.values, rs.index.values)
        if inter_p_method == 'linear':
            # values and timestamp before/after resample
            rs['after'] = df.iloc[idx_after][col].values.astype(float)
            rs['before'] = df.iloc[idx_after - 1][col].values.astype(float)
            rs['after_time'] = df.index[idx_after]
            rs['before_time'] = df.index[idx_after - 1]
            #calculate new weighted value
            rs['span'] = (rs['after_time'] - rs['before_time'])
            rs['after_weight'] = ((rs.index - rs['before_time']) / rs['span']).astype(float)
            rs['before_weight'] = ((rs['after_time'] - rs.index) / rs['span']).astype(float)
            rs[col] = rs.eval('before * before_weight + after * after_weight')
            rs.drop(columns=['after', 'before', 'after_time', 'before_time', 'span', 'after_weight', 'before_weight'], inplace=True)
            return rs
        else:
            raise Exception(f'The interpolation method ({inter_p_method}) is not suported!')
    