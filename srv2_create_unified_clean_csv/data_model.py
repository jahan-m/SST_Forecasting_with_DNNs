import numpy as np
import pandas as pd
import datetime as dt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
import shared_modules.file_operations as fo                     # pylint: disable=import-error
import shared_modules.global_settings as gs                     # pylint: disable=import-error
import shared_modules.shared_views as shv                       # pylint: disable=import-error
import shared_modules.shared_functions as shf                   # pylint: disable=import-error
import srv2_create_unified_clean_csv.srv2_local_settings as ls  # pylint: disable=import-error

class DataModel:
    def __init__(self):
        self.df = None#pd.DataFrame([], columns=gs.ci.date_air_water_columns())
        # self.df.set_index(gs.ci.date_column(sz=True), inplace=True)
    
    def create_unified_csv(self):
        parameters_dict = {'dataset_name': ls.dataset_name,
                           'sampling_period_str': ls.dataset_sampling_period_str}
        hyper_params_str = shf.hyper_parameters_str(hyper_str_of='unified_clean_csv', parameters_dict=parameters_dict)
        
        unified_csv_folder_path, unified_csv_file_path = fo.file_paths(path_of='unified_clean_csv', str_hyper_parameters=hyper_params_str)
        
        if not os.path.isdir(unified_csv_folder_path):
            os.mkdir('./'+unified_csv_folder_path)
        
        if not os.path.isfile(unified_csv_file_path):
            self.df = self.create_unified_df()
            
            self.df = self.handle_nan_values(self.df, ls.nan_strategy)
            self.df = self.handle_outliers(self.df, win_size=ls.outlier_win_size, std_factor=ls.outlier_std_factor,
                                           IQR_factor=ls.outlier_IQR_factor, nan_strategy=ls.nan_strategy)
            
            if self.df.index[-1] - self.df.index[-2] != ls.dataset_sampling_period_dt:
                raise Exception(f'Real dataset temporal resolution ({self.df.index[-1] - self.df.index[-2]}) does not match to expectes interval ({ls.dataset_sampling_period_dt})!')
            fo.write_df_to_csv(unified_csv_file_path, self.df, write_index=True, var_columns=gs.ci.air_water_columns())
        else:
            shv.print_separator('Creating Unified CSV')
            self.df = fo.read_df_from_csv(unified_csv_file_path, header=0, names=gs.ci.date_air_water_columns(),
                                          dtype=gs.ci.date_air_water_dtypes(), parse_dates=gs.ci.date_column(),
                                          index_col=gs.ci.date_column())
            print(f'The raw CSV file ("{unified_csv_file_path}") already exists!')
    
    def create_unified_df(self):
        air_file = os.path.join(gs.datasets_folder, ls.merging_files['air'])
        water_file = os.path.join(gs.datasets_folder, ls.merging_files['water'])
        df_air = fo.read_df_from_csv(air_file, header=0, names=gs.ci.date_air_columns(), dtype=gs.ci.date_air_dtypes(),
                                     parse_dates=gs.ci.date_column(), index_col=gs.ci.date_column())
        df_water = fo.read_df_from_csv(water_file, header=0, names=gs.ci.date_water_columns(), dtype=gs.ci.date_water_dtypes(),
                                       parse_dates=gs.ci.date_column(), index_col=gs.ci.date_column())
        start_date = max(df_air.index[0], df_water.index[0])
        stop_date = min(df_air.index[-1], df_water.index[-1])
        
        df_air = df_air.loc[start_date:stop_date]
        df_water = df_water.loc[start_date:stop_date]
        df_temp = df_air.copy()
        df_temp[gs.ci.water_column(sz=True)] = df_water
        return df_temp
    
    def handle_nan_values(self, df, nan_strategy):
        if nan_strategy is None:
            return df
        elif nan_strategy is 'drop':
            return df.dropna()#inplace=True)
        elif nan_strategy is 'linear_interp':
            return df.interpolate(method='linear')
        else:
            raise Exception('Unknown strategy to handle_nan_values!')
    
    def handle_outliers(self, df, win_size, std_factor=1.5, IQR_factor=1.5, nan_strategy=''):
        
        temp_row_0 = df.iloc[0]
        
        if std_factor >= 0:
            rw = df.rolling(window=win_size)  # Create a rolling object (no computation yet)
            r_median = rw.median()
            r_std = rw.std()
            df = df[ ~((df - r_median).abs() > std_factor*r_std) ]
        if IQR_factor >= 0:
            qs = df.quantile([0.25, 0.5, 0.75])
            IQR = pd.DataFrame(index=['h_extrm', 'l_extrm'], columns=df.columns)
            IQR.loc['h_extrm'] = qs.loc[0.5] + IQR_factor* ( (qs.loc[0.25]-qs.loc[0.75]).abs() )
            IQR.loc['l_extrm'] = qs.loc[0.5] - IQR_factor* ( (qs.loc[0.25]-qs.loc[0.75]).abs() )
            df = df[ ~((df>IQR.loc['h_extrm'])  |  (df<IQR.loc['l_extrm'])) ]
        
        df.iloc[0] = temp_row_0
        
        df = self.handle_nan_values(df, nan_strategy)
        for ii in range(win_size-1):
            df.iloc[ii] = df.iloc[0:win_size].median()
        return df

    def create_csv_for_other_sampling_periods(self):
        shv.print_separator(f'Creating Downsampled CSV Files')
        print(f'Requsted downsampling periods are {ls.other_requested_sampling_period_str}.')
        for sampling_period_str in ls.other_requested_sampling_period_str:
            sampling_period_dt = gs.str_timedelta[sampling_period_str]
            if sampling_period_dt <= ls.dataset_sampling_period_dt:
                raise Exception(f'Downsampling period ({sampling_period_dt}) is less than the original data periode ({ls.dataset_sampling_period_dt}). This cannot be considered as a downsampling!')
            if ls.downsampling_aggregation=='mean':
                df_temp = self.df.resample(sampling_period_dt).mean()
            else:
                raise Exception(f'Aggregation method ({ls.downsampling_aggregation}) is not supported!')
            
            parameters_dict = {'dataset_name': ls.dataset_name,
                               'sampling_period_str': sampling_period_str}
            hyper_params_str = shf.hyper_parameters_str(hyper_str_of='unified_clean_csv', parameters_dict=parameters_dict)
            _, unified_csv_file_path = fo.file_paths(path_of='unified_clean_csv', str_hyper_parameters=hyper_params_str)
            if not os.path.isfile(unified_csv_file_path):
                fo.write_df_to_csv(unified_csv_file_path, df_temp, write_index=True, var_columns=gs.ci.air_water_columns())
            else:
                print(f'The raw CSV file ("{unified_csv_file_path}") already exists!')
            
    @staticmethod
    def read_unified_csv(dataset_name, dataset_sampling_period_str):
        parameters_dict = {'dataset_name': dataset_name,
                           'sampling_period_str': dataset_sampling_period_str}
        hyper_params_str = shf.hyper_parameters_str(hyper_str_of='unified_clean_csv', parameters_dict=parameters_dict)
        _, unified_csv_file_path = fo.file_paths(path_of='unified_clean_csv', str_hyper_parameters=hyper_params_str)
        try:
            df = fo.read_df_from_csv(unified_csv_file_path, header=0, names=gs.ci.date_air_water_columns(),
                                     dtype=gs.ci.date_air_water_dtypes(), parse_dates=gs.ci.date_column(), index_col=gs.ci.date_column())
            return df
        except Exception as inst:
            print(inst)
            raise Exception(f'Could not read the unified csv file ("{unified_csv_file_path}")!')
    
