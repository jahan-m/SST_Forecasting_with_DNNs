import pandas as pd
import numpy as np
import xarray as xr
import datetime as dt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
import shared_modules.file_operations as fo                         # pylint: disable=import-error
import shared_modules.global_settings as gs                         # pylint: disable=import-error
import shared_modules.shared_views as shv                           # pylint: disable=import-error
import shared_modules.shared_functions as shf                       # pylint: disable=import-error
import srv1_A_create_raw_water_csv.srv1_A_local_settings as ls      # pylint: disable=import-error

class DataModel:
    def __init__(self):
        self.df = pd.DataFrame([], columns=gs.ci.date_water_columns())
        self.df.set_index(gs.ci.date_column(sz=True), inplace=True)
        
    def convert_nc_water_to_raw_csv(self):
        # parameters_dict = {'location_name': ls.location_name,
        #                    'years_from_to': ls.years_from_to,
        #                    'sampling_period_str': ls.dataset_sampling_period_str,
        #                    'air_or_water': gs.ci.water_column(sz=True)}
        parameters_dict = {'location_name': ls.location_name,
                           'years_from_to': ls.years_from_to,
                           'sampling_period_str': ls.dataset_sampling_period_str}
        hyper_params_str = shf.hyper_parameters_str(hyper_str_of='raw_csv', parameters_dict=parameters_dict)
        
        raw_csv_file_path = fo.file_paths(path_of='raw_csv', str_hyper_parameters=hyper_params_str)
        
        if not os.path.isdir(gs.datasets_folder):
            os.mkdir('./'+gs.datasets_folder)
        
        if not os.path.isfile(raw_csv_file_path):
            self.extract_water_df_from_nc_files()
            if self.df.index[-1] - self.df.index[-2] != ls.dataset_sampling_period_dt:
                raise Exception(f'Rel dataset temporal resolution ({self.df.index[-1] - self.df.index[-2]}) does not match to expectes interval ({ls.dataset_sampling_period_dt})!')
            self.df = self.regulate_sampling_intervals(self.df, ls.dataset_sampling_period_dt, 'linear')
            fo.write_df_to_csv(raw_csv_file_path, self.df, write_index=True, var_columns=gs.ci.water_column())
        else:
            shv.print_separator('Converting NC Water Files to DF')
            self.df = fo.read_df_from_csv(raw_csv_file_path, header=0, names=gs.ci.date_water_columns(),
                                          dtype=gs.ci.date_water_dtypes(), parse_dates=gs.ci.date_column(),
                                          index_col=gs.ci.date_column())
            print(f'The raw CSV file ("{raw_csv_file_path}") already exists!')
    
    def extract_water_df_from_nc_files(self):
        shv.print_separator('Extracting Water DF from NC Files')
        years = range(ls.years_from_to[0], ls.years_from_to[1]+1)
        for y_ind, year in enumerate(years):
            print(f'{(y_ind)/len(years)*100:.1f} %')
            nc_file_path = os.path.join(ls.nc_folder_path, ls.nc_file_name_with_ext(year))
            sst_dataset = xr.open_dataset(nc_file_path, decode_times=True, use_cftime=True)
            times = sst_dataset['time'].values
            sst_values = sst_dataset['sst'].values
            if sum([time.year == int(year) for time in times]) != len(times):
                raise Exception(f'Some year values in "{ls.nc_file_name_with_ext(year)}" are not {int(year)}!')
            for t_ind, time in enumerate(times):
                recorded_time = dt.datetime(time.year, time.month, time.day, time.hour, time.minute, time.second)
                if ls.area_aggregation == 'mean':
                    recorded_value = sst_values[t_ind, ls.lat_ind_from_to[0]:ls.lat_ind_from_to[1], ls.lon_ind_from_to[0]:ls.lon_ind_from_to[1]]
                    recorded_value = np.nanmean(recorded_value)
                else:
                    raise Exception(f'Area aggregation methos ({ls.area_aggregation}) is unknown!')
                df_temp = pd.DataFrame({gs.ci.date_column(sz=True):[recorded_time], gs.ci.water_column(sz=True):[recorded_value]})
                df_temp.set_index(gs.ci.date_column(sz=True), inplace=True)
                self.df = self.df.append(df_temp)
        print(f'Extraction completed successfully for {years[0]} to {years[-1]}.')
    
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
    
    