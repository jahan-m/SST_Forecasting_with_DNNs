from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np
import datetime as dt
import calendar
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
import shared_modules.global_settings as gs                     # pylint: disable=import-error
import shared_modules.file_operations as fo                     # pylint: disable=import-error
import shared_modules.shared_functions as shf                   # pylint: disable=import-error
import shared_modules.shared_views as shv                       # pylint: disable=import-error
import srv2_create_unified_clean_csv.data_model as unifiedCSV   # pylint: disable=import-error
import srv3_create_decomposed_csv.srv3_local_settings as ls     # pylint: disable=import-error


class DataModel:
    def __init__(self):
        if gs.decomp_method not in ['simple_seasonal']:
            raise Exception(f'Unknown decomposition Method ({gs.decomp_method})!')
     
    def create_decomposed_CSVs(self):
        for (dataset_sampling_period_str, dataset_sampling_period_dt) in zip(ls.dataset_sampling_periods_str, ls.dataset_sampling_periods_dt):
            shv.print_separator(f'Creating Decomposed CSVs for {dataset_sampling_period_str} Sampling Period')
            parameters_dict = {'dataset_name': ls.dataset_name, 'sampling_period_str': dataset_sampling_period_str,
                            'test_data_duration_str': ls.test_data_duration_str}
            hyper_params_str = shf.hyper_parameters_str(hyper_str_of='decomposed_csv', parameters_dict=parameters_dict)
            decomp_csv_folder_path, decomp_csv_file_paths = fo.file_paths(path_of='decomposed_csv', str_hyper_parameters=hyper_params_str)
            
            n_existing_decomp_files = 0
            for decomp_csv_file in decomp_csv_file_paths.values():
                if os.path.isfile(decomp_csv_file):
                    n_existing_decomp_files += 1
            if n_existing_decomp_files == 3:
                print(f'All {n_existing_decomp_files} decomposed files exist for "{ls.dataset_name}" dataset with "{dataset_sampling_period_str}" sampling periode and "{ls.test_data_duration_str}" test data duration!')
                return True
            if n_existing_decomp_files != 0:
                raise Exception(f'{n_existing_decomp_files} decomposed CSV files exist, while we expect {3}! Try to delete existing ones, and run again to create all.')
            
            df = unifiedCSV.DataModel.read_unified_csv(ls.dataset_name, dataset_sampling_period_str)
            d_s = pd.Series(df[gs.ci.water_column(sz=True)], index=df.index, name=gs.ci.water_column(sz=True))
            del(df)
            
            td = d_s.index[1] - d_s.index[0]
            if td != dataset_sampling_period_dt:
                raise Exception(f'Existing dataset temporal resolution ({td}) does not match to expected interval ({dataset_sampling_period_dt})!')
            test_start_date = d_s.index[-1] - gs.str_timedelta[ls.test_data_duration_str] + td
            ds_train = d_s[:test_start_date-td]
            
            if gs.decomp_method == 'simple_seasonal':
                decomp_DFs = self.decompose_1YEAR_with_simple_seasonal(ds_train, gs.ci.water_column(sz=True))
                del(ds_train)
            else:
                raise Exception('Unknown decomposition Method!')
            df_Year = self.extend_df_1Y(decomp_DFs[gs.SS_1YEAR], d_s.index)
            decomp_DFs.update({gs.SS_ResTrend: pd.DataFrame(d_s) - df_Year})
            decomp_DFs.update({gs.SS_YEAR: df_Year})
            
            if not os.path.exists(decomp_csv_folder_path):
                os.mkdir(decomp_csv_folder_path)
            for k in decomp_csv_file_paths.keys():
                fo.write_df_to_csv(decomp_csv_file_paths[k], decomp_DFs[k], write_index=True, var_columns=gs.ci.water_column())
        
    def decompose_1YEAR_with_simple_seasonal(self, s_df, aimed_column):
        column = [aimed_column]
        if len(column) != 1:
            raise Exception(f'Number of columns in seriali_df is {len(column)}, which should be 1 for decomposition!')
        decomp_DFs = {}
        data_per_year = dt.timedelta(days=365) / (s_df.index[1]-s_df.index[0])
        decomposition = seasonal_decompose(s_df, model='additive', period=round(data_per_year), extrapolate_trend='freq')
        seasonal_year = decomposition.seasonal
        seasonal_year.name = column[0]
        seasonal_year.index.name = s_df.index.name
        # if dataset_sampling_period_str == 'idealM':
        #     decomp_DFs.update({gs.SS_1YEAR: seasonal_year.iloc[0:12]})
        # else:
        a_typ_year = seasonal_year.index[0].year + 1
        if calendar.isleap(a_typ_year):
            a_typ_year += 1
        year_start = dt.datetime(a_typ_year, 1, 1, 0, 0, 0, 0)
        year_end = dt.datetime(a_typ_year, 12, 31, 23, 59, 59, 999999)
        decomp_DFs.update({gs.SS_1YEAR: pd.DataFrame(seasonal_year.loc[year_start:year_end])}) # YEAR_1PRD
        return decomp_DFs
    
    @staticmethod
    def extend_df_1Y(df_1Y, desired_indexes):
        y_start = desired_indexes[0].year
        y_end = desired_indexes[-1].year
        
        df_1Y_extend = df_1Y.copy()
        df_1Y_extend.drop(df_1Y_extend.index, inplace=True)
        for yy in range(y_start-1, y_end+2):
            df_temp = df_1Y.copy()
            df_temp['new_date'] = df_temp.index
            df_temp['new_date'] = df_temp['new_date'].apply(lambda x: x.replace(year=yy)) #, axis=1
            df_temp.set_index('new_date', inplace=True)
            df_temp.index.name = gs.ci.date_column(sz=True)
            df_1Y_extend = df_1Y_extend.append(df_temp)
        
        col = df_1Y_extend.columns[0]
        df = pd.DataFrame(index=desired_indexes)
        # array of indexes corresponding with closest timestamp after resample
        idx_after = np.searchsorted(df_1Y_extend.index.values, df.index.values)
        # values and timestamp before/after resample
        df['after'] = df_1Y_extend.iloc[idx_after][col].values.astype(float)
        df['before'] = df_1Y_extend.iloc[idx_after - 1][col].values.astype(float)
        df['after_time'] = df_1Y_extend.index[idx_after]
        df['before_time'] = df_1Y_extend.index[idx_after - 1]
        #calculate new weighted value
        df['span'] = (df['after_time'] - df['before_time'])
        df['after_weight'] = ((df.index - df['before_time']) / df['span']).astype(float)
        df['before_weight'] = ((df['after_time'] - df.index) / df['span']).astype(float)
        df[col] = df.eval('before * before_weight + after * after_weight')
        df.drop(columns=['after', 'before', 'after_time', 'before_time', 'span', 'after_weight', 'before_weight'], inplace=True)
        
        df.index.name = df_1Y.index.name
        return df
    
    @staticmethod
    def read_decomposed_CSVs(dataset_name, dataset_sampling_period_str, test_data_duration_str):
        parameters_dict = {'dataset_name': dataset_name, 'sampling_period_str': dataset_sampling_period_str,
                           'test_data_duration_str': test_data_duration_str}
        hyper_params_str = shf.hyper_parameters_str(hyper_str_of='decomposed_csv', parameters_dict=parameters_dict)
        _, decomp_csv_file_paths = fo.file_paths(path_of='decomposed_csv', str_hyper_parameters=hyper_params_str)
        decomp_DFs = {}
        try:
            for k in decomp_csv_file_paths.keys():
                df = fo.read_df_from_csv(decomp_csv_file_paths[k], header=0, names=gs.ci.date_water_columns(),
                                     dtype=gs.ci.date_water_dtypes(), parse_dates=gs.ci.date_column(), index_col=gs.ci.date_column())
                decomp_DFs.update({k: df})
            return decomp_DFs
        except Exception as inst:
            print(inst)
            raise Exception(f'Could not read decomposed csv files for "{hyper_params_str}"!')
    