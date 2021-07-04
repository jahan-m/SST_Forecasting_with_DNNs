import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
import shared_modules.global_settings as gs         # pylint: disable=import-error
import shared_modules.shared_views as shv           # pylint: disable=import-error


def file_paths(path_of="", str_hyper_parameters=None):
    
    # downsampled_csv_folder_path = os.path.join(gs.datasets_folder, 'processed/')
    # downsampled_csv_with_ext = excel_file_name + '_' + str_effective_columns + '_' + gs.downsampling_period_str + '.csv'
    # downsampled_csv_file_path = os.path.join(downsampled_csv_folder_path, downsampled_csv_with_ext)
        
    if path_of is 'raw_csv':
        raw_csv_file_name_with_ext = str_hyper_parameters + '.csv'
        raw_csv_file_path = os.path.join(gs.datasets_folder, raw_csv_file_name_with_ext)
        return raw_csv_file_path
    elif path_of is "unified_clean_csv":
        unified_csv_file_name_with_ext = str_hyper_parameters + '.csv'
        unified_csv_folder_path = os.path.join(gs.datasets_folder, 'processed/')
        unified_csv_file_path = os.path.join(unified_csv_folder_path, unified_csv_file_name_with_ext)
        return (unified_csv_folder_path, unified_csv_file_path)
    elif path_of is "decomposed_csv":
        decomp_csv_folder_path = os.path.join(gs.datasets_folder, 'processed/')
        decomp_csv_file_paths = {}
        f_path = os.path.join(decomp_csv_folder_path, str_hyper_parameters + '_' + gs.SS_1YEAR + '.csv')
        decomp_csv_file_paths.update({gs.SS_1YEAR: f_path})
        f_path = os.path.join(decomp_csv_folder_path, str_hyper_parameters + '_' + gs.SS_ResTrend + '.csv')
        decomp_csv_file_paths.update({gs.SS_ResTrend: f_path})
        f_path = os.path.join(decomp_csv_folder_path, str_hyper_parameters + '_' + gs.SS_YEAR + '.csv')
        decomp_csv_file_paths.update({gs.SS_YEAR: f_path})
        return (decomp_csv_folder_path, decomp_csv_file_paths)
    # elif path_of is "foreARIMA":
    #     pkl_folder_path = os.path.join(gs.datasets_folder, 'model_foreARIMA/')
    #     pkl_f_name = excel_file_name + '_' + str_hyper_parameters + '.pkl'
    #     pkl_file_path = os.path.join(pkl_folder_path, pkl_f_name)
    #     return (pkl_file_path, pkl_folder_path)
    elif path_of in ['fore', 'est', 'estSide', 'ens', 'ensSide']:
        model_folder_path = os.path.join(gs.datasets_folder, 'model_' + path_of + '/')
        model_folder_path = os.path.join(model_folder_path, str_hyper_parameters[:str_hyper_parameters.find('_')])
        model_file_name = str_hyper_parameters + '.h5'
        model_file_path = os.path.join(model_folder_path, model_file_name)
        txt_file_name = str_hyper_parameters + '.txt'
        txt_file_path = os.path.join(model_folder_path, txt_file_name)
        model_file_name_without_ext = str_hyper_parameters
        return (model_folder_path, model_file_path, txt_file_path, model_file_name_without_ext)
    else:
        raise Exception('Unknown path_for in CSV file_paths!')
    

def write_df_to_csv(csv_file_path, df, write_index=True, var_columns=[]):
    try:
        shv.print_separator('Writing to CSV')
        print(f'Writing data-frame to csv file at "{csv_file_path}" ...')
        df.to_csv(csv_file_path, index=write_index, columns=var_columns)
        print("Writing done.")
        return True
    except Exception as inst:
        print(inst)
        raise Exception(f'Could not write df to "{csv_file_path}"!')


def read_df_from_csv(csv_file_path, header=0, names=[], dtype=[], na_values=[], parse_dates=[], index_col=[]):
    df = pd.read_csv(csv_file_path, header=header, names=names, dtype=dtype, na_values=na_values, parse_dates=parse_dates)
    first_valid_index = df.first_valid_index()
    last_valid_index = df.last_valid_index()
    df = df[first_valid_index:last_valid_index+1]
    df.set_index(index_col, inplace=True)
    return df
