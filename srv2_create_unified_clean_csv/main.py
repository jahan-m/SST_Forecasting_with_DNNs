import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
import shared_modules.global_settings as gs                     # pylint: disable=import-error
import shared_modules.shared_views as shv                       # pylint: disable=import-error
import shared_modules.shared_functions as shf                   # pylint: disable=import-error
import shared_modules.file_operations as fo                     # pylint: disable=import-error
import srv2_create_unified_clean_csv.data_model as dm           # pylint: disable=import-error
import srv2_create_unified_clean_csv.srv2_local_settings as ls  # pylint: disable=import-error

######################################################################################### Creating Unified CSV-File

# dataModel = dm.DataModel()

# dataModel.create_unified_csv()
# dataModel.create_csv_for_other_sampling_periods()


######################################################################################### Inspecting Unified CSV-File

all_sampling_periods = ['1d']
for sampling_period_str in ls.other_requested_sampling_period_str:
    all_sampling_periods.append(sampling_period_str)

for sampling_period_str in all_sampling_periods:
    parameters_dict = {'dataset_name': ls.dataset_name,
                       'sampling_period_str': sampling_period_str}
    hyper_params_str = shf.hyper_parameters_str(hyper_str_of='unified_clean_csv', parameters_dict=parameters_dict)
    _, unified_csv_file_path = fo.file_paths(path_of='unified_clean_csv', str_hyper_parameters=hyper_params_str)

    df_temp = fo.read_df_from_csv(unified_csv_file_path, header=0, names=gs.ci.date_air_water_columns(),
                                  dtype=gs.ci.date_air_water_dtypes(), na_values=[], parse_dates=['date'], index_col=['date'])
    shv.rect_plot(f'Plotting {sampling_period_str} df', {'Air Temperature':df_temp, 'Water Temperature':df_temp}, ['air_temp', 'water_temp'], subplotting=False)
    shv.show_plot()
