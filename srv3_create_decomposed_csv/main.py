import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
import shared_modules.shared_views as shv                       # pylint: disable=import-error
import shared_modules.global_settings as gs                     # pylint: disable=import-error
import srv2_create_unified_clean_csv.data_model as unifiedCSV   # pylint: disable=import-error
import srv3_create_decomposed_csv.data_model as dm              # pylint: disable=import-error
import srv3_create_decomposed_csv.srv3_local_settings as ls     # pylint: disable=import-error

dataModel = dm.DataModel()

######################################################################################### Creating Decomposed CSV-Files
# dataModel.create_decomposed_CSVs()

######################################################################################### Inspecting Decomposed CSV-Files
''' Based on the 2nd image below, I have to give "decomp_DFs[gs.SS_ResTrend]" to forecasters.
    Then I have to add Year seasonalities to their outcome '''
for (dataset_sampling_period_str, dataset_sampling_period_dt) in zip(ls.dataset_sampling_periods_str, ls.dataset_sampling_periods_dt):
    df = unifiedCSV.DataModel.read_unified_csv(ls.dataset_name, dataset_sampling_period_str)
    decomp_DFs = dataModel.read_decomposed_CSVs(ls.dataset_name, dataset_sampling_period_str, ls.test_data_duration_str)
    if gs.decomp_method is 'simple_seasonal':
        sumDf = decomp_DFs[gs.SS_ResTrend] + decomp_DFs[gs.SS_YEAR]
        shv.rect_plot(f'Comparing for {dataset_sampling_period_str}', {'Original': df, 'SumOfDecomp': sumDf}, gs.ci.water_column(), subplotting=False)
        shv.show_plot()
        shv.rect_plot(f'Plotting decomp_DFs for {dataset_sampling_period_str}', decomp_DFs, gs.ci.water_column(), subplotting=True)
        shv.show_plot()
    else:
        raise Exception(f'Unknown "{gs.decomp_method}" decomposition method!')



