import numpy as np
from tensorflow import keras
import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
import shared_modules.global_settings as gs                         # pylint: disable=import-error
import shared_modules.file_operations as fo                         # pylint: disable=import-error
import shared_modules.shared_functions as shf                       # pylint: disable=import-error
import shared_modules.shared_views as shv                           # pylint: disable=import-error
import srv2_create_unified_clean_csv.data_model as unifiedCSV       # pylint: disable=import-error
import srv3_create_decomposed_csv.data_model as decompCSV           # pylint: disable=import-error
import srvFE_Ensemble.ensemble_controller as ensemble_controller    # pylint: disable=import-error
import srvFE_Ensemble.srvFE_ensemble_local_settings as ls           # pylint: disable=import-error



ensController = ensemble_controller.EnsembleController(seed_value=60, LPF_window_size_on_air=ls.LPF_window_size_on_air)

# Training Section
shv.print_separator('Training (finding the optimum coefficients)')
fore_vote_coef, est_vote_coef = ensController.load_latest_voting_coefs()
continue_training = True
if fore_vote_coef is None:
    continue_training = True
else:
    answ = input(f'Are you happy with the current values of the, fore_vote_coef={fore_vote_coef} and est_vote_coef={est_vote_coef}? ').lower()
    if answ in ['y', 'yes', 'ok']:
        continue_training = False
    else:
        continue_training = True
if continue_training:
    fore_vote_coef, est_vote_coef = ensController.train_voting_model()

# Prediction Section
shv.print_separator('Prediction Results')
df_hat, _, _, _, test_true_df = ensController.predict_voting_model(fore_vote_coef, est_vote_coef, time_periode='test')

SSE_for_test_data = (df_hat - test_true_df).pow(2).sum()[gs.ci.water_column(sz=True)]
MSE_for_test_data = SSE_for_test_data / len(df_hat)
RMSE_for_test_data = np.sqrt(MSE_for_test_data)
shv.dict_print(f'Test-data prediction accuracy in "{ls.ML_model_type}"', {'SSE': SSE_for_test_data, 'MSE': MSE_for_test_data, 'RMSE': RMSE_for_test_data})
shv.rect_plot(f'{ls.ML_model_type} Prediction', {'real': ensController.df, 'predicted': df_hat}, gs.ci.water_column(), subplotting=False)
shv.show_plot()
