## Sea Surface Temperature Forecasting with Ensemble of Stacked Deep Neural Networks
Oceanic temperature has a great impact on global climate and worldwide ecosystems, as its anomalies have been shown to have a direct impact on atmospheric anomalies. The major parameter for measuring the thermal energy of oceans is the Sea Surface Temperature (SST). SST prediction plays an essential role in climatology and ocean-related studies. However, SST prediction is challenging due to the involvement of complex and nonlinear sea thermodynamic factors. To address this challenge,
we design a novel ensemble of two stacked Deep Neural Networks (DNN) that uses air temperature, in addition to water temperature, to improve the SST prediction accuracy.
To train our model and compare its accuracy with the state-of-the-art, we employ two well-known datasets from the national oceanic and atmospheric administration as well as the international Argo project. Using DNNs, our proposed method is capable of automatically extracting required features from the input timeseries and utilizing them internally to provide a highly accurate SST prediction that outperforms previously published works.

## Model Architecture
Block diagram of (a) the voting ensemble model for SST forecasting, which consists of two (b) stacked LSTM-MLP deep neural networks:
<p align="center">
  <img src=https://user-images.githubusercontent.com/65441107/124379851-bc108280-dcfc-11eb-9848-a4f81567baa0.png max-width="500" width=50% title="Ensemble of Stacked Deep Neural Networks Model Architecture">
</p>

## Results and Comparisons
Mean squared error of area-averaged SST forecasting at the Bohai Sea, compared with the different schemes used in [1]:
<p align="center">
  <img src=https://user-images.githubusercontent.com/65441107/124380341-7f925600-dcff-11eb-93d5-5edf3e0a155e.png max-width="500" width=85% title="SST forecasting at the Bohai Sea">
</p>

Mean squared error of area-averaged SST forecasting at the North Pacific Ocean, compared with the proposed model in [2]:
<p align="center">
  <img src=https://user-images.githubusercontent.com/65441107/124380345-85883700-dcff-11eb-8c51-8f4f1693e318.png max-width="500" width=85% title="SST forecasting at the North Pacific Ocean">
</p>

<ol>
<li>J. Xie, J. Zhang, J. Yu, and L. Xu, “An adaptive scale sea surface temperature predicting method based on deep learning with attention mechanism,” IEEE Geoscience and Remote Sensing Letters, vol. 17, no. 5, pp. 740–744, Jul. 2019.
<li>K. Zhang, X. Geng, and X. Yan, “Prediction of 3-D ocean temperature by multilayer convolutional LSTM,” IEEE Geoscience and Remote Sensing Letters, pp. 1–5, Jan. 2020.
</ol>

## Code Description
Before you proceed with this code, the following datasets must be downloaded into your local machine:

<ul>
  <li><a href="https://www.ncdc.noaa.gov/cdo-web/">NCDC NOAA</a>: Daily air temperature data for any desired weather station,
  <li><a href="https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html">PSL NOAA</a>: Daily SST data for any desired range of latitudes and longitudes,
  <li><a href="http://www.argo.ucsd.edu/">Argo</a>: Monthly SST data for any desired range of latitudes and longitudes.
</ul>

The proposed ensemble model is implemented using Keras APIs of TensorFlow in Python.
It follows a multi-service structure, where every service has its own duties to accomplish. In this regard,
<ul>
  <li><b>srv1_A_create_raw_water_csv</b> and <b>srv1_B_create_raw_air_csv</b>: Read the pre-downloaded SST and air temperature NC-files and save them as CSV-files on the local machine.</li>
  <li><b>srv2_create_unified_clean_csv</b>: Merges the separated CSV-files for SST and air temperature into a unified clean CSV-file.</li>
  <li><b>srv3_create_decomposed_csv</b>: Performs a seasonal decomposition on both the SST and air temperature timeseries to extract their seasonal behaviour, as well as their trends and residuals.</li>
  <li><b>srvFE_Deep</b>: Trains two distinct stacked LSTM-MLP DNNs to predict future SST values, based on both the historical SST values and historical air temperatures.</li>
  <li><b>srvFE_Ensemble</b>: Trains the final voting ensemble model to merge two former stacked LSTM-MLP DNN prediction results into the final high-accurate SST prediction.</li>
</ul>

## Citation
Please cite this work as follows:

<ul>
  <li>Mohammad Jahanbakht, Wei Xiang, and Mostafa Rahimi Azghadi, "Sea Surface Temperature Forecasting with Ensemble of Stacked Deep Neural Networks," Submitted to <i>IEEE Geoscience and Remote Sensing Letters</i>, 2021.
</ul>
