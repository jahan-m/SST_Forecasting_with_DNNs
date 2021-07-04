import statsmodels.graphics.tsaplots as st_graph
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def print_separator(title):
    print(f'******************  {title}  ******************')

def nan_check(title, list_or_dict_of_DFs):
    print_separator(title)
    if isinstance(list_or_dict_of_DFs, dict):
        for k, v in list_or_dict_of_DFs.items():
            print(f'Number of NaN values in "{k}" dataframe is:')
            print(v.isna().sum())
    elif isinstance(list_or_dict_of_DFs, list):
        th = ['st', 'nd', 'rd', 'th']
        for ii, df in enumerate(list_or_dict_of_DFs):
            print(f'Number of NaN values in {ii+1}{th[min(ii,3)]} element in given list:')
            print(df.isna().sum())
    else:
        raise Exception('Unknown "DF" type!')

def dict_print(title, dict_or_list_of_dicts):
    print_separator(title)
    if isinstance(dict_or_list_of_dicts, list):
        for dd in dict_or_list_of_dicts:
            for k, v in dd.items():
                print(f'>> {k}: {v}')
            print('     -------------------------------')
    else:
        for k, v in dict_or_list_of_dicts.items():
            print(f'>> {k}: {v}')
    
def rect_plot(title, DF_dict, one_variable_list=[], markers=None, subplotting=True):
    if markers and subplotting:
        raise Exception('Markers for Sub-plots are not supported yet!')
    length = len(DF_dict.keys())
    if len(one_variable_list) == 1 and length != 1:
        one_variable_list = length * one_variable_list
    plt.figure()
    for ii, (k, v) in enumerate(DF_dict.items()):
        if subplotting:
            plt.subplot(length, 1, ii+1)
        if isinstance(v, np.ndarray) or isinstance(v, list):
            plt.plot(v, label=k)
        else: #isinstance(v, pd.core.frame.DataFrame)
            plt.plot(v[one_variable_list[ii]], label=k)
        if subplotting:
            plt.legend(loc='upper right')
        plt.ylabel(one_variable_list[ii])
    if markers:
        plt.plot(list(markers.keys()), list(markers.values()), marker='o', linestyle='None')
    if not subplotting:
        plt.legend()
    print_separator(title)
    print(f'Rect_plot is plotted in figure {plt.gcf().number}.')
    
def stationary_reports(title, df, variables_to_test=[], win_size=12):
    if len(variables_to_test)>0:
        for variable in variables_to_test:
            print_separator(title)
            print(f'Results of Dickey-Fuller Test for {variable}:')
            dftest = adfuller(df[variable].values, autolag='AIC')
            dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value','#Lags Used', 'Number of Observations Used'])
            for key,value in dftest[4].items():
                dfoutput['Critical Value (%s)'%key] = value
            print(dfoutput)

    plt.figure()
    rolmean = df.rolling(window=win_size, min_periods=1).mean()
    plt.plot(rolmean, label='Rolling Mean')
    rolstd = df.rolling(window=win_size, min_periods=1).std()
    plt.plot(rolstd, label='Rolling STD')
    plt.legend(loc='center right') # upper|center|lower right|left
    print_separator(title)
    print(f'Rolling mean and std are plotted in figure {plt.gcf().number}.')
    
def acf_pacf_plot(title, df, one_variable, n_lags=10):
    print_separator(title)
    st_graph.plot_acf(df[one_variable].values, lags=n_lags)
    print(f'ACF is plotted in figure {plt.gcf().number}.')
    st_graph.plot_pacf(df[one_variable].values, lags=n_lags)
    print(f'Partial-ACF is plotted in figure {plt.gcf().number}.')
    
def show_plot():
    plt.show()
    