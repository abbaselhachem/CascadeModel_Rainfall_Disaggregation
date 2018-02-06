# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: EL Hachem Abbas, IWS
Institut für Wasser- und Umweltsystemmodellierung - IWS
"""

from scipy.stats import beta, rankdata
from scipy.stats import spearmanr as spr
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import pandas as pd

import os
import timeit
import time
import fnmatch

plt.ioff()

''' this script is for plotting the result of cascadeModel.py:

    plot the histograam of the sample weights and plot the fitted beta
    distribution, this function is called for level one and two.

    plot the probability P01 that W=0 or W=1, for each station, each level

    plot the seasonal effect on the probability P01, for each month and level

    plot the logistic regression fitted function vs the classified rainfall
    part of the unbounded model, do it for all stations and each level

    plot the results of the baseline and unbounded model, using the fitted
    parameters, a subplot with original, baseline and unbounded rainfall

    plot the result of the mean of the 100 simulations done for the baseline
    and the unbounded model and compare to original values
'''


print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
START = timeit.default_timer()  # to get the runtime of the program

# main_dir = ....
os.chdir(main_dir)

# def data dir
in_data_dir = os.path.join(main_dir,
                           r'CascadeModel\Weights')
# =============================================================================
# Level ONE
# =============================================================================
cascade_level_1 = 'Level one'

# in_df of P0 and P1 per month
in_data_prob_df_file_L1 = os.path.join(in_data_dir,
                                       'P1 P0 per month %s.csv'
                                       % cascade_level_1)
# in_df of fitted beta dist params
in_data_beta_df_file_L1 = os.path.join(in_data_dir,
                                       r'bounded maximumlikelihood %s.csv'
                                       % cascade_level_1)
# in_df of P01 per stn
in_data_prob_stn_df_file_L1 = os.path.join(in_data_dir,
                                           r'Prob W P01 %s.csv'
                                           % cascade_level_1)
# in_df of w_vals, R_wals and logRegression cols
unbounded_model_dir_L1 = os.path.join(in_data_dir,
                                      r'%s P01 volume dependancy'
                                      % cascade_level_1)
# in_df of logistic regression params
params_file_L1 = os.path.join(unbounded_model_dir_L1,
                              r'%s log_regress params' % cascade_level_1,
                              r'loglikehood params.csv')

# in_df results of simulation for model evaluation
in_dfs_simulation = os.path.join(in_data_dir,
                                 r'%s model evaluation' % cascade_level_1)

# read original values, to compare to model
in_df_30min_orig = os.path.join(in_data_dir, r'resampled 30min.csv')

# read dfs holding the results of Lorenz curves of simulated values
in_lorenz_df_L1_sim = os.path.join(in_data_dir,
                                   r'%s Lorenz curves simulations'
                                   % cascade_level_1)

# =============================================================================
# Level TWO ONE
# =============================================================================

cascade_level_2 = 'Level two'

# in_df of P0 and P1 per month
in_data_prob_df_file_L2 = os.path.join(in_data_dir,
                                       'P1 P0 per month %s.csv'
                                       % cascade_level_2)
# in_df of fitted beta dist params
in_data_beta_df_file_L2 = os.path.join(in_data_dir,
                                       r'bounded maximumlikelihood %s.csv'
                                       % cascade_level_2)
# in_df of P01 per stn
in_data_prob_stn_df_file_L2 = os.path.join(in_data_dir,
                                           r'Prob W P01 %s.csv'
                                           % cascade_level_2)

# in_df of w_vals, R_wals and logRegression cols
unbounded_model_dir_L2 = os.path.join(in_data_dir,
                                      r'%s P01 volume dependancy'
                                      % cascade_level_2)
# in_df of logistic regression params
params_file_L2 = os.path.join(unbounded_model_dir_L2,
                              r'%s log_regress params' % cascade_level_2,
                              r'loglikehood params.csv')
in_dfs_simulation_2 = os.path.join(in_data_dir,
                                   r'%s model evaluation' % cascade_level_2)
# in_dfs_simulation_2 = os.path.join(in_data_dir,
#                                   r'%s model evaluation' % cascade_level_2)

in_df_15min_orig = os.path.join(in_data_dir, r'resampled 15min.csv')

in_lorenz_df_L2_sim = os.path.join(in_data_dir,
                                   r'%s Lorenz curves simulations'
                                   % cascade_level_2)

# =============================================================================
# create OUT dir and make sure all IN dir are correct
# =============================================================================

# def out_dir to hold plots
out_dir = os.path.join(main_dir,
                       r'Histograms_Weights')
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# define out_dir for these plots
out_figs_dir0 = os.path.join(out_dir, 'out_subplots')

if not os.path.exists(out_figs_dir0):
    os.mkdir(out_figs_dir0)

# make sure all defined directories exist
assert os.path.exists(in_data_dir), 'wrong data DF location'

# LEVEL ONE
assert os.path.exists(in_data_prob_df_file_L1),\
        'wrong data prob location L1'
assert os.path.exists(in_data_prob_stn_df_file_L1),\
        'wrong data stn prob location L1'
assert os.path.exists(in_data_beta_df_file_L1),\
        'wrong data beta location L1'
assert os.path.exists(unbounded_model_dir_L1),\
        'wrong unbounded model DF location L1'
assert os.path.exists(params_file_L1),\
        'wrong params DF location L1'
assert os.path.exists(in_dfs_simulation),\
        'wrong simulation DF location L1'
assert os.path.exists(in_df_30min_orig),\
        'wrong orig DF location L1'
assert os.path.exists(in_lorenz_df_L1_sim),\
         'wrong Lorenz Curves simu Df L1'

# LEVEL TWO
assert os.path.exists(in_data_prob_df_file_L2),\
        'wrong data prob location L2'
assert os.path.exists(in_data_prob_stn_df_file_L2),\
         'wrong data stn prob location L2'
assert os.path.exists(in_data_beta_df_file_L2),\
        'wrong data beta location L2'
assert os.path.exists(unbounded_model_dir_L2),\
        'wrong unbounded model DF location L2'
assert os.path.exists(params_file_L2),\
        'wrong params DF location L2'
assert os.path.exists(in_dfs_simulation_2),\
        'wrong simulation DF location L2'
assert os.path.exists(in_df_15min_orig),\
        'wrong orig DF location L2'
assert os.path.exists(in_lorenz_df_L2_sim),\
         'wrong Lorenz Curves Df L2'

# used for plotting and labelling
stn_list = ['EM02', 'EM03', 'EM05', 'EM06', 'EM07', 'EM08',
            'EM09', 'EM10',  'EM12', 'EM13', 'EM15', 'EM16']

# which stations are most interesting for us
wanted_stns_list = ['EM08', 'EM09', 'EM10', 'EM13', 'EM15', 'EM16']

# =============================================================================
# # define whtat to plot
# =============================================================================
rain_weights = False
weights = False
dependency = False
histogram = False
lorenz = False
cdf_max = False

plotP01Month = False
plotP01Station = False
buildDfSim = False
plotCdfSim = False
rankedHist = False
boxPlot = False
plotShiftedOrigSimVals = False
# =============================================================================
# class to break inner loop and execute outer loop when needed
# =============================================================================


class ContinueI(Exception):
    pass

continue_i = ContinueI()
# ============================================================================
# PLOT Histogram WEIGHTS and fitted BETA distribution
# =============================================================================

# def fig size, will be read in imported module to plot Latex like plots
fig_size = (12, 7)
dpi = 300
save_format = '.pdf'

font_size_title = 16
font_size_axis = 19
font_size_legend = 20

marker_size = 60

# define df_seperator
df_sep = ';'
date_format = '%Y-%m-%d %H:%M:%S'

# define transparency
transparency = 0.65

# define ratio of bars, and width decrease factor
ratio_bars = 0.04

# define if normed or not
norme_it = False

# define line width
line_width = 0.01


# =============================================================================
# create function to get files based on dir and cascade level
# =============================================================================


def getFiles(data_dir, cascade_level):
    dfs_files = []
    for r, dir_, f in os.walk(os.path.join(data_dir,
                                           r'%s' % cascade_level)):
        for fs in f:
            if fs.endswith('.csv'):
                dfs_files.append(os.path.join(r, fs))
    return dfs_files

# get files for l1
dfs_files_L1 = getFiles(in_data_dir, cascade_level_1)
# get files for l2
dfs_files_L2 = getFiles(in_data_dir, cascade_level_2)

# =============================================================================
# get Weights and do histogram and fitted beta function, all in a dict
# =============================================================================


def getHistWeights(in_dfs_data_list, in_df_beta_param, cascade_level):

    '''
    input: weights_dfs , beta_params_df, cascade_level
    output: plots of weights histogram and fitted beta pdf
    '''
    wanted_stns_data = {k: {'fct': [], 'data': []} for k in wanted_stns_list}

    # read beta pdf params results
    df_beta = pd.read_csv(in_df_beta_param, sep=df_sep, index_col=0)

    # go through file and stations
    for df_file in in_dfs_data_list:

        for station in (df_beta.index):

            for stn_wanted in wanted_stns_list:

                if station == stn_wanted and station in df_file:

                    # read each df-file
                    d = pd.read_csv(df_file, sep=df_sep, index_col=0)

                    # select stn beta distribution params
                    a = df_beta.loc[station, 'alfa']
                    b = df_beta.loc[station, 'beta']

                    for k, col in enumerate(d.columns):

                        # plot weights sub interval left W1
                        if k == 0:

                            # select weights between 0 and 1 for plot
                            for val in d[col].values:
                                if val != 0 and val <= 1e-8:  # if low vals
                                    d[col].replace(val, value=0.,
                                                   inplace=True)

                            d = d.loc[(d[col] != 0.0) & (d[col] != 1.0)]

                            # define bins nbr for histogram of weights

                            bins = np.arange(0., 1.01, 0.045)
                            center = (bins[:-1] + bins[1:]) / 2

                            # plot hist weights 0 < W < 1
                            hist, bins = np.histogram(d[col].values,
                                                      bins=bins,
                                                      normed=norme_it)

                            wanted_stns_data[station]['data'].append(
                                    [center, hist,
                                     len(d[col].index)])

                            wanted_stns_data[station]['fct'].append(
                                     [d[col].values,
                                      beta.pdf(d[col].values, a, b),
                                      (a, 0)])

    return wanted_stns_data

# call fct to get files and find W for level one and level two
dict1 = getHistWeights(dfs_files_L1, in_data_beta_df_file_L1, cascade_level_1)
dict2 = getHistWeights(dfs_files_L2, in_data_beta_df_file_L2, cascade_level_2)

# =============================================================================
# Get weights and correspondind rainfall for scatter plot, all in a dict
# =============================================================================


def dictScatterWeightsRainfall(in_df_data, cascade_level):

    wanted_stns_data = {k: {'fct': [], 'data': []} for k in wanted_stns_list}

    for stn_wanted in wanted_stns_list:
        for df_file in in_df_data:
            if stn_wanted in df_file:
                in_df = pd.read_csv(df_file, sep=df_sep, index_col=0)

                w_vals = in_df['%s Sub Int Left' % stn_wanted].values
                r_vals = in_df['%s Original Volume' % stn_wanted].values
                wanted_stns_data[stn_wanted]['fct'].append([w_vals, r_vals, 0])
    return wanted_stns_data

wr1 = dictScatterWeightsRainfall(dfs_files_L1, cascade_level_1)
wr2 = dictScatterWeightsRainfall(dfs_files_L2, cascade_level_2)

# =============================================================================
# function for plotting, based on what is defined as True
# =============================================================================


def plotdictData(data_dict, cascade_level, xlabel, ylabel, var):

    f, axes = plt.subplots(3, 2,
                           figsize=(17, 21), dpi=300,

                           sharex='col')

    for j, stn in enumerate(data_dict.keys()):

        if j <= 2:

            if histogram is False\
                    and lorenz is not True and cdf_max is False:

                    if rain_weights is True:
                        label_ = 'Sampled Weights'
                    if weights is True:
                        label_ = 'Fitted symmetric Beta distribution function'
                    if dependency is True:
                        label_ = 'Maximum likelihood model'

                    axes[j, 0].scatter(data_dict[stn]['fct'][0][0],
                                       data_dict[stn]['fct'][0][1],
                                       color='r', marker='o',
                                       alpha=0.5,
                                       label=label_)

                    axes[j, 0].set_axisbelow(True)
                    axes[j, 0].yaxis.grid(color='gray',
                                          linestyle='dashed',
                                          linewidth=line_width,
                                          alpha=0.2)
                    axes[j, 0].xaxis.grid(color='gray',
                                          linestyle='dashed',
                                          linewidth=line_width,
                                          alpha=0.2)

            if rain_weights is True:
                title = (('%s \n%s')
                         % (stn, cascade_level))
                if cascade_level == cascade_level_1:
                    if stn == 'EM08':
                        xscale, yscale = 0.8, 5.3
                    elif stn == 'EM13':
                        xscale, yscale = 0.8, 10.5
                    elif stn == 'EM09':
                        xscale, yscale = 0.8, 6.5
                    elif stn == 'EM15':
                        xscale, yscale = 0.8, 5.5
                    elif stn == 'EM10':
                        xscale, yscale = 0.8, 20.5
                    elif stn == 'EM16':
                        xscale, yscale = 0.8, 20.5
                elif cascade_level == cascade_level_2:
                    if stn == 'EM08':
                        xscale, yscale = 0.8, 4.85
                    elif stn == 'EM13':
                        xscale, yscale = 0.8, 6.9
                    elif stn == 'EM09':
                        xscale, yscale = 0.8, 5.4
                    elif stn == 'EM15':
                        xscale, yscale = 0.8, 4.3
                    elif stn == 'EM10':
                        xscale, yscale = 0.8, 20.6
                    elif stn == 'EM16':
                        xscale, yscale = 0.8, 11.5
                in_title = ''
                xscale2, yscale2 = 0.84, 4.1

            if weights is True:

                title = (('%s \n%s')
                         % (stn, cascade_level))
                in_title = (r'$\beta$=%0.2f'
                            % data_dict[stn]['fct'][0][2][0])

                xscale, yscale = 0.84, 4.5
                xscale2, yscale2 = 0.84, 4.1

                axes[j, 0].bar(data_dict[stn]['data'][0][0],
                               data_dict[stn]['data'][0][1],
                               align='center',
                               width=ratio_bars,
                               alpha=transparency,
                               linewidth=line_width,
                               color='b',
                               label='Observed weights')
                axes[j, 0].set_ylim([0, 5])
                axes[j, 0].set_yticks(np.arange(0, 5.01, 1))

            elif dependency is True:

                title = ('%s \n%s'
                         % (stn, cascade_level))

                in_title = (('a=%.2f \n'
                            'b=%.2f')
                            % (data_dict[stn]['fct'][0][2][0],
                               data_dict[stn]['fct'][0][2][1]))
#                in_title2 = (('Count of\n'
#                              '$W=0$ or\n$W=1$:\n'
#                              '%d')
#                             % ((data_dict[stn]['data'][0][2])))
#
                xscale, yscale = 0.94, 0.43
                xscale2, yscale2 = 0.94, 0.36

                axes[j, 0].scatter(data_dict[stn]['data'][0][0],
                                   data_dict[stn]['data'][0][1],
                                   color='b',
                                   marker='*',
                                   alpha=0.85,
                                   label='Observed rainfall')
                axes[j, 0].set_ylim([-0.01, 0.5])
                axes[j, 0].set_xlim([-0.5, 1.26])

            elif histogram is True:
                x_loc_ = 0.95
                if cascade_level == cascade_level_1:

                    if stn == 'EM09':
                        xscale, yscale = x_loc_, 184
                        xscale2, yscale2 = x_loc_, 179

                    elif stn == 'EM08':
                        xscale, yscale = x_loc_, 44
                        xscale2, yscale2 = x_loc_, 39

                    elif stn == 'EM13':
                        xscale, yscale = x_loc_, 122
                        xscale2, yscale2 = x_loc_, 117
                    elif stn == 'EM15':
                        xscale, yscale = x_loc_, 56
                        xscale2, yscale2 = x_loc_, 51
                    elif stn == 'EM16':
                        xscale, yscale = x_loc_, 180
                        xscale2, yscale2 = x_loc_, 175
                    elif stn == 'EM10':
                        xscale, yscale = x_loc_, 311
                        xscale2, yscale2 = x_loc_, 306

                if cascade_level == cascade_level_2:
                    if stn == 'EM08':
                        xscale, yscale = x_loc_, 55
                        xscale2, yscale2 = x_loc_, 50
                    elif stn == 'EM09':
                        xscale, yscale = x_loc_, 212
                        xscale2, yscale2 = x_loc_, 207

                    elif stn == 'EM13':
                        xscale, yscale = x_loc_, 320
                        xscale2, yscale2 = x_loc_, 315
                    elif stn == 'EM15':
                        xscale, yscale = x_loc_, 89
                        xscale2, yscale2 = x_loc_, 84
                    elif stn == 'EM16':
                        xscale, yscale = x_loc_, 577
                        xscale2, yscale2 = x_loc_, 572
                    elif stn == 'EM10':
                        xscale, yscale = x_loc_, 890
                        xscale2, yscale2 = x_loc_, 885
                title = (('%s \n%s')
                         % (stn, cascade_level))

                in_title = ''
                axes[j, 0].bar(data_dict[stn]['fct'][0][0][0],
                               data_dict[stn]['fct'][0][0][1],
                               align='center',
                               width=ratio_bars,
                               alpha=transparency,
                               linewidth=line_width,
                               color='b',
                               label='Original values')
                axes[j, 0].bar(data_dict[stn]['fct'][0][1][0],
                               data_dict[stn]['fct'][0][1][1],
                               align='center',
                               width=ratio_bars*0.6,
                               alpha=transparency*0.9,
                               linewidth=line_width,
                               color='r',
                               label='Basic model')
                axes[j, 0].bar(data_dict[stn]['fct'][0][2][0],
                               data_dict[stn]['fct'][0][2][1],
                               align='center',
                               width=ratio_bars*0.3,
                               alpha=transparency*0.8,
                               linewidth=line_width,
                               color='g',
                               label='Dependent model')

                axes[j, 0].set_axisbelow(True)
                axes[j, 0].yaxis.grid(color='gray',
                                      linestyle='dashed',
                                      linewidth=line_width,
                                      alpha=0.2)
                axes[j, 0].xaxis.grid(color='gray',
                                      linestyle='dashed',
                                      linewidth=line_width,
                                      alpha=0.2)

            elif lorenz is True:

                axes[j, 0].set_ylim([0, 1])
                xscale, yscale = 0.15, 0.9
                xscale2, yscale2 = 0.15, 0.85

                title = (('%s \n%s')
                         % (stn, cascade_level))

                in_title = ''
                axes[j, 0].scatter(data_dict[stn]['fct'][0][0][0],
                                   data_dict[stn]['fct'][0][0][1],
                                   color='b',
                                   alpha=transparency,
                                   s=marker_size,
                                   marker='o',
                                   label='Observed rainfall')
                axes[j, 0].scatter(data_dict[stn]['fct'][0][1][0],
                                   data_dict[stn]['fct'][0][1][1],
                                   color='r',
                                   s=marker_size*0.9,
                                   marker='+',
                                   alpha=transparency*0.8,
                                   label='Basic model')
                axes[j, 0].scatter(data_dict[stn]['fct'][0][2][0],
                                   data_dict[stn]['fct'][0][2][1],
                                   color='g',
                                   s=marker_size*0.9,
                                   marker='*',
                                   alpha=transparency*0.6,
                                   label='Dependent model')

                axes[j, 0].set_axisbelow(True)
                axes[j, 0].yaxis.grid(color='gray',
                                      linestyle='dashed',
                                      linewidth=line_width,
                                      alpha=0.2)
                axes[j, 0].xaxis.grid(color='gray',
                                      linestyle='dashed',
                                      linewidth=line_width,
                                      alpha=0.2)
            elif cdf_max is True:
                title = (('%s \n%s')
                         % (stn, cascade_level))

                in_title = ''
                axes[j, 0].step(data_dict[stn]['fct'][0][0][0],
                                data_dict[stn]['fct'][0][0][1],
                                color='b',
                                alpha=1,
                                label='Observed rainfall')
                axes[j, 0].step(data_dict[stn]['fct'][0][1][0],
                                data_dict[stn]['fct'][0][1][1],
                                color='r',
                                alpha=1,
                                label='Basic model')
                axes[j, 0].step(data_dict[stn]['fct'][0][2][0],
                                data_dict[stn]['fct'][0][2][1],
                                color='g',
                                alpha=1,
                                label='Dependent model')

                axes[j, 0].set_axisbelow(True)
                axes[j, 0].yaxis.grid(color='gray',
                                      linestyle='dashed',
                                      linewidth=line_width,
                                      alpha=0.2)
                axes[j, 0].xaxis.grid(color='gray',
                                      linestyle='dashed',
                                      linewidth=line_width,
                                      alpha=0.2)

                axes[j, 0].tick_params(axis='x',
                                       labelsize=font_size_axis)
                probs = data_dict[stn]['data']
                min_prob = np.min(probs)
                print(probs, 1-min_prob)
                axes[j, 0].set_ylim([0, 1])

                if cascade_level == cascade_level_1:
                    if stn == 'EM09':
                        xscale, yscale = 5.33, 0.08
                        xscale2, yscale2 = 5.332, 0.065

                    elif stn == 'EM08':
                        xscale, yscale = 5.1, 0.08
                        xscale2, yscale2 = 5.1, 0.065

                    elif stn == 'EM13':
                        xscale, yscale = 10.25, 0.08
                        xscale2, yscale2 = 10.25, 0.065
                    elif stn == 'EM15':
                        xscale, yscale = 5.525, 0.08
                        xscale2, yscale2 = 5.525, 0.065
                    elif stn == 'EM16':
                        xscale, yscale = 12.5, 0.08
                        xscale2, yscale2 = 12.5, 0.065
                    elif stn == 'EM10':
                        xscale, yscale = 21.2, 0.08
                        xscale2, yscale2 = 21.2, 0.065

                if cascade_level == cascade_level_2:
                    if stn == 'EM09':
                        xscale, yscale = 3.95, 0.08
                        xscale2, yscale2 = 3.95, 0.065
                    elif stn == 'EM08':
                        xscale, yscale = 4.65, 0.08
                        xscale2, yscale2 = 4.65, 0.065

                    elif stn == 'EM13':
                        xscale, yscale = 6.251, 0.08
                        xscale2, yscale2 = 6.251, 0.065
                    elif stn == 'EM15':
                        xscale, yscale = 4.4, 0.08
                        xscale2, yscale2 = 4.4, 0.065
                    elif stn == 'EM16':
                        xscale, yscale = 8.7, 0.08
                        xscale2, yscale2 = 8.7, 0.065
                    elif stn == 'EM10':
                        xscale, yscale = 14.35, 0.08
                        xscale2, yscale2 = 14.35, 0.065

            axes[j, 0].text(xscale, yscale,
                            title,
                            fontsize=font_size_title)

            axes[j, 0].text(xscale2, yscale2,
                            in_title,
                            fontsize=font_size_title)
            axes[j, 0].set_ylabel(ylabel,
                                  fontsize=font_size_axis)
            axes[j, 0].tick_params(axis='y',
                                   labelsize=font_size_axis)
            if j == 2:
                axes[j, 0].set_xlabel(xlabel,
                                      fontsize=font_size_axis)
                axes[j, 0].tick_params(axis='x',
                                       labelsize=font_size_axis)

# =============================================================================
#
# =============================================================================
        elif j > 2:
            # still need fixing
            if stn == 'EM13':
                k = 0
            elif stn == 'EM15':
                k = 1
            elif stn == 'EM16':
                k = 2
            axes[k, 1].set_axisbelow(True)
            axes[k, 1].yaxis.grid(color='gray',
                                  linestyle='dashed',
                                  linewidth=line_width,
                                  alpha=0.2)
            axes[k, 1].xaxis.grid(color='gray',
                                  linestyle='dashed',
                                  linewidth=line_width,
                                  alpha=0.2)
            axes[k, 1].tick_params(axis='y',
                                   labelsize=font_size_axis)

            if histogram is False\
                    and lorenz is not True and cdf_max is False:

                if rain_weights is True:
                    label_ = 'Sampled Weights'
                if weights is True:
                    label_ = 'Fitted symmetric Beta distribution function'
                if dependency is True:
                    label_ = 'Maximum likelihood model'

                axes[k, 1].scatter(data_dict[stn]['fct'][0][0],
                                   data_dict[stn]['fct'][0][1],
                                   color='r',
                                   marker='o',
                                   alpha=0.65,
                                   label=label_)
            if rain_weights is True:
                title = (('%s \n%s')
                         % (stn, cascade_level))
                in_title = ''
                if cascade_level == cascade_level_1:
                    if stn == 'EM08':
                        xscale, yscale = 0.8, 5.3
                    elif stn == 'EM13':
                        xscale, yscale = 0.8, 10.5
                    elif stn == 'EM09':
                        xscale, yscale = 0.8, 6.5
                    elif stn == 'EM15':
                        xscale, yscale = 0.8, 5.5
                    elif stn == 'EM10':
                        xscale, yscale = 0.8, 20.5
                    elif stn == 'EM16':
                        xscale, yscale = 0.8, 20.5

                elif cascade_level == cascade_level_2:
                    if stn == 'EM08':
                        xscale, yscale = 0.8, 4.85
                    elif stn == 'EM13':
                        xscale, yscale = 0.8, 6.9
                    elif stn == 'EM09':
                        xscale, yscale = 0.8, 5.4
                    elif stn == 'EM15':
                        xscale, yscale = 0.8, 4.3
                    elif stn == 'EM10':
                        xscale, yscale = 0.8, 20.6
                    elif stn == 'EM16':
                        xscale, yscale = 0.8, 11.5

            if weights is True:

                title = (('%s \n%s')
                         % (stn, cascade_level))

                in_title = (r'$\beta$=%0.2f'
                            % data_dict[stn]['fct'][0][2][0])

                xscale, yscale = 0.84, 4.5
                xscale2, yscale2 = 0.84, 4.1

                axes[k, 1].bar(data_dict[stn]['data'][0][0],
                               data_dict[stn]['data'][0][1],
                               align='center',
                               width=ratio_bars,
                               alpha=transparency,
                               linewidth=line_width,
                               color='b',
                               label='Observed weights')
                axes[k, 1].set_ylim([0, 5])
                axes[k, 1].set_yticks(np.arange(0, 5.01, 1))

            elif dependency is True:
                title = (('%s \n%s')
                         % (stn, cascade_level))

                in_title = (('a=%.2f \n'
                             'b=%.2f')
                            % (data_dict[stn]['fct'][0][2][0],
                               data_dict[stn]['fct'][0][2][1]))
                xscale, yscale = 0.94, 0.43
                xscale2, yscale2 = 0.94, 0.36
#                xscale3, yscale3 = 0.94, 0.24

                axes[k, 1].scatter(data_dict[stn]['data'][0][0],
                                   data_dict[stn]['data'][0][1],
                                   color='b',
                                   marker='*',
                                   alpha=0.85,
                                   label='Observed rainfall')
                axes[k, 1].set_ylim([-0.01, 0.5])
                axes[k, 1].set_xlim([-0.5, 1.26])

#                in_title2 = (('Count of\n'
#                              '$W=0$ or\n$W=1$:\n'
#                              '%d')
#                             % ((data_dict[stn]['data'][0][2])))
            elif histogram is True:
                x_loc_ = 0.95
                if cascade_level == cascade_level_1:

                    if stn == 'EM09':
                        xscale, yscale = x_loc_, 184
                        xscale2, yscale2 = x_loc_, 179

                    elif stn == 'EM08':
                        xscale, yscale = x_loc_, 44
                        xscale2, yscale2 = x_loc_, 39

                    elif stn == 'EM13':
                        xscale, yscale = x_loc_, 122
                        xscale2, yscale2 = x_loc_, 117
                    elif stn == 'EM15':
                        xscale, yscale = x_loc_, 56
                        xscale2, yscale2 = x_loc_, 51
                    elif stn == 'EM16':
                        xscale, yscale = x_loc_, 180
                        xscale2, yscale2 = x_loc_, 175
                    elif stn == 'EM10':
                        xscale, yscale = x_loc_, 311
                        xscale2, yscale2 = x_loc_, 306

                if cascade_level == cascade_level_2:
                    if stn == 'EM08':
                        xscale, yscale = x_loc_, 55
                        xscale2, yscale2 = x_loc_, 50
                    elif stn == 'EM09':
                        xscale, yscale = x_loc_, 212
                        xscale2, yscale2 = x_loc_, 207

                    elif stn == 'EM13':
                        xscale, yscale = x_loc_, 320
                        xscale2, yscale2 = x_loc_, 315
                    elif stn == 'EM15':
                        xscale, yscale = x_loc_, 89
                        xscale2, yscale2 = x_loc_, 84
                    elif stn == 'EM16':
                        xscale, yscale = x_loc_, 577
                        xscale2, yscale2 = x_loc_, 572
                    elif stn == 'EM10':
                        xscale, yscale = x_loc_, 890
                        xscale2, yscale2 = x_loc_, 885

                title = (('%s \n%s')
                         % (stn, cascade_level))

                in_title = ''

                axes[k, 1].bar(data_dict[stn]['fct'][0][0][0],
                               data_dict[stn]['fct'][0][0][1],
                               align='center',
                               width=ratio_bars,
                               alpha=transparency,
                               linewidth=line_width,
                               color='b',
                               label='Original values')

                axes[k, 1].bar(data_dict[stn]['fct'][0][1][0],
                               data_dict[stn]['fct'][0][1][1],
                               align='center',
                               width=ratio_bars*0.6,
                               alpha=transparency*0.9,
                               linewidth=line_width,
                               color='r',
                               label='Basic model')
                axes[k, 1].bar(data_dict[stn]['fct'][0][2][0],
                               data_dict[stn]['fct'][0][2][1],
                               align='center',
                               width=ratio_bars*0.3,
                               alpha=transparency*0.8,
                               linewidth=line_width,
                               color='g',
                               label='Dependent model')

            elif lorenz is True:

                axes[k, 1].set_ylim([0, 1])

                xscale, yscale = 0.15, 0.9
                xscale2, yscale2 = 0.15, 0.85

                title = (('%s \n%s')
                         % (stn, cascade_level))

                in_title = ''

                axes[k, 1].scatter(data_dict[stn]['fct'][0][0][0],
                                   data_dict[stn]['fct'][0][0][1],
                                   color='b',
                                   alpha=transparency,
                                   label='Observed rainfall')
                axes[k, 1].scatter(data_dict[stn]['fct'][0][1][0],
                                   data_dict[stn]['fct'][0][1][1],
                                   color='r',
                                   alpha=transparency*0.8,
                                   label='Basic model')
                axes[k, 1].scatter(data_dict[stn]['fct'][0][2][0],
                                   data_dict[stn]['fct'][0][2][1],
                                   color='g',
                                   alpha=transparency*0.6,
                                   label='Dependent model')
            elif cdf_max is True:
                title = (('%s \n%s')
                         % (stn, cascade_level))

                in_title = ''

                axes[k, 1].step(data_dict[stn]['fct'][0][0][0],
                                data_dict[stn]['fct'][0][0][1],
                                color='b',
                                alpha=1,
                                label='Observed rainfall')
                axes[k, 1].step(data_dict[stn]['fct'][0][1][0],
                                data_dict[stn]['fct'][0][1][1],
                                color='r',
                                alpha=1,
                                label='Basic model')
                axes[k, 1].step(data_dict[stn]['fct'][0][2][0],
                                data_dict[stn]['fct'][0][2][1],
                                color='g',
                                alpha=1,
                                label='Dependent model')

                axes[k, 1].tick_params(axis='x',
                                       labelsize=font_size_axis)
                probs = data_dict[stn]['data']
                min_prob = np.min(probs)
                print(probs, 1-min_prob)
                axes[k, 1].set_ylim([0, 1])

                if cascade_level == cascade_level_1:
                    if stn == 'EM09':
                        xscale, yscale = 5.33, 0.08
                        xscale2, yscale2 = 5.332, 0.065

                    elif stn == 'EM08':
                        xscale, yscale = 5.1, 0.08
                        xscale2, yscale2 = 5.1, 0.065

                    elif stn == 'EM13':
                        xscale, yscale = 10.25, 0.08
                        xscale2, yscale2 = 10.25, 0.065
                    elif stn == 'EM15':
                        xscale, yscale = 5.525, 0.08
                        xscale2, yscale2 = 5.525, 0.065
                    elif stn == 'EM16':
                        xscale, yscale = 12.5, 0.08
                        xscale2, yscale2 = 12.5, 0.065
                    elif stn == 'EM10':
                        xscale, yscale = 21.2, 0.08
                        xscale2, yscale2 = 21.2, 0.065

                if cascade_level == cascade_level_2:
                    if stn == 'EM09':
                        xscale, yscale = 3.95, 0.08
                        xscale2, yscale2 = 3.95, 0.065
                    elif stn == 'EM08':
                        xscale, yscale = 4.65, 0.08
                        xscale2, yscale2 = 4.65, 0.065

                    elif stn == 'EM13':
                        xscale, yscale = 6.251, 0.08
                        xscale2, yscale2 = 6.251, 0.065
                    elif stn == 'EM15':
                        xscale, yscale = 4.4, 0.08
                        xscale2, yscale2 = 4.4, 0.065
                    elif stn == 'EM16':
                        xscale, yscale = 8.7, 0.08
                        xscale2, yscale2 = 8.7, 0.065
                    elif stn == 'EM10':
                        xscale, yscale = 14.35, 0.08
                        xscale2, yscale2 = 14.35, 0.065

            axes[k, 1].text(xscale, yscale,
                            title,
                            fontsize=font_size_title)

            axes[k, 1].text(xscale2, yscale2,
                            in_title,
                            fontsize=font_size_title)

            if k == 2:

                axes[k, 1].set_xlabel(xlabel,
                                      fontsize=font_size_axis)
                axes[k, 1].tick_params(axis='x',
                                       labelsize=font_size_axis)

    plt.legend(bbox_to_anchor=(-1.25, -0.25, 2.25, .0502),
               ncol=4,
               fontsize=font_size_title*1.15,
               mode="expand", borderaxespad=0.)

    plt.subplots_adjust(hspace=0.25, wspace=0.25, top=0.85)
    plt.savefig(os.path.join(out_figs_dir0,
                             'all_stns' + '_' +
                             cascade_level + var +
                             save_format),
                frameon=True,
                papertype='a4',
                bbox_inches='tight')

if weights:
    plotdictData(dict1, cascade_level_1, '0 < W < 1',
                 'Probability density values', 'weights')
    plotdictData(dict2, cascade_level_2, '0 < W < 1',
                 'Probability density values', 'weights')

if rain_weights:
    plotdictData(wr1, cascade_level_1, '0 $\leq$ W $\leq$ 1',
                 'Observed Rainfall (mm/30min)', 'Rweight')
    plotdictData(wr2, cascade_level_2, '0 $\leq$ W $\leq$ 1',
                 'Observed Rainfall (mm/15min)', 'Rweight')

    print('done plotting scatter plot weights and rainfall')
# =============================================================================
# Plot Prob that P(W=0) or P(W=1) per Month
# =============================================================================


def plotProbMonth(prob_df_file1, prob_df_file2):

    out_figs_dir1 = os.path.join(out_dir,
                                 'out_subplots')
    fact = 2
    fig, (ax1, ax2) = plt.subplots(figsize=(60, 30), ncols=2,
                                   dpi=300, sharey=True)

    in_prob_df = pd.read_csv(prob_df_file1, sep=df_sep, index_col=0)
    in_prob_df = in_prob_df[in_prob_df >= 0.]
    x = np.array([(in_prob_df.index)])

    ax1.set_xticks(np.linspace(1, 12, 12))

    y_1 = np.array([(in_prob_df['P1 per Month'].values)])
    y_1 = y_1.reshape((x.shape))

    ax1.scatter(x, y_1, c='b', marker='X',
                s=marker_size,
                label=r'P($W_{1}$) = 1')

    y_0 = np.array([(in_prob_df['P0 per Month'].values)])
    y_0 = y_0.reshape((x.shape))

    ax1.scatter(x, y_0, c='r', marker='h',
                s=marker_size,
                label='P($W_{1}$) = 0')

    y_3 = np.array([(in_prob_df['P01 per month'].values)])
    y_3 = y_3.reshape((x.shape))

    ax1.scatter(x, y_3, c='darkgreen', marker='D',
                s=marker_size,
                label=r'P($W_{1}$)=0 or P($W_{1}$)=1')
    ax1.yaxis.set_ticks(np.arange(0, 0.36, 0.05))
    ax1.tick_params(axis='x', labelsize=font_size_axis*fact)
    ax1.tick_params(axis='y', labelsize=font_size_axis*fact)

    ax1.grid(color='k', linestyle='dotted', linewidth=0.1, alpha=0.5)

    ax1.set_xlabel(r'Month', fontsize=font_size_axis*2.3)
    ax1.set_ylabel('$P_{01}$', fontsize=font_size_axis*2.3)
    ax1.yaxis.labelpad = 25
    ax1.xaxis.labelpad = 25
    ax1.text(10.6, 0.34, 'Level one',
             fontsize=font_size_title*2.5)
# =============================================================================
#
# =============================================================================

    in_prob_df2 = pd.read_csv(prob_df_file2, sep=df_sep, index_col=0)
    in_prob_df2 = in_prob_df2[in_prob_df2 >= 0.]
    x2 = np.array([(in_prob_df2.index)])

    ax2.set_xticks(np.linspace(1, 12, 12))

    y_12 = np.array([(in_prob_df2['P1 per Month'].values)])
    y_12 = y_12.reshape((x2.shape))

    ax2.scatter(x2, y_12, c='b', marker='X',
                s=marker_size,
                label=r'P($W_{1}$) = 1')

    y_02 = np.array([(in_prob_df2['P0 per Month'].values)])
    y_02 = y_02.reshape((x.shape))

    ax2.scatter(x2, y_02, c='r', marker='h',
                s=marker_size,
                label='P($W_{1}$) = 0')

    y_32 = np.array([(in_prob_df2['P01 per month'].values)])
    y_32 = y_32.reshape((x2.shape))

    ax2.scatter(x2, y_32, c='darkgreen', marker='D',
                s=marker_size,
                label=r'P($W_{1}$)=0 or P($W_{1}$)=1')

    ax2.set_xlabel(r'Month', fontsize=font_size_axis*2.3)
    ax2.tick_params(axis='x', labelsize=font_size_axis*fact)
    ax2.tick_params(axis='y', labelsize=font_size_axis*fact)
    ax2.grid(color='k', linestyle='dotted', linewidth=0.1, alpha=0.5)
    ax2.text(10.6, 0.34, 'Level two',
             fontsize=font_size_title*2.5)
    ax2.yaxis.labelpad = 25
    ax2.xaxis.labelpad = 25
    plt.legend(bbox_to_anchor=(-1.05, -0.2, 2.05, .102),
               ncol=4,
               fontsize=font_size_title*3,
               mode="expand", borderaxespad=0.)

    plt.subplots_adjust(hspace=0.3, wspace=0.05, top=0.85)

    plt.savefig(os.path.join(out_figs_dir1,
                             r'P01perStation2%s' % save_format),
                frameon=True,
                papertype='a4',
                bbox_inches='tight')
    plt.close('all')
    return

# call fct level 1 and level 2
if plotP01Month:
    plotProbMonth(in_data_prob_df_file_L1, in_data_prob_df_file_L2)
    print('done plotting seasonal effect on P01')
# =============================================================================
# Plot Prob that P(W=0) or P(W=1) per Station
# =============================================================================


def probStation(prob_stn_df_file1,
                prob_stn_df_file2):

    out_figs_dir2 = os.path.join(out_dir, 'out_subplots')
    fact = 2
    fig, (ax3, ax4) = plt.subplots(figsize=(60, 30), ncols=2,
                                   dpi=300, sharey=True)

    # read prob df file and select >= 0 values
    in_df = pd.read_csv(prob_stn_df_file1, sep=df_sep, index_col=0)
    in_df = in_df[in_df >= 0.]

    # for labeling x axis by station names
    u, x = np.unique([(in_df.index)], return_inverse=True)
    alpha = 0.85

    # plot P1 values
    y_1 = np.array([(in_df['P1'].values)])
    y_1 = y_1.reshape(x.shape)
    ax3.scatter(x, y_1, c='b', marker='X',
                s=marker_size,
                label=r'P($W_{1}$) = 1', alpha=alpha)

    # plot P0 values
    y_2 = np.array([(in_df['P0'].values)])
    y_2 = y_2.reshape(x.shape)
    ax3.scatter(x, y_2, c='r', marker='h',
                s=marker_size,
                label=r'P($W_{1}$) = 0', alpha=alpha)

    # plot P01 values
    y_3 = np.array([(in_df['P01'].values)])
    y_3 = y_3.reshape(x.shape)
    ax3.scatter(x, y_3, c='darkgreen', marker='D',
                s=marker_size,
                label=r'P($W_{1}$)=0 or P($W_{1}$)=1', alpha=alpha)

    ax3.set(xticks=range(len(u)), xticklabels=u)

    ax3.yaxis.set_ticks(np.arange(0, 0.36, 0.05))
    ax3.grid(color='k', linestyle='dotted', linewidth=0.1, alpha=0.5)

    plt.setp(ax3.get_xticklabels(), rotation='horizontal',
             fontsize=font_size_axis*fact)

    plt.setp(ax3.get_yticklabels(), rotation='horizontal',
             fontsize=font_size_axis*fact)

    ax3.set_ylabel('$P_{01}$', fontsize=font_size_axis*2.3)
    ax3.set_xlabel('Station ID', fontsize=font_size_axis*2.3)

    ax3.set_ylim([0, 0.35])
    ax3.tick_params(axis='x', labelsize=font_size_axis*fact)
    ax3.tick_params(axis='y', labelsize=font_size_axis*fact)
    ax3.text(9.6, 0.325, 'Level one',
             fontsize=font_size_title*2.5)
    ax3.yaxis.labelpad = 25
    ax3.xaxis.labelpad = 25
# =============================================================================
#
# =============================================================================
    # read prob df file and select >= 0 values
    in_df2 = pd.read_csv(prob_stn_df_file2, sep=df_sep, index_col=0)
    in_df2 = in_df2[in_df2 >= 0.]

    # for labeling x axis by station names
    u2, x2 = np.unique([(in_df2.index)], return_inverse=True)

    # plot P1 values
    y_2 = np.array([(in_df2['P1'].values)])
    y_2 = y_2.reshape(x2.shape)
    ax4.scatter(x2, y_2, c='b', marker='X',
                s=marker_size,
                label=r'P($W_{1}$) = 1', alpha=alpha)

    # plot P0 values
    y_22 = np.array([(in_df2['P0'].values)])
    y_22 = y_22.reshape(x.shape)
    ax4.scatter(x2, y_22, c='r', marker='h',
                s=marker_size,
                label=r'P($W_{1}$) = 0', alpha=alpha)

    # plot P01 values
    y_32 = np.array([(in_df2['P01'].values)])
    y_32 = y_32.reshape(x2.shape)
    ax4.scatter(x2, y_32, c='darkgreen', marker='D',
                s=marker_size,
                label=r'P($W_{1}$)=0 or P($W_{1}$)=1', alpha=alpha)

    ax4.set(xticks=range(len(u2)), xticklabels=u2)

    ax4.grid(color='k', linestyle='dotted', linewidth=0.1, alpha=0.5)
    ax4.tick_params(axis='x', labelsize=font_size_axis*fact)
    ax4.tick_params(axis='y', labelsize=font_size_axis*fact)
    ax4.text(9.6, 0.325, 'Level two',
             fontsize=font_size_title*2.5)
    plt.setp(ax4.get_xticklabels(), rotation='horizontal',
             fontsize=font_size_axis*fact)

    plt.setp(ax4.get_yticklabels(), rotation='horizontal',
             fontsize=font_size_axis*fact)

    ax4.set_xlabel('Station ID', fontsize=font_size_axis*2.3)

    ax4.yaxis.labelpad = 25
    ax4.xaxis.labelpad = 25

    plt.legend(bbox_to_anchor=(-1.05, -0.2, 2.05, .102),
               ncol=4,
               fontsize=font_size_title*3,
               mode="expand", borderaxespad=0.)

    plt.subplots_adjust(hspace=0.3, wspace=0.05, top=0.85)
    plt.savefig(os.path.join(out_figs_dir2,
                             r'P01perStation%s' % save_format),
                frameon=True,
                papertype='a4',
                bbox_inches='tight')

    plt.close('all')
    return

# call fct Level 1 and Level 2
if plotP01Station:
    probStation(in_data_prob_stn_df_file_L1, in_data_prob_stn_df_file_L2)
    print('done plotting P0, P1, P01 for every station')
# =============================================================================
# PLOT Unbounded Model Volume Dependency
# =============================================================================


def getFilesfrDir(unbouded_model_dir):

    dfs_files = []
    for r, dir_, f in os.walk(os.path.join(in_data_dir,
                                           r'%s' % unbouded_model_dir)):
        for fs in f:
            if fs.endswith('.csv'):
                dfs_files.append(os.path.join(r, fs))
    return dfs_files

# read df values of R_vals, W_vals and logLikelihood vals L1
dfs_files_P01_L1 = getFilesfrDir(unbounded_model_dir_L1)

# read df values of R_vals, W_vals and logLikelihood vals L2
dfs_files_P01_L2 = getFilesfrDir(unbounded_model_dir_L2)


def volumeDependacyP01(in_df_files, in_param_file, cascade_level):

    percentile = 0.005  # divide R values into classes and fill classes with W
    min_w_nbrs = 10  # when calculating P01, min nbr of W to consider

    wanted_stns_data = {k: {'fct': [], 'data': []} for k in wanted_stns_list}

    for station in wanted_stns_list:
        for df_file in in_df_files:
            if fnmatch.fnmatch(df_file, '*.csv') and station in df_file:

                    # read df_file: R_vals, W_vals, W_vals(0_1), L(teta)
                    d = pd.read_csv(df_file, sep=df_sep, index_col=0)

                    # calculate P01 as freq from observed R, W values
                    '''
                    superimposed are the 'observed' values of P01 estimated:
                    by fitting to the observed values of W with in each third
                    percentile of R, plotted against the mean values of R
                    in these ranges.
                    '''
                    # new df to plot P01 vs log_10R
                    d_plot = pd.DataFrame(index=np.arange(0, len(d.index), 1))

                    # new cols for  R vals and W1 vals
                    d_plot['R vals'] = d['R vals']
                    d_plot['W_01'] = d['W 01']

                    # define classes min and max R values
                    r_min = min(d_plot['R vals'].values)
                    r_max = max(d_plot['R vals'].values)

                    # define classes width increase
                    k_inc = percentile * r_max

                    # find needed nbr of classes
                    nbr_classes = int(round((r_max-r_min) / (r_min*k_inc)))

                    # new dicts to hold klasses intervals
                    klasses = {}

                    # new dicts to hold klasses W values
                    klassen_w_vals = {}

                    # new dicts to hold klasses R values
                    klassen_r_vals = {}

                    # new dicts to hold klasses W01 values for P01 observed
                    w_01 = {}

                    # create new classes and lists to hold values
                    for i in range(nbr_classes+1):
                        klasses[i] = [r_min+i*k_inc*r_min,
                                      r_min+(1+i)*k_inc*r_min]
                        klassen_w_vals[i] = []
                        klassen_r_vals[i] = []
                        w_01[i] = []

                    # go through values
                    for val, w_val in zip(d_plot['R vals'].values,
                                          d_plot['W_01'].values):
                        # find Rvals and Wvals per class
                        for klass in klasses.keys():
                            # if R val is in class, append w_val r_val to class
                            if min(klasses[klass]) <= val <= \
                                    max(klasses[klass]):

                                klassen_w_vals[klass].append(w_val)
                                klassen_r_vals[klass].append(val)

                    # find P01 as frequency per class
                    for klass in klassen_w_vals.keys():

                        # if enough values per class
                        if len(klassen_w_vals[klass]) >= min_w_nbrs:

                            # if w_val = 0, w=0 or w=1 elif w_val=1 then 0<w<1
                            # this is why use P01 = 1-sum(W01)/len(W01)
                            w_01[klass].append(1-(sum(klassen_w_vals[klass]) /
                                               len(klassen_w_vals[klass])))

                            # calculate mean of rainfall values of the class
                            w_01[klass].append(np.mean(np.
                                               log10(klassen_r_vals[klass])))

                    # convert dict Class: [P01, Log(Rmean)] to df, Class as idx
                    ds = pd.DataFrame.from_dict(w_01, orient='index')

                    # count 0<w<1 for plotting it in title
                    ct = 0
                    for val in d_plot['W_01'].values:
                        if val == 0.:
                            ct += 1

                    if len(d['R vals'].values) >= min_w_nbrs:

                        # plot observed P01, x=mean(log10(R_values)), y=(P01)
                        wanted_stns_data[station]['data'].append([ds[1],
                                                                  ds[0],
                                                                  ct])

                        # read df for logRegression parameters
                        df_param = pd.read_csv(in_param_file,
                                               sep=df_sep,
                                               index_col=0)

                        # implement logRegression fct
                        def logRegression(r_vals, a, b):
                            return np.array([1 - 1 / (1 +
                                                      np.exp(-
                                                             (np.array(
                                                                 [a + b * np.
                                                                      log10
                                                                      (r_vals)]
                                                                      ))))])

                        # x values = log10 R values
                        x_vals = np.log10(d['R vals'].values)

                        # extract logRegression params from df
                        a_ = df_param.loc[station, 'a']
                        b_ = df_param.loc[station, 'b']

                        # calculate logRegression fct
                        y_vals = logRegression(d['R vals'].values, a_, b_)

                        # plot x, y values of logRegression
                        wanted_stns_data[station]['fct'].append([x_vals,
                                                                y_vals,
                                                                (a_, b_)])

    return wanted_stns_data

# call fct Level one and two
dictlg1 = volumeDependacyP01(dfs_files_P01_L1, params_file_L1, cascade_level_1)
dictlg2 = volumeDependacyP01(dfs_files_P01_L2, params_file_L2, cascade_level_2)

if dependency:
    plotdictData(dictlg1, cascade_level_1,
                 'log$_{10}$ R', 'P$_{01}$', 'dependency')

    plotdictData(dictlg2, cascade_level_2,
                 'log$_{10}$ R', 'P$_{01}$', 'dependency')
    print('done plotting the volume dependency of P01')

# =============================================================================
# Model EVALUATION
# =============================================================================

# get simulated files level one
dfs_files_sim = []
for r, dir_, f in os.walk(in_dfs_simulation):
    for fs in f:
        if fs.endswith('.csv'):
            dfs_files_sim.append(os.path.join(r, fs))

# get simulated files level two
dfs_files_sim_2 = []
for r, dir_, f in os.walk(in_dfs_simulation_2):
    for fs in f:
        if fs.endswith('.csv'):
            dfs_files_sim_2.append(os.path.join(r, fs))


def compareHistRain(in_df_orig_vals_file,
                    in_df_simulation_files,
                    cascade_level):
    '''
        input: original precipitaion values
                simuated precipitation values, baseline and unbounded
                cascade level
        output: subplot of baseline and unbounded model vs orig vals
                np.log10(values) is plotted
    '''

    wanted_stns_data = {k: {'fct': [], 'data': []} for k in wanted_stns_list}

    # define bins and centering of bars
    bins2 = np.arange(-0.6, 1.5, 0.05)
    center2 = (bins2[:-1] + bins2[1:]) / 2

    # read df orig values
    in_df_orig_vals = pd.read_csv(in_df_orig_vals_file,
                                  sep=df_sep, index_col=0)

    for station in (in_df_orig_vals.columns):

        for i, df_file in enumerate(in_df_simulation_files):

            for stn_wanted in wanted_stns_list:
                if station == stn_wanted:
                    if station in df_file:

                        # read file as dataframe
                        df_sim_vals = pd.read_csv(df_file,
                                                  sep=df_sep,
                                                  index_col=0)

                        # start going through index of station data
                        tx_int = df_sim_vals.index.intersection(
                                in_df_orig_vals[station].index)
                        df_sim_vals['orig vals'] =\
                            in_df_orig_vals[station].loc[tx_int]

                        # plot hist for original vals
                        hist0, bins = np.histogram(
                                np.log10(
                                         df_sim_vals['orig vals']
                                         .values),
                                bins=bins2, range=(-0.6, 1.5),
                                normed=norme_it)

                        # extract baseline values from simulated file
                        hist1, bins1 = np.histogram(np.log10(df_sim_vals
                                                    ['baseline rainfall %s'
                                                     % cascade_level].values),
                                                    bins=bins2,
                                                    range=(-0.6, 1.5),
                                                    normed=norme_it)

                        # extract unbounded values from simulated file
                        hist2, bins2 = np.histogram(
                                np.log10(
                                         df_sim_vals
                                         ['unbounded rainfall %s'
                                          % cascade_level]
                                         .values),
                                bins=bins2,
                                range=(-0.6, 1.5),
                                normed=norme_it)

                        wanted_stns_data[station]['fct'].append([
                                (center2,
                                 hist0),
                                (center2,
                                 hist1),
                                (center2,
                                 hist2)])

    return wanted_stns_data

# call fct
dictev1 = compareHistRain(in_df_30min_orig, dfs_files_sim, cascade_level_1)
dictev2 = compareHistRain(in_df_15min_orig, dfs_files_sim_2, cascade_level_2)

if histogram:
    plotdictData(dictev1, cascade_level_1, 'log$_{10}$ R',
                 'Frequency', 'hists')
    plotdictData(dictev2, cascade_level_2, 'log$_{10}$ R',
                 'Frequency', 'hists')
    print('done plotting the results of the simulations')

# =============================================================================
#
# =============================================================================


def buildDfSimulations(orig_vals_df, in_df_simulation, cascade_level):

    '''
    idea: build one df from all simulations for each model
    '''

    global in_df_orig, df_all_basic_sims, df_basic_sim

    # read df orig values and make index a datetime obj
    in_df_orig = pd.read_csv(orig_vals_df, sep=df_sep, index_col=0)

    in_df_orig.index = pd.to_datetime(in_df_orig.index,
                                      format=date_format)
    # df to hold original values
    df_orig_vals = pd.DataFrame(columns=in_df_orig.columns)

    for station in wanted_stns_list:
        for i, df_file in enumerate(in_df_simulation):

            if station in df_file:

                    # read simulations file
                    df_sim_vals = pd.read_csv(df_file,
                                              sep=df_sep,
                                              index_col=0)

                    df_sim_vals.index = pd.to_datetime(df_sim_vals.index,
                                                       format=date_format)
                    # intersect original with suimulations
                    idx_orig_sim = in_df_orig.index.intersection(
                            df_sim_vals.index)

                    # new df to hold sorted simulations per idx
                    df_all_basic_sims = pd.DataFrame(index=idx_orig_sim)
                    df_all_depend_sims = pd.DataFrame(index=idx_orig_sim)
                    break

        for i, df_file in enumerate(in_df_simulation):

            if station in df_file:
                # read simulations file
                df_sim_vals = pd.read_csv(df_file,
                                          sep=df_sep,
                                          index_col=0)

                df_sim_vals.index = pd.to_datetime(df_sim_vals.index,
                                                   format=date_format)
                # intersect original with suimulations
                idx_orig_sim = in_df_orig.index.intersection(
                        df_sim_vals.index)
                df_orig_vals = in_df_orig.loc[idx_orig_sim,
                                              station]
                # append results to df simulations
                df_basic_vals = df_sim_vals.loc[idx_orig_sim,
                                                'baseline rainfall %s'
                                                % cascade_level].values
                df_depdent_sim = df_sim_vals.loc[idx_orig_sim,
                                                 'unbounded rainfall %s'
                                                 % cascade_level].values

                df_all_basic_sims[i] = df_basic_vals
                df_all_depend_sims[i] = df_depdent_sim

        df_orig_vals.to_csv(os.path.join(out_figs_dir0,
                                         '%s_%s_orig_vals.csv'
                                         % (station, cascade_level)),
                            sep=df_sep, float_format='%0.2f')
        df_all_basic_sims.to_csv(os.path.join(out_figs_dir0,
                                              '%s_%s_basic_sim.csv'
                                              % (station, cascade_level)),
                                 sep=df_sep, float_format='%0.2f')
        df_all_depend_sims.to_csv(os.path.join(out_figs_dir0,
                                               '%s_%s_depend_sim.csv'
                                               % (station, cascade_level)),
                                  sep=df_sep, float_format='%0.2f')

# call function level one and level two
if buildDfSim:

    buildDfSimulations(in_df_30min_orig, dfs_files_sim, cascade_level_1)
    buildDfSimulations(in_df_15min_orig, dfs_files_sim_2, cascade_level_2)
    print('done plotting building the df of all the simulations')

df_bs = os.path.join(out_figs_dir0, 'EM10_Level one_basic_sim.csv')
df_de = os.path.join(out_figs_dir0, 'EM10_Level one_depend_sim.csv')
df_lo = os.path.join(out_figs_dir0, 'EM10_Level one_orig_vals.csv')

df_bs2 = os.path.join(out_figs_dir0, 'EM10_Level two_basic_sim.csv')
df_de2 = os.path.join(out_figs_dir0, 'EM10_Level two_depend_sim.csv')
df_lo2 = os.path.join(out_figs_dir0, 'EM10_Level two_orig_vals.csv')

assert df_bs
assert df_de
assert df_lo

assert df_bs2
assert df_de2
assert df_lo2

# =============================================================================
# start Plotting and evaluating simulations
# =============================================================================

# select one event plot
#start_date = '2015-10-20 20:00:00'
#end_date = '2015-10-22 21:00:00'

# all dates
start_date = '2012-09-23 19:30:00'
end_date = '2016-12-31 20:00:00'

# select one event plot
#start_date = '2015-04-07 18:00:00'
#end_date = '2015-04-09 21:00:00'


def sliceDF(data_frame, start, end):

    # slice data
    mask = (data_frame.index > start) &\
        (data_frame.index <= end)
    data_frame = data_frame.loc[mask]
    return data_frame


def readDFsObsSim(basic_df_vals,
                  depdt_df_vals,
                  orig_df_vals):

    _df_basic = pd.read_csv(basic_df_vals, sep=df_sep, index_col=0)
    _df_basic.index = pd.to_datetime(_df_basic.index, format=date_format)

    # read df dependent model all simulations
    _df_depdnt = pd.read_csv(depdt_df_vals,
                             sep=df_sep, index_col=0)
    _df_depdnt.index = pd.to_datetime(_df_depdnt.index,
                                      format=date_format)

    # read df original values
    _df_orig = pd.read_csv(orig_df_vals, sep=df_sep,
                           header=None, index_col=0)
    _df_orig.fillna(0.1, inplace=True)
    _df_orig.index = pd.to_datetime(_df_orig.index,
                                    format=date_format)

    return _df_basic, _df_depdnt, _df_orig


def pltCdfSimulations(in_df_basic_simulations,
                      in_df_dependt_simulations,
                      in_df_orig_vals,
                      fig_name,
                      ylabel):

    global df_sorted_basic, _df_depdnt, df_cdf,\
        df_sorted_depdnt, _df_orig, x_axis_00

    # def fig and subplots
    fig, ax = plt.subplots(figsize=(20, 10))

    # read df basic model all simulations
    _df_basic, _df_depdnt, _df_orig = readDFsObsSim(in_df_basic_simulations,
                                                    in_df_dependt_simulations,
                                                    in_df_orig_vals)

    # slice df to get one event
    _df_basic = sliceDF(_df_basic, start_date, end_date)
    _df_orig = sliceDF(_df_orig, start_date, end_date)
    _df_depdnt = sliceDF(_df_depdnt, start_date, end_date)

    # new df to hold cdf values per index per model as column
    df_cdf = pd.DataFrame(index=_df_basic.index)

    # extract original vals in a df
    orig_vals = _df_orig.values

    # def a list of nbrs and strs for desired percentiles for cdf
    percentages_nbr = [5, 25, 50, 75, 95]
    percentages_strg = ['5', '25', '50', '75', '95']
    # def list of colors
    colors = ['r', 'b', 'm', 'y', 'g']

    # go through idx of simulations and build cdf for each date
    for idx in _df_basic.index:

        # extract vals per date per model
        vals = _df_basic.loc[idx].values
        vals2 = _df_depdnt.loc[idx].values

        # go through the percentiles and build the cdf
        for percent_nbr, percent_str in zip(percentages_nbr,
                                            percentages_strg):

            df_cdf.loc[idx, '%s_percent_basic' % (percent_str)] =\
                np.percentile(vals, percent_nbr)
            df_cdf.loc[idx, '%s_percent_depdt' % (percent_str)] =\
                np.percentile(vals2, percent_nbr)

    # get idx for plotting make it numeric for plotting
    x_axis_0 = df_cdf.index
    t = x_axis_0.to_pydatetime()
    x_axis_00 = md.date2num(t)

    for percent_str, color in zip(percentages_strg,
                                  colors):
        # get values to plot
        basic_vals = df_cdf['%s_percent_basic' % percent_str].values
        dependt_vals = df_cdf['%s_percent_depdt' % percent_str].values

        ax.plot(x_axis_00, basic_vals, marker='+',
                color=color, alpha=0.7,
                label='%s_Percent_Basic_model' % (percent_str))

        ax.plot(x_axis_00, dependt_vals, marker='o',
                color=color, alpha=0.7,
                label='%s_Percent_Dependent_model' % (percent_str))

    # plot original values
    ax.plot(x_axis_00, orig_vals, marker='*',
            color='k', alpha=0.95,
            lw=2,
            label='Original_values')

    # customize plot
    ax.grid(True)
    xfmt = md.DateFormatter('%H:%M')
    ax.tick_params(labelsize=12)
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_xlabel('Time', fontsize=font_size_title)
    ax.set_ylabel(ylabel, fontsize=font_size_title)
    plt.gcf().autofmt_xdate()

    plt.legend(loc='best', fontsize=font_size_legend/1.5)
    plt.subplots_adjust(hspace=0.3, wspace=0.05, top=0.85)
    plt.savefig(os.path.join(out_figs_dir0,
                             'dpt_cdf_%s.pdf'
                             % (fig_name)),
                frameon=True,
                papertype='a4',
                bbox_inches='tight')

    return

if plotCdfSim:

    pltCdfSimulations(df_bs, df_de, df_lo, 'julyl1',
                      'Rainfall (mm/30min)')
    pltCdfSimulations(df_bs2, df_de2, df_lo2, 'july2',
                      'Rainfall (mm/15min)')
    print('done plotting the cdf of all simulations')

# =============================================================================
#
# =============================================================================


def rankz(obs, ensemble):
    ''' Parameters
    ----------
    obs : array of observations
    ensemble : array of ensemble, with the first dimension being the
        ensemble member and the remaining dimensions being identical to obs
    Returns
    -------
    histogram data for ensemble.shape[0] + 1 bins.
    The first dimension of this array is the height of
    each histogram bar, the second dimension is the histogram bins.
         '''

    obs = obs
    ensemble = ensemble

    combined = np.vstack((obs[np.newaxis], ensemble))

    # print('computing ranks')
    ranks = np.apply_along_axis(lambda x: rankdata(x, method='min'),
                                0, combined)

    # print('computing ties')
    ties = np.sum(ranks[0] == ranks[1:], axis=0)
    ranks = ranks[0]
    tie = np.unique(ties)

    for i in range(1, len(tie)):
        index = ranks[ties == tie[i]]
        ranks[ties == tie[i]] = [np.random.randint(index[j], index[j] +
                                 tie[i]+1, tie[i])
                                 [0] for j in range(len(index))]

    return np.histogram(ranks, bins=np.linspace(0.5, combined.shape[0]+0.5,
                                                combined.shape[0]+1))


def plotRankedHistsSimulations(in_df_basic_simulations,
                               in_df_dependt_simulations,
                               in_df_orig_vals,
                               figname):

    global df_sorted_basic, _df_depdnt, df_cdf,\
        df_sorted_depdnt, _df_orig, x_axis_00

    _df_basic, _df_depdnt, _df_orig = readDFsObsSim(in_df_basic_simulations,
                                                    in_df_dependt_simulations,
                                                    in_df_orig_vals)

    # slice DF
    _df_basic = sliceDF(_df_basic, start_date, end_date)
    _df_orig = sliceDF(_df_orig, start_date, end_date)
    _df_depdnt = sliceDF(_df_depdnt, start_date, end_date)

    if _df_depdnt.shape[0] != _df_orig.shape[0]:
        _df_depdnt = _df_depdnt[1:]

    if _df_basic.shape[0] != _df_orig.shape[0]:
        _df_basic = _df_basic[1:]

    obs = _df_orig.values
    ensemble = np.array([_df_basic.values]).T
    ensemble2 = np.array([_df_depdnt.values]).T

    result = rankz(obs, ensemble)
    result2 = rankz(obs, ensemble2)

    fig, ax = plt.subplots(figsize=(20, 10), dpi=dpi)

    ax.bar(range(1, ensemble.shape[0]+2), result[0],
           color='b', alpha=0.5, label='Basic Model')

    ax.bar(range(1, ensemble2.shape[0]+2), result2[0],
           color='g', alpha=0.4, label='Dependent Model')

    ax.grid(color='gray',
            linestyle='dashed',
            linewidth=line_width*0.1,
            alpha=0.2)
    ax.set_xlabel('Simulation Number',
                  fontsize=font_size_axis)
    ax.tick_params(axis='x',
                   labelsize=font_size_axis)

    ax.set_ylabel('Frequency',
                  fontsize=font_size_axis)
    ax.tick_params(axis='y',
                   labelsize=font_size_axis)

    plt.legend(loc='best', fontsize=font_size_title)

    plt.savefig(os.path.join(out_figs_dir0,
                             'EM_10_ensemble_%s.pdf' % (figname)),
                frameon=True,
                papertype='a4',
                bbox_inches='tight')

if rankedHist:
    plotRankedHistsSimulations(df_bs, df_de, df_lo, 'levelone')
    plotRankedHistsSimulations(df_bs2, df_de2, df_lo2, 'leveltwo')


# =============================================================================
#
# =============================================================================


# =============================================================================
#
# =============================================================================


def plotBoxplot(in_df_basic_simulations,
                in_df_dependt_simulations,
                in_df_orig_vals,
                figname,
                ylabel):

    f, axarr = plt.subplots(2, figsize=(20, 10), dpi=dpi,
                            sharex='col', sharey='row')
    f.tight_layout()
    f.subplots_adjust(top=1.1)

    meanlineprops = dict(linestyle='--', linewidth=2., color='purple')
    global _df_basic, _df_depdnt, _df_orig, _df_2

    # read dfs
    _df_basic, _df_depdnt, _df_orig = readDFsObsSim(in_df_basic_simulations,
                                                    in_df_dependt_simulations,
                                                    in_df_orig_vals)

    # slice df
    _df_basic = sliceDF(_df_basic, start_date, end_date)
    _df_orig = sliceDF(_df_orig, start_date, end_date)
    _df_depdnt = sliceDF(_df_depdnt, start_date, end_date)

    idx1_h = pd.Series(_df_orig.index.format())

    data_sim1 = _df_basic.values.T
    data_sim2 = _df_depdnt.values.T
    orig_data = _df_orig.values

    inter_ = np.arange(1, _df_orig.size+1)

    axarr[0].boxplot(data_sim1,
                     showmeans=True,
                     meanline=True,
                     meanprops=meanlineprops)

    axarr[0].set_title('Basic model', fontsize=font_size_title)
    axarr[0].scatter(inter_,
                     orig_data,
                     alpha=0.85,
                     marker='D',
                     color='r',
                     s=marker_size/2)
    axarr[0].yaxis.grid()
    axarr[0].set_ylabel(ylabel,
                        fontsize=font_size_title)
    axarr[0].tick_params(labelsize=font_size_title)

    axarr[1].boxplot(data_sim2,
                     meanprops=meanlineprops,
                     showmeans=True, meanline=True)

    axarr[1].set_title('Dependent model',
                       fontsize=font_size_title)

    axarr[1].scatter(inter_,
                     orig_data,
                     color='r',
                     marker='D',
                     alpha=0.85,
                     s=marker_size/2)

    axarr[1].yaxis.grid()

    axarr[1].get_xaxis().tick_bottom()
    axarr[1].get_yaxis().tick_left()

    axarr[1].set_ylabel('Rainfall (mm/15min)',
                        fontsize=font_size_title)

    axarr[1].set_xticklabels([i[-8:-3] for i in idx1_h],
                             rotation=60)

    axarr[1].tick_params(labelsize=font_size_title)

    f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    f.savefig(os.path.join(out_figs_dir0,
                           'Boxplot_all_%s.pdf'
                           % (figname)),
              bbox_inches='tight',
              frameon=True,
              papertype='a4')

    return
if boxPlot:
    plotBoxplot(df_bs, df_de, df_lo, 'julex1', 'Rainfall (mm/30min)')
    plotBoxplot(df_bs2, df_de2, df_lo2, 'julex2', 'Rainfall (mm/15min)')


def get_lag_ser(in_ser_raw, lag=0):
    in_ser = in_ser_raw.copy()
    # shift time for simulated values
    if lag < 0:
        in_ser.values[:lag] = in_ser.values[-lag:]
        in_ser.values[lag:] = np.nan
    elif lag > 0:
        in_ser.values[lag:] = in_ser.values[:-lag]
        in_ser.values[:lag] = np.nan
    return in_ser

lags = [i for i in range(-1, 2)]  # 3 shifts


def plotShiftedData(in_df_basic_simulations,
                    in_df_dependt_simulations,
                    in_df_orig_vals,
                    model_name,
                    time_shifts,
                    time):

    global _df_basic, _df_orig

    # read dfs
    _df_basic, _df_depdnt, _df_orig = readDFsObsSim(in_df_basic_simulations,
                                                    in_df_dependt_simulations,
                                                    in_df_orig_vals)

    # get the mean of all simulations
    df_mean_sim = pd.DataFrame()

    for idx in _df_basic.index:

        vals = _df_basic.loc[idx].values
        vals2 = _df_depdnt.loc[idx].values
        df_mean_sim.loc[idx, '50_percent_basic'] = np.percentile(vals, 50)
        df_mean_sim.loc[idx, '50_percent_depdnt'] = np.percentile(vals2, 50)

    # max val for [Timestamp('2013-07-24 12:15:00')]

    # shift and plot the scatter plots
    for shift in time_shifts:
        plt.figure(figsize=(10, 10), dpi=dpi)

        print('scatter plots, simulations shifted: %d' % shift)
        shifted_sr = get_lag_ser(df_mean_sim, shift)
        shifted_sr.dropna(inplace=True)
        orig_stn = _df_orig.loc[shifted_sr.index]
        orig_stn = orig_stn[orig_stn >= 0]
        orig_stn.dropna(inplace=True)

        simstn = shifted_sr.loc[orig_stn.index, '50_percent_basic']
        simstn = simstn[simstn >= 0]
        simstn.dropna(inplace=True)

        simstn2 = shifted_sr.loc[orig_stn.index, '50_percent_depdnt']
        simstn2 = simstn2[simstn2 >= 0]
        simstn2.dropna(inplace=True)

        orig_stn = orig_stn.loc[simstn.index]
        # pearson and spearman correlation
        rho1 = spr(orig_stn.values, simstn.values)[0]
        rho2 = spr(orig_stn.values, simstn2.values)[0]

        print(shift, rho1, rho2)

        plt.scatter(orig_stn,
                    simstn,
                    marker='+',
                    c='r',
                    alpha=0.5,
                    label='Basic model, Spearman correlation=%0.3f'
                    % (rho1))
        plt.scatter(orig_stn,
                    simstn2,
                    marker='o',
                    c='g',
                    alpha=0.5,
                    label='Dependent model, Spearman correlation=%0.3f'
                    % (rho2))

        _min = min(orig_stn.values.min(), simstn.values.min())
        _max = max(orig_stn.values.max(), simstn.values.max())

        plt.plot([_min, _max], [_min, _max],
                 c='b', linestyle='--',
                 alpha=0.65)

        plt.xlim(_min, _max)
        plt.ylim(_min, _max)

        plt.xlabel('Original Rainfall Values %s' % time,
                   fontsize=font_size_title-2)
        plt.ylabel('Average of all Simulated Rainfall Values %s' % time,
                   fontsize=font_size_title-2)
        plt.xticks(fontsize=font_size_axis-1)
        plt.yticks(fontsize=font_size_axis-1)
        plt.grid(color='lightgrey',
                 linestyle='--',
                 linewidth=0.01)

        plt.legend(loc='best', fontsize=font_size_legend/1.5)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(os.path.join(out_figs_dir0,
                                 (r'orig_vs_sim_shift_%d_%s.pdf')
                                 % (shift, model_name)))
        plt.close('all')

if plotShiftedOrigSimVals:

    plotShiftedData(df_bs, df_de, df_lo, 'depbasicEM10_1',
                    lags, '(mm/30min)')
    plotShiftedData(df_bs2, df_de2, df_lo2, 'depbasicEM10_2',
                    lags, '(mm/15min)')
raise Exception
# =============================================================================
#
# =============================================================================


def boundsSimulated(orig_vals_df, in_df_simulation, cascade_level):

    '''
    idea: plot all simulations bounds, min and max simulated vs orig values
    '''
    global df_baseline, df_unbounded

    # read df orig values
    in_df_orig = pd.read_csv(orig_vals_df, sep=df_sep, index_col=0)

    for i, df_file in enumerate(in_df_simulation):

        for station in wanted_stns_list:
            if station in df_file and station == 'EM10':
                fig = plt.figure(figsize=(45, 30), dpi=300, facecolor='w')
                ax = fig.add_subplot(111)

                # new df to hold all simulated values, baseline and unbounded
                df_baseline = pd.DataFrame()
                df_unbounded = pd.DataFrame()
                df_bounds = pd.DataFrame()
                df_bounds_unbounded = pd.DataFrame()

                df_sim_vals = pd.read_csv(df_file,
                                          sep=df_sep,
                                          index_col=0)

                for idx in df_sim_vals.index:
                    try:

                        # for each idx,extract all simulated values
                        df_sim_vals.loc[idx, 'orig vals'] =\
                            in_df_orig[station].loc[idx]

                        df_baseline.loc[idx, i] =\
                            df_sim_vals.loc[idx,
                                            'baseline rainfall %s'
                                            % cascade_level]

                        df_unbounded.loc[idx, i] =\
                            df_sim_vals.loc[idx,
                                            'unbounded rainfall %s'
                                            % cascade_level]

                    except KeyError as msg:
                        print(msg)
                        continue

                df_sim_vals.index = pd.to_datetime(df_sim_vals.index,
                                                   format=date_format)
                df_sim_vals = sliceDF(df_sim_vals, start_date, end_date)

                df_bounds['min baseline'] = df_baseline.min(axis=1)

                df_bounds['max baseline'] = df_baseline.max(axis=1)

                # adjust index so it is time
                df_bounds.reindex(df_baseline.index)
                df_bounds.index = pd.to_datetime(df_bounds.index,
                                                 format=date_format)
                df_bounds = sliceDF(df_bounds, start_date, end_date)

                df_bounds_unbounded['min unbounded'] = df_unbounded.min(axis=1)
                df_bounds_unbounded['max unbounded'] = df_unbounded.max(axis=1)

                df_bounds_unbounded = sliceDF(df_bounds_unbounded,
                                              start_date, end_date)
                # adjust index so it is time
                df_bounds_unbounded.reindex(df_unbounded.index)
                df_bounds_unbounded.index = pd.to_datetime(
                        df_bounds_unbounded.index,
                        format=date_format)

                x_axis2 = df_sim_vals.index
                t = x_axis2.to_pydatetime()
                x_axis2 = md.date2num(t)
#                ax.bar(x_axis2,
#                       df_sim_vals['orig vals'].values,
#                       color='b',
#
#                       width=ratio_bars*(1/0.04)*3,
#                       alpha=0.9,
#                       linewidth=line_width,
#                       label='Original values')
                ax.scatter(x_axis2,
                       df_sim_vals['orig vals'].values,
                       color='b',
                       marker='1',
                       alpha=0.7,

                       label='Original values')

                x_axis0 = df_bounds.index
                t = x_axis0.to_pydatetime()
                x_axis0 = md.date2num(t)

#                ax.bar(x_axis0,
#                       df_bounds['max baseline'].values,
#                       color='r',
#                       bottom=df_sim_vals['orig vals'].values,
#                       width=ratio_bars*(1/0.04)*2.5,
#                       alpha=0.8,
#                       label='Basic model')
                ax.scatter(x_axis0,
                       df_bounds['max baseline'].values,
                       color='r',

                       marker='*',
                       alpha=0.6,
                       label='Basic model')
                x_axis1 = df_bounds_unbounded.index
                t = x_axis1.to_pydatetime()
                x_axis1 = md.date2num(t)
#                ax.bar(x_axis1,
#                       df_bounds_unbounded['max unbounded'].values,
#                       align='center',
#                       color='g',
#                       bottom=df_sim_vals['orig vals'].values,
#                       width=ratio_bars*(1/0.04)*2,
#                       alpha=0.6,
#                       linewidth=line_width,
#                       label='Dependent model')
                ax.scatter(x_axis1,
                       df_bounds_unbounded['max unbounded'].values,

                       color='g',
                       marker='+',

                       alpha=0.6,

                       label='Dependent model')

                ax.grid(color='k', axis='both',
                        linestyle='dashdot',
                        linewidth=0.05, alpha=0.5)
                if cascade_level == cascade_level_1:
                    ylabel = 'Rainfall volume (mm / 30min)'
                elif cascade_level == cascade_level_2:
                    ylabel = 'Rainfall volume (mm / 15min)'
                ax.set_ylabel(ylabel, fontsize=42, rotation=-90)
                ax.set_xlabel('Time', fontsize=42, rotation=-90)

                plt.legend(loc=0,
                           fontsize=40)

                xfmt = md.DateFormatter('%m-%Y')
                ax.tick_params(labelsize=40)
                ax.xaxis.set_major_formatter(xfmt)
                ax.set_xticks(np.arange(x_axis1[0], x_axis1[-1]+1, 200))
                ax.xaxis.labelpad = 40
                ax.yaxis.labelpad = 45
                plt.gcf().autofmt_xdate()
                plt.xticks(rotation=-90)
                plt.yticks(rotation=-90)

                plt.savefig(os.path.join(out_figs_dir0,
                                         r'%d%s%ssim2%s' %
                                         (i, station,
                                          cascade_level,
                                          save_format)),
                            )
                plt.close('all')
                break
    return df_baseline

#boundsSimulated(in_df_30min_orig, dfs_files_sim, cascade_level_1)
#boundsSimulated(in_df_15min_orig, dfs_files_sim_2, cascade_level_2)
print('done plotting the bounds of the simulations')
#raise Exception
# =============================================================================
# LORENZ Curves
# =============================================================================

dfs_lorenz_L1_sim = []
for r, dir_, f in os.walk(in_lorenz_df_L1_sim):
    for fs in f:
        if fs.endswith('.csv'):
            dfs_lorenz_L1_sim.append(os.path.join(r, fs))

dfs_lorenz_L2_sim = []
for r, dir_, f in os.walk(in_lorenz_df_L2_sim):
    for fs in f:
        if fs.endswith('.csv'):
            dfs_lorenz_L2_sim.append(os.path.join(r, fs))

# to calculate gini coeffecient, call fct when plotting


def gini(list_of_values):
    height, area = 0, 0
    for value in list_of_values:
        height += value
        area += height - value / 2.
    fair_area = height * len(list_of_values) / 2.
    return (fair_area - area) / fair_area


def plotLorenzCurves(in_lorenz_vals_dir_sim,
                     cascade_level):
    '''
    cumulative frequency of rainy days (X)is
    plotted against associated precipitation amount (Y).
    '''
    wanted_stns_data = {k: {'fct': [], 'data': []} for k in wanted_stns_list}

    original_files = []
    baseline_files = []
    unbounded_files = []

    for in_sim_file in in_lorenz_vals_dir_sim:

        if 'Orig' in in_sim_file:
            original_files.append(in_sim_file)

        if 'baseline' in in_sim_file and 'Orig' not in in_sim_file:
            baseline_files.append(in_sim_file)

        if 'unbounded' in in_sim_file:
            unbounded_files.append(in_sim_file)

    for station in wanted_stns_list:

            for orig_file, base_file, unbound_file in\
                zip(original_files,
                    baseline_files, unbounded_files):

                if station in orig_file and\
                    station in base_file and\
                        station in unbound_file:

                    in_lorenz_vlas_df_orig = pd.read_csv(orig_file,
                                                         sep=df_sep,
                                                         index_col=0)

                    x_vals = in_lorenz_vlas_df_orig['X O']
                    y_vals = in_lorenz_vlas_df_orig['Y O']

                    in_lorenz_df_sim_base = pd.read_csv(base_file,
                                                        sep=df_sep,
                                                        index_col=0)

                    x_vals_sim = in_lorenz_df_sim_base['X']
                    y_vals_sim = in_lorenz_df_sim_base['Y']

                    in_lorenz_df_sim_unbound = pd.read_csv(unbound_file,
                                                           sep=df_sep,
                                                           index_col=0)

                    x_vals_sim_ = in_lorenz_df_sim_unbound['X']
                    y_vals_sim_ = in_lorenz_df_sim_unbound['Y']

                    wanted_stns_data[station]['fct'].append(
                            [(x_vals, y_vals),
                             (x_vals_sim, y_vals_sim),
                             (x_vals_sim_, y_vals_sim_)])

    return wanted_stns_data

# call fct level one and two
L1_orig_sim = plotLorenzCurves(dfs_lorenz_L1_sim,
                               cascade_level_1)

L2_orig_sim = plotLorenzCurves(dfs_lorenz_L2_sim,
                               cascade_level_2)
if lorenz is True:
    plotdictData(L1_orig_sim, cascade_level_1,
                 'Accumulated occurences', 'Rainfall contribution',
                 'lorenz')

    plotdictData(L2_orig_sim, cascade_level_2,
                 'Accumulated occurences', 'Rainfall contribution',
                 'lorenz')
print('done plotting the Lorenz curves')
# ==============================================================================
# find distribution of the maximums
# ==============================================================================


def probability(value, data):
    c = Counter(data)
    # returns the probability of a given number a
    return float(c[value]) / len(data)


def distMaximums(in_df_orig_file, in_df_simu_files, cascade_level):
    '''
    Idea: read observed and simulated values
        select highest 20_30 values
        see how they are distributed
        compare observed to simulated

    '''
    wanted_stns_data = {k: {'fct': [], 'data': []} for k in wanted_stns_list}

    # read df orig values
    in_df_orig = pd.read_csv(in_df_orig_file, sep=df_sep, index_col=0)
    in_df_orig.index = pd.to_datetime(in_df_orig.index, format=date_format)

    for station in wanted_stns_list:

        for i, df_file in enumerate(in_df_simu_files):

            if station in df_file:

                # new df to hold all simulated values, baseline and unbounded
                df_baseline = pd.DataFrame()
                df_unbounded = pd.DataFrame()

                df_max_baseline = pd.DataFrame()
                df_max_unbounded = pd.DataFrame()
                df_max_orig = pd.DataFrame()

                df_sim_vals = pd.read_csv(df_file,
                                          sep=df_sep,
                                          index_col=0)

                df_sim_vals.index = pd.to_datetime(df_sim_vals.index,
                                                   format=date_format)
                idx_intersct = df_sim_vals.index.intersection(
                        in_df_orig[station].index)

                df_sim_vals['orig vals'] = in_df_orig[station].loc[
                        idx_intersct]
                for idx in df_sim_vals.index:
                    try:

                        # for each idx,extract orig values
                        df_sim_vals.loc[idx, 'orig vals'] =\
                            in_df_orig[station].loc[idx]

                        # get values from each simulation
                        df_baseline.loc[idx, i] =\
                            df_sim_vals.loc[idx,
                                            'baseline rainfall %s'
                                            % cascade_level]

                        df_unbounded.loc[idx, i] =\
                            df_sim_vals.loc[idx,
                                            'unbounded rainfall %s'
                                            % cascade_level]

                    except KeyError as msg:
                        print(msg)
                        continue

                # sort values, to extract extremes
                df_max_baseline = df_baseline[i].\
                    sort_values(ascending=False,
                                kind='mergesort')
                df_max_unbounded = df_unbounded[i].\
                    sort_values(ascending=False,
                                kind='mergesort')
                df_max_orig = df_sim_vals['orig vals'].\
                    sort_values(ascending=False,
                                kind='mergesort')

                # extract extremes, what interest us : 20 vals
                y1, y2, y3 = df_max_orig[:20],\
                    df_max_baseline[:20],\
                    df_max_unbounded[:20]
                py1 = probability(y1[-1], df_max_orig)
                py2 = probability(y2[-1], df_max_baseline)
                py3 = probability(y3[-1], df_max_baseline)
                # Cumulative counts:
                x0 = np.concatenate([y1.values[::-1], y1.values[[0]]])
                y0 = (np.arange(y1.values.size+1)/len(y1.values))

                x11 = np.concatenate([y2.values[::-1], y2.values[[0]]])
                y11 = (np.arange(y2.values.size+1)/len(y2.values))

                x22 = np.concatenate([y3.values[::-1], y3.values[[0]]])
                y22 = (np.arange(y3.values.size+1)/len(y3.values))

                wanted_stns_data[station]['fct'].append(
                        [(x0, y0), (x11, y11), (x22, y22)])
                wanted_stns_data[station]['data'].append(
                        [py1, py2, py3])

    return wanted_stns_data
max_l1 = distMaximums(in_df_30min_orig, dfs_files_sim, cascade_level_1)
max_l2 = distMaximums(in_df_15min_orig, dfs_files_sim_2, cascade_level_2)

if cdf_max is True:
    plotdictData(max_l1, cascade_level_1,
                 'Rainfall Volume (mm)',
                 'Cumulative Distribution function',
                 'cdfmax')

    plotdictData(max_l2, cascade_level_2,
                 'Rainfall Volume (mm)',
                 'Cumulative Distribution function',
                 'cdfmax')


STOP = timeit.default_timer()  # Ending time
print(('\n\a\a\a Done with everything on %s. Total run time was'
       ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP-START)))
