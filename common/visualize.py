from __future__ import absolute_import, division, print_function

from copy import deepcopy
import json
import glob
import os
import time
import math
import pprint 
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('white')
sns.set_context('poster')
pp = pprint.PrettyPrinter(indent=2, width=100)
COLORS = [ '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',  '#98df8a', '#d62728' ,
           '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',  '#e377c2', '#f7b6d2' ,
           '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf',  '#9edae5', '#1f77b4']
           
BLUE     = '#1f77b4'
LBLUE    = '#aec7e8'
ORANGE   = '#ff7f0e'
LORANGE  = '#ffbb78'
RED      = '#d62728'
LRED     = '#ff9896'
GREEN    = '#2ca02c'
LGREEN   = '#98df8a'
BROWN    = '#8c564b'
LBROWN   = '#c49c94'
PINK     = '#e377c2'
LPINK    = '#f7b6d2'
PURPLE   = '#9467bd'
LPURPLE  = '#c5b0d5'
GRAY     = '#7f7f7f'
LGRAY    = '#c7c7c7'
GOLD     = '#bcbd22'
LGOLD    = '#dbdb8d'
AQUA     = '#17becf'
LAQUA    = '#9edae5'

SCORE_COLORS = {  'mrcnn_score_orig'  :  BLUE

                , 'mrcnn_score_0'     :  LORANGE
                , 'mrcnn_score_1'     :  LRED
                , 'mrcnn_score_2'     :  LGREEN
                , 'mrcnn_score_1_norm':  LBROWN
                , 'mrcnn_score_2_norm':  LPINK
                
                , 'fcn_score_0'       :  ORANGE 
                , 'fcn_score_1'       :  RED
                , 'fcn_score_2'       :  GREEN
                , 'fcn_score_1_norm'  :  BROWN
                , 'fcn_score_2_norm'  :  PINK
               }
               
               
def plot_pr_curves_by_scores_for_one_class(class_data, class_id, class_name, scores, iou = None , 
                                            ax = None , legend = 'upper right', 
                                            min_x = 0.0, max_x = 1.05, 
                                            min_y = 0.0, max_y = 1.05, labels = None):
    avg_precs = {}
    iou_thrs = {}
    score_keys = []
    iou_key = np.round(iou,2)
    
    if ax is None:
        plt.figure(figsize=(10,10))
        ax = plt.gca()

    # scores is always passed ffom plot_mAP_by_scores, so it's nver None
    # so we loop on scores instead of sorted(class_data)
    # for idx, score_key in enumerate(sorted(class_data)):
    for idx, (score_key, score_label) in enumerate(zip(scores, labels)):
        # if  scores is not None and score_key not in  scores:
            # continue        
#         print('score_key is: {:20s} iou: {:6.3f}  avg_prec: {:10.4f}'.format(score_key,  iou_key, class_data[score_key][iou_key]['avg_prec']))
        score_keys.append(score_key)
        avg_precs[score_key] = class_data[score_key][iou_key]['avg_prec']
        precisions = class_data[score_key][iou_key]['precisions']
        recalls    = class_data[score_key][iou_key]['recalls']
        label      = '{:15s}'.format(score_label)
        
        score_idx  = scores.index(score_key)
        # print('idx: ', idx, ' Score_key: ' , score_key, 'Score Index: ' , score_idx, 'color:', SCORE_COLORS[score_key])
        
        #### ax = plot_pr_curve(precisions, recalls, label= label, color=COLORS[idx*2], ax=ax)
        ax.plot(recalls, precisions, label=label,  color=SCORE_COLORS[score_key])


    ax.set_title(' {} '.format( class_name), fontsize=14)
    ax.set_xlabel('recall', fontsize= 12)
    ax.set_ylabel('precision', fontsize= 12)
    ax.tick_params(axis='both', labelsize = 16)
    ax.set_xlim([min_x,max_x])
    ax.set_ylim([min_y,max_y])
    leg = plt.legend(loc=legend,frameon=True, fontsize = 12, markerscale = 6)
    leg.set_title('IoU Thr {:.2f}'.format(iou_key),prop={'size':11})
                  
    for xval in np.linspace(0.0, 1.0, 11):
        plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed', linewidth=2)
                         
    return avg_precs               