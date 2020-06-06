import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def rec_plot_info(x, y1, color_start, name_list, i, axes, identifier_name, metric, goal, range_str):
    """
    Fixed format for plot drawing.
    """

    axes.plot(x, y1, 
              '-o', 
              color=sns.cubehelix_palette(10, 
                                          start=color_start, 
                                          rot=0, 
                                          dark=0.3, 
                                          light=.85, 
                                          reverse=True)[i], 
              markersize=10, 
              alpha=0.9, 
              label=name_list[i])
    
    axes.set_ylim(0, 1.1)
    plt.legend()
    sns.despine()
    axes.set_ylabel('{}'.format(metric))
    
    if goal == 'joint':
        axes.set_xlim(-2, 32)
        axes.set_xlabel('KL Divergence between Source and Target – KL(P_train || P_test)')
        axes.set_title('[{} v.s. KL(P_train || P_test)];       Normal: {};  Bucket: {}.'.format(metric, identifier_name, range_str))
    
    elif goal == 'division':
#         axes.set_xlim(-2, 32)
        axes.set_xlabel('KL(P_1 || P_i) / KL(P_1, P_0)')
        axes.set_title('[{} v.s. KL(P_1 || P_i) / KL(P_1, P_0)];       Normal: {};  Bucket: {}.'.format(metric, identifier_name, range_str))

    elif goal == 'triangle':
#         axes.set_xlim(-2, 32)
        axes.set_xlabel('KL(P_0 || P_1) + KL(P`_0 || P_i) - KL(P_1 || P_i)')
        axes.set_title('[{} v.s. KL(P_0 || P_1) + KL(P`_0 || P_i) - KL(P_1 || P_i)];       Normal: {};  Bucket: {}.'.format(metric, identifier_name, range_str))
        

def rec_line(identifier, goal='triangle', metric='f1', range_='00_10'):
    name_list = ['tshirt', 'trouser', 'pullover', 'dress', 'coat',
                 'sandal', 'shirt', 'sneaker', 'bag', 'boot']
    identifier_name = name_list[identifier]

    root = '/net/leksai/nips/result/fmnist'
    recall_rec = pd.read_pickle(Path(root)/'recall_df_{}_90.pkl'.format(identifier))['Reconstruction Model']
    ind_to_df_ind = {i: 'A/N = 0.1, Abnormal: {}'.format(i) for i in range(10)}

    ind_list_ = list(recall_rec.index)
    y_un = {k: {} for k in ind_list_}
    y_semi = {k: {} for k in ind_list_}

    for i in ind_list_:
        y_un[i] = [recall_rec.loc[i, 'A/N = 0']] * 9
        y_semi[i] = list(recall_rec[[ind_to_df_ind[i] for i in ind_list_]].loc[i, :])

    div_joint_rec = joblib.load('div_joint.pkl')
    div_margin_rec = joblib.load('div_margin.pkl')
    div_p1_pi_rec = joblib.load('div_margin_train.pkl')

    name_list = ['tshirt', 'trouser', 'pullover', 'dress', 'coat',
                 'sandal', 'shirt', 'sneaker', 'bag', 'boot']

    x_joint = {k: {} for k in ind_list_}
    x_margin = {k: {} for k in ind_list_}
    x_p1_pi = {k: {} for k in ind_list_}

    for i in ind_list_:
        name = name_list[i]
        x_joint[i] = list(div_joint_rec[name].values())
        x_margin[i] = list(div_margin_rec[name].values())
        x_p1_pi[i] = list(div_p1_pi_rec[name].values())

    margin_test_for_identifier = joblib.load('margin_test_for_identifier.pkl')
    i_list = list(np.argsort(margin_test_for_identifier))
    i_list.remove(identifier)

    temp = []
    for i in i_list:
        temp.extend(x_joint[i])
    xlim_left = min(temp) - 2
    xlim_right = max(temp) + 2

    n = len(i_list)
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(18, 9))
    plt.subplots_adjust(hspace=0.3)

    for ind, i in enumerate(i_list):
        recall_0 = np.array(y_un[i])
        precision_0 = 1000 * recall_0 / (1000 * recall_0 + 100)
        f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0)
        
        recall_1 = np.array(y_semi[i])
        precision_1 = 1000 * recall_1 / (1000 * recall_1 + 100)
        f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1)
        
        if metric == 'f1':
            y0 = f1_0
            y1 = f1_1
        
        if metric == 'recall':
            y0 = recall_0
            y1 = recall_1
        
        joint_x = np.array(x_joint[i])
        margin_x = np.array(x_margin[i])
        p1_pi_x = np.array(x_p1_pi[i])
        p1_pi__p0_p1_x = p1_pi_x / margin_x
        p0_p1__p0_pi__p1_pi_x = margin_x + margin_test_for_identifier[i]- p1_pi_x

        
        if goal == 'joint':
            y1 = y1[np.argsort(joint_x)]
            x = np.sort(joint_x)

        if goal == 'division':
            y1 = y1[np.argsort(p1_pi__p0_p1_x)]
            x = np.sort(p1_pi__p0_p1_x)
        
        if goal == 'triangle':
            y1 = y1[np.argsort(p0_p1__p0_pi__p1_pi_x)]
            x = np.sort(p0_p1__p0_pi__p1_pi_x)
            
        if range_ == '00_10':
            if (margin_test_for_identifier[i]) < 10:
                rec_plot_info(x, y1, 2.6, name_list, i, axes, identifier_name,  
                              metric, goal, 'KL(P`_0 || P_1) <= 10')
                
        elif range_ == '10_20':
            if ((margin_test_for_identifier[i]) > 10) & ((margin_test_for_identifier[i]) <= 20):
                rec_plot_info(x, y1, 2, name_list, i, axes, identifier_name, 
                              metric, goal, '10 <= KL(P`_0 || P_1) <= 20')
        
        elif range_ == '20_30':
            if ((margin_test_for_identifier[i]) > 20) & ((margin_test_for_identifier[i]) <= 30):
                rec_plot_info(x, y1, 1, name_list, i, axes, identifier_name, 
                              metric, goal, '20 <= KL(P`_0 || P_1) <= 30')
                
        elif range_ == '30_40':
            if ((margin_test_for_identifier[i]) > 30) & ((margin_test_for_identifier[i]) <= 40):
                rec_plot_info(x, y1, 0.4, name_list, i, axes, identifier_name, 
                              metric, goal, '30 <= KL(P`_0 || P_1) <= 40')
        
        elif range_ == '40_50':
            if ((margin_test_for_identifier[i]) > 40) & ((margin_test_for_identifier[i]) <= 50):
                rec_plot_info(x, y1, 0, name_list, i, axes, identifier_name, 
                              metric, goal, '40 <= KL(P`_0 || P_1) <= 50')

        elif range_ == '50_00':
            if (margin_test_for_identifier[i]) > 50:
                rec_plot_info(x, y1, 1.3, name_list, i, axes, identifier_name, 
                              metric, goal, 'KL(P`_0 || P_1) >= 50')


def show_division(identifier, metric='f1'):
    rec_line(identifier, metric=metric, range_='00_10', goal='division')
    rec_line(identifier, metric=metric, range_='10_20', goal='division')
    rec_line(identifier, metric=metric, range_='20_30', goal='division')
    rec_line(identifier, metric=metric, range_='30_40', goal='division')
    rec_line(identifier, metric=metric, range_='40_50', goal='division')
    rec_line(identifier, metric=metric, range_='50_00', goal='division')
    

def show_triangle(identifier, metric='f1'):
    rec_line(identifier, metric=metric, range_='00_10')
    rec_line(identifier, metric=metric, range_='10_20')
    rec_line(identifier, metric=metric, range_='20_30')
    rec_line(identifier, metric=metric, range_='30_40')
    rec_line(identifier, metric=metric, range_='40_50')
    rec_line(identifier, metric=metric, range_='50_00')