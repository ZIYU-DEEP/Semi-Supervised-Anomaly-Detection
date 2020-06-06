import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def f_00_10(identifier):
    name_list = ['tshirt', 'trouser', 'pullover', 'dress', 'coat',
                 'sandal', 'shirt', 'sneaker', 'bag', 'boot']
    identifier_name = name_list[identifier]

    root = '/net/leksai/nips/result/fmnist'
    recall_rec = pd.read_pickle(Path(root)/'recall_df_{}_90.pkl'.format(0))['Reconstruction Model']
    recall_oc = pd.read_pickle(Path(root)/'recall_df_{}_90.pkl'.format(identifier))['One Class Model']
    ind_to_df_ind = {i: 'A/N = 0.1, Abnormal: {}'.format(i) for i in range(10)}
    ind_list_ = list(recall_oc.index)
    y_un = {k: {} for k in ind_list_}
    y_semi = {k: {} for k in ind_list_}

    for i in ind_list_:
        y_un[i] = [recall_oc.loc[i, 'A/N = 0']] * 9
        y_semi[i] = list(recall_oc[[ind_to_df_ind[i] for i in ind_list_]].loc[i, :])


    div_joint_oc = joblib.load('div_joint_oc.pkl')
    div_margin_oc = joblib.load('div_margin_oc.pkl')

    name_list = ['tshirt', 'trouser', 'pullover', 'dress', 'coat',
                 'sandal', 'shirt', 'sneaker', 'bag', 'boot']

    x_joint = {k: {} for k in ind_list_}
    x_margin = {k: {} for k in ind_list_}

    for i in ind_list_:
        name = name_list[i]
        x_joint[i] = list(div_joint_oc[name].values())
        x_margin[i] = list(div_margin_oc[name].values())

    margin_test_for_identifier = joblib.load('margin_test_for_identifier_oc.pkl')
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
        y0 = np.array(y_un[i])
        y1 = np.array(y_semi[i])
        joint_x = np.array(x_joint[i])
        margin_x = np.array(x_margin[i])

        y1 = y1[np.argsort(joint_x)]
        joint_x = np.sort(joint_x)
        

        if (margin_test_for_identifier[i]) < 10:
            axes.plot(joint_x, y1, '-o', color=sns.color_palette("Blues_r")[min(5, i)], 
                      markersize=10, alpha=0.9, label=name_list[i])

            axes.set_ylim(0, 1.1)
            axes.set_xlim(xlim_left, xlim_right)
            plt.legend()

            sns.despine()
            axes.set_ylabel('Recall')
            axes.set_xlabel('KL Divergence between Source and Target – KL(P_train || P_test)')
            axes.set_title('[Recall v.s. KL(P_train || P_test)] - Normal: {}.'.format(identifier_name, name_list[i]))
        

def f_10_20(identifier):
    name_list = ['tshirt', 'trouser', 'pullover', 'dress', 'coat',
                 'sandal', 'shirt', 'sneaker', 'bag', 'boot']
    identifier_name = name_list[identifier]

    root = '/net/leksai/nips/result/fmnist'
    recall_rec = pd.read_pickle(Path(root)/'recall_df_{}_90.pkl'.format(0))['Reconstruction Model']
    recall_oc = pd.read_pickle(Path(root)/'recall_df_{}_90.pkl'.format(identifier))['One Class Model']
    ind_to_df_ind = {i: 'A/N = 0.1, Abnormal: {}'.format(i) for i in range(10)}
    ind_list_ = list(recall_oc.index)
    y_un = {k: {} for k in ind_list_}
    y_semi = {k: {} for k in ind_list_}

    for i in ind_list_:
        y_un[i] = [recall_oc.loc[i, 'A/N = 0']] * 9
        y_semi[i] = list(recall_oc[[ind_to_df_ind[i] for i in ind_list_]].loc[i, :])


    div_joint_oc = joblib.load('div_joint_oc.pkl')
    div_margin_oc = joblib.load('div_margin_oc.pkl')

    name_list = ['tshirt', 'trouser', 'pullover', 'dress', 'coat',
                 'sandal', 'shirt', 'sneaker', 'bag', 'boot']

    x_joint = {k: {} for k in ind_list_}
    x_margin = {k: {} for k in ind_list_}

    for i in ind_list_:
        name = name_list[i]
        x_joint[i] = list(div_joint_oc[name].values())
        x_margin[i] = list(div_margin_oc[name].values())

    margin_test_for_identifier = joblib.load('margin_test_for_identifier_oc.pkl')
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
        y0 = np.array(y_un[i])
        y1 = np.array(y_semi[i])
        joint_x = np.array(x_joint[i])
        margin_x = np.array(x_margin[i])

        y1 = y1[np.argsort(joint_x)]
        joint_x = np.sort(joint_x)


        if ((margin_test_for_identifier[i]) > 10) & ((margin_test_for_identifier[i]) <= 20):
            axes.plot(joint_x, y1, '-o', color=sns.color_palette("Greens_r")[min(5, i)], 
                      markersize=10, alpha=0.9, label=name_list[i])

            axes.set_ylim(0, 1.1)
            axes.set_xlim(xlim_left, xlim_right)
            plt.legend()

            sns.despine()
            axes.set_ylabel('Recall')
            axes.set_xlabel('KL Divergence between Source and Target – KL(P_train || P_test)')
            axes.set_title('[Recall v.s. KL(P_train || P_test)] - Normal: {}.'.format(identifier_name, name_list[i]))


def f_20_30(identifier):
    name_list = ['tshirt', 'trouser', 'pullover', 'dress', 'coat',
                 'sandal', 'shirt', 'sneaker', 'bag', 'boot']
    identifier_name = name_list[identifier]

    root = '/net/leksai/nips/result/fmnist'
    recall_rec = pd.read_pickle(Path(root)/'recall_df_{}_90.pkl'.format(0))['Reconstruction Model']
    recall_oc = pd.read_pickle(Path(root)/'recall_df_{}_90.pkl'.format(identifier))['One Class Model']
    ind_to_df_ind = {i: 'A/N = 0.1, Abnormal: {}'.format(i) for i in range(10)}
    ind_list_ = list(recall_oc.index)
    y_un = {k: {} for k in ind_list_}
    y_semi = {k: {} for k in ind_list_}

    for i in ind_list_:
        y_un[i] = [recall_oc.loc[i, 'A/N = 0']] * 9
        y_semi[i] = list(recall_oc[[ind_to_df_ind[i] for i in ind_list_]].loc[i, :])


    div_joint_oc = joblib.load('div_joint_oc.pkl')
    div_margin_oc = joblib.load('div_margin_oc.pkl')

    name_list = ['tshirt', 'trouser', 'pullover', 'dress', 'coat',
                 'sandal', 'shirt', 'sneaker', 'bag', 'boot']

    x_joint = {k: {} for k in ind_list_}
    x_margin = {k: {} for k in ind_list_}

    for i in ind_list_:
        name = name_list[i]
        x_joint[i] = list(div_joint_oc[name].values())
        x_margin[i] = list(div_margin_oc[name].values())

    margin_test_for_identifier = joblib.load('margin_test_for_identifier_oc.pkl')
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
        y0 = np.array(y_un[i])
        y1 = np.array(y_semi[i])
        joint_x = np.array(x_joint[i])
        margin_x = np.array(x_margin[i])

        y1 = y1[np.argsort(joint_x)]
        joint_x = np.sort(joint_x)



        if ((margin_test_for_identifier[i]) > 20) & ((margin_test_for_identifier[i]) <= 30):
            axes.plot(joint_x, y1, '-o', color=sns.color_palette("Reds_r")[min(5, i)], 
                      markersize=10, alpha=0.9, label=name_list[i])

            axes.set_ylim(0, 1.1)
            axes.set_xlim(xlim_left, xlim_right)
            plt.legend()

            sns.despine()
            axes.set_ylabel('Recall')
            axes.set_xlabel('KL Divergence between Source and Target – KL(P_train || P_test)')
            axes.set_title('[Recall v.s. KL(P_train || P_test)] - Normal: {}.'.format(identifier_name, name_list[i]))

       

def f_30_40(identifier):
    name_list = ['tshirt', 'trouser', 'pullover', 'dress', 'coat',
                 'sandal', 'shirt', 'sneaker', 'bag', 'boot']
    identifier_name = name_list[identifier]

    root = '/net/leksai/nips/result/fmnist'
    recall_rec = pd.read_pickle(Path(root)/'recall_df_{}_90.pkl'.format(0))['Reconstruction Model']
    recall_oc = pd.read_pickle(Path(root)/'recall_df_{}_90.pkl'.format(identifier))['One Class Model']
    ind_to_df_ind = {i: 'A/N = 0.1, Abnormal: {}'.format(i) for i in range(10)}
    ind_list_ = list(recall_oc.index)
    y_un = {k: {} for k in ind_list_}
    y_semi = {k: {} for k in ind_list_}

    for i in ind_list_:
        y_un[i] = [recall_oc.loc[i, 'A/N = 0']] * 9
        y_semi[i] = list(recall_oc[[ind_to_df_ind[i] for i in ind_list_]].loc[i, :])


    div_joint_oc = joblib.load('div_joint_oc.pkl')
    div_margin_oc = joblib.load('div_margin_oc.pkl')

    name_list = ['tshirt', 'trouser', 'pullover', 'dress', 'coat',
                 'sandal', 'shirt', 'sneaker', 'bag', 'boot']

    x_joint = {k: {} for k in ind_list_}
    x_margin = {k: {} for k in ind_list_}

    for i in ind_list_:
        name = name_list[i]
        x_joint[i] = list(div_joint_oc[name].values())
        x_margin[i] = list(div_margin_oc[name].values())

    margin_test_for_identifier = joblib.load('margin_test_for_identifier_oc.pkl')
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
        y0 = np.array(y_un[i])
        y1 = np.array(y_semi[i])
        joint_x = np.array(x_joint[i])
        margin_x = np.array(x_margin[i])
        
        y1 = y1[np.argsort(joint_x)]
        joint_x = np.sort(joint_x)

        if ((margin_test_for_identifier[i]) > 30) & ((margin_test_for_identifier[i]) <= 40):
            axes.plot(joint_x, y1, '-o', color=sns.color_palette("cool")[min(5, i)], 
                      markersize=10, alpha=0.9, label=name_list[i])

            axes.set_ylim(0, 1.1)
            axes.set_xlim(xlim_left, xlim_right)
            plt.legend()

            sns.despine()
            axes.set_ylabel('Recall')
            axes.set_xlabel('KL Divergence between Source and Target – KL(P_train || P_test)')
            axes.set_title('[Recall v.s. KL(P_train || P_test)] - Normal: {}.'.format(identifier_name, name_list[i]))


def f_40_50(identifier):
    name_list = ['tshirt', 'trouser', 'pullover', 'dress', 'coat',
                 'sandal', 'shirt', 'sneaker', 'bag', 'boot']
    identifier_name = name_list[identifier]

    root = '/net/leksai/nips/result/fmnist'
    recall_rec = pd.read_pickle(Path(root)/'recall_df_{}_90.pkl'.format(0))['Reconstruction Model']
    recall_oc = pd.read_pickle(Path(root)/'recall_df_{}_90.pkl'.format(identifier))['One Class Model']
    ind_to_df_ind = {i: 'A/N = 0.1, Abnormal: {}'.format(i) for i in range(10)}
    ind_list_ = list(recall_oc.index)
    y_un = {k: {} for k in ind_list_}
    y_semi = {k: {} for k in ind_list_}

    for i in ind_list_:
        y_un[i] = [recall_oc.loc[i, 'A/N = 0']] * 9
        y_semi[i] = list(recall_oc[[ind_to_df_ind[i] for i in ind_list_]].loc[i, :])


    div_joint_oc = joblib.load('div_joint_oc.pkl')
    div_margin_oc = joblib.load('div_margin_oc.pkl')

    name_list = ['tshirt', 'trouser', 'pullover', 'dress', 'coat',
                 'sandal', 'shirt', 'sneaker', 'bag', 'boot']

    x_joint = {k: {} for k in ind_list_}
    x_margin = {k: {} for k in ind_list_}

    for i in ind_list_:
        name = name_list[i]
        x_joint[i] = list(div_joint_oc[name].values())
        x_margin[i] = list(div_margin_oc[name].values())

    margin_test_for_identifier = joblib.load('margin_test_for_identifier_oc.pkl')
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
        y0 = np.array(y_un[i])
        y1 = np.array(y_semi[i])
        joint_x = np.array(x_joint[i])
        margin_x = np.array(x_margin[i])

        y1 = y1[np.argsort(joint_x)]
        joint_x = np.sort(joint_x)

        if ((margin_test_for_identifier[i]) > 40) & ((margin_test_for_identifier[i]) <= 50):
            axes.plot(joint_x, y1, '-o', color=sns.color_palette("gray_r")[min(5, i)], 
                      markersize=10, alpha=0.9, label=name_list[i])

            axes.set_ylim(0, 1.1)
            axes.set_xlim(xlim_left, xlim_right)
            plt.legend()

            sns.despine()
            axes.set_ylabel('Recall')
            axes.set_xlabel('KL Divergence between Source and Target – KL(P_train || P_test)')
            axes.set_title('[Recall v.s. KL(P_train || P_test)] - Normal: {}.'.format(identifier_name, name_list[i]))



def f_50_00(identifier):
    name_list = ['tshirt', 'trouser', 'pullover', 'dress', 'coat',
                 'sandal', 'shirt', 'sneaker', 'bag', 'boot']
    identifier_name = name_list[identifier]

    root = '/net/leksai/nips/result/fmnist'
    recall_rec = pd.read_pickle(Path(root)/'recall_df_{}_90.pkl'.format(0))['Reconstruction Model']
    recall_oc = pd.read_pickle(Path(root)/'recall_df_{}_90.pkl'.format(identifier))['One Class Model']
    ind_to_df_ind = {i: 'A/N = 0.1, Abnormal: {}'.format(i) for i in range(10)}
    ind_list_ = list(recall_oc.index)
    y_un = {k: {} for k in ind_list_}
    y_semi = {k: {} for k in ind_list_}

    for i in ind_list_:
        y_un[i] = [recall_oc.loc[i, 'A/N = 0']] * 9
        y_semi[i] = list(recall_oc[[ind_to_df_ind[i] for i in ind_list_]].loc[i, :])


    div_joint_oc = joblib.load('div_joint_oc.pkl')
    div_margin_oc = joblib.load('div_margin_oc.pkl')

    name_list = ['tshirt', 'trouser', 'pullover', 'dress', 'coat',
                 'sandal', 'shirt', 'sneaker', 'bag', 'boot']

    x_joint = {k: {} for k in ind_list_}
    x_margin = {k: {} for k in ind_list_}

    for i in ind_list_:
        name = name_list[i]
        x_joint[i] = list(div_joint_oc[name].values())
        x_margin[i] = list(div_margin_oc[name].values())

    margin_test_for_identifier = joblib.load('margin_test_for_identifier_oc.pkl')
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
        y0 = np.array(y_un[i])
        y1 = np.array(y_semi[i])
        joint_x = np.array(x_joint[i])
        margin_x = np.array(x_margin[i])

        y1 = y1[np.argsort(joint_x)]
        joint_x = np.sort(joint_x)

        if ((margin_test_for_identifier[i]) > 50):
            axes.plot(joint_x, y1, '-o', color=sns.color_palette("twilight")[min(5, i)], 
                      markersize=10, alpha=0.9, label=name_list[i])


            axes.set_ylim(0, 1.1)
            axes.set_xlim(xlim_left, xlim_right)
            plt.legend()

            sns.despine()
            axes.set_ylabel('Recall')
            axes.set_xlabel('KL Divergence between Source and Target – KL(P_train || P_test)')
            axes.set_title('[Recall v.s. KL(P_train || P_test)] - Normal: {}.'.format(identifier_name, name_list[i]))
