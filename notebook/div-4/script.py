import sys
sys.path.append('../../dataset/')
sys.path.append('../../network/')
sys.path.append('../../model/')

import os
import glob
import time
import torch
import joblib
import logging
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import torch
import seaborn as sns

from pathlib import Path
from main_loading import *
from main_network import *
from main_model_rec import *
from main_model_one_class import *
from scipy.spatial import KDTree
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt


identifier = 4
device = 'cuda:1'
root = '/net/leksai/data/FashionMNIST'
rec_model_path = '/net/leksai/nips/model/rec/fmnist/rec_unsupervised_[{}]_[]_[0.0]/net_fmnist_LeNet_rec_eta_100_epochs_150_batch_128/model.tar'.format(identifier)
oc_model_path = '/net/leksai/nips/model/one_class/fmnist/one_class_unsupervised_[{}]_[]_[1]_[0.0]/net_fmnist_LeNet_one_class_eta_100_epochs_150_batch_128/model.tar'.format(identifier)

class OneClassEncoder:
    def __init__(self):
        self.net = None
        self.net_name = None

    def set_network(self, net_name):
        self.net_name = net_name
        self.net = build_network(net_name)

    def load_model(self, model_path, map_location):
        model_dict = torch.load(model_path, map_location=map_location)
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])

    def test(self, train, dataset, device, batch_size, n_jobs_dataloader):
        if train:
            all_loader, _ = dataset.loaders(batch_size=batch_size,
                                            num_workers=n_jobs_dataloader)
        else:
            all_loader = dataset.loaders(batch_size=batch_size,
                                         num_workers=n_jobs_dataloader)
        net = self.net.to(device)
        criterion = nn.MSELoss(reduction='none')

        n_batches = 0
        X_pred_list = []
        net.eval()

        with torch.no_grad():
            for data in all_loader:
                X, y, idx = data
                X, y, idx = X.to(device), y.to(device), idx.to(device)

                X_pred = net(X)
                X_pred_list += X_pred

        return np.array(X_pred_list)


oc_encoder = OneClassEncoder()
oc_encoder.set_network('fmnist_LeNet_one_class')
oc_encoder.load_model(oc_model_path, device)


class RecEncoder:
    def __init__(self):

        self.net_name = None
        self.net = None
        self.ae_net = None


    def set_network(self, net_name: str='fmnist_LeNet_one_class'):
        """
        Set the network structure for the model.
        The key here is to initialize <self.net>.
        """
        self.net_name = net_name
        self.net = build_network(net_name)
        self.ae_net = build_network('fmnist_LeNet_rec')

    def load_model(self,
                   model_path,
                   map_location='cuda:1'):
        """
        Load the trained model for the model.
        The key here is to initialize <self.c>.
        """
        # Load the general model
        model_dict = torch.load(model_path, map_location=map_location)
        self.ae_net.load_state_dict(model_dict['net_dict'])

        # Obtain the net dictionary
        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}

        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)

        # Load the new state_dict
        self.net.load_state_dict(net_dict)


    def save_model(self, export_model, save_ae=True):
        net_dict = self.net.state_dict()
        torch.save({'net_dict': net_dict}, export_model)

    def test(self, train, dataset, device, batch_size, n_jobs_dataloader):
        if train:
            all_loader, _ = dataset.loaders(batch_size=batch_size,
                                            num_workers=n_jobs_dataloader)
        else:
            all_loader = dataset.loaders(batch_size=batch_size,
                                         num_workers=n_jobs_dataloader)
        net = self.net.to(device)
        criterion = nn.MSELoss(reduction='none')

        n_batches = 0
        X_pred_list = []
        net.eval()

        with torch.no_grad():
            for data in all_loader:
                X, y, idx = data
                X, y, idx = X.to(device), y.to(device), idx.to(device)

                X_pred = net(X)
                X_pred_list += X_pred

        return np.array(X_pred_list)


rec_encoder = RecEncoder()
rec_encoder.set_network()
rec_encoder.load_model(rec_model_path, device)

dataset_dict_train = {}
dataset_dict_test = {}
name_list = ['tshirt', 'trouser', 'pullover', 'dress', 'coat',
             'sandal', 'shirt', 'sneaker', 'bag', 'boot']

for i, name in enumerate(name_list):
    dataset_dict_train[name] = load_dataset(loader_name='fmnist',
                                            root=root,
                                            label_normal=(i,),
                                            ratio_abnormal=0)


for i, name in enumerate(name_list):
    dataset_dict_test[name] = load_dataset(loader_name='fmnist_eval',
                                            root=root,
                                            label_eval=(i,),
                                            test_eval=True)

latent_dict_train = {'oc':{}, 'rec':{}}
latent_dict_test = {'oc':{}, 'rec':{}}

for name in name_list:
    dataset_train = dataset_dict_train[name]
    data_train = oc_encoder.test(True, dataset_train, device, 6000, 0)
    data_train = np.array([x.cpu().numpy() for x in data_train])
    latent_dict_train['oc'][name] = data_train


for name in name_list:
    dataset_test = dataset_dict_test[name]
    data_test = oc_encoder.test(False, dataset_test, device, 1000, 0)
    data_test = np.array([x.cpu().numpy() for x in data_test])
    latent_dict_test['oc'][name] = data_test

for name in name_list:
    dataset_train = dataset_dict_train[name]
    data_train = rec_encoder.test(True, dataset_train, device, 6000, 0)
    data_train = np.array([x.cpu().numpy() for x in data_train])
    latent_dict_train['rec'][name] = data_train


for name in name_list:
    dataset_test = dataset_dict_test[name]
    data_test = rec_encoder.test(False, dataset_test, device, 1000, 0)
    data_test = np.array([x.cpu().numpy() for x in data_test])
    latent_dict_test['rec'][name] = data_test

from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree

def knn_distance(point, sample, k):
    """
    Euclidean distance from `point` to it's `k`-Nearest
    Neighbour in `sample`
    """
    norms = np.linalg.norm(sample-point, axis=1)
    return np.sort(norms)[k]


def verify_sample_shapes(s1, s2, k):
    # Expects [N, D]
    assert(len(s1.shape) == len(s2.shape) == 2)
    # Check dimensionality of sample is identical
    assert(s1.shape[1] == s2.shape[1])


def skl_estimator(s1, s2, k=3):
    """
    KL-Divergence estimator using scikit-learn's NearestNeighbours.
    Inputs:
        s1: (N_1,D) Sample drawn from distribution P
        s2: (N_2,D) Sample drawn from distribution Q
        k: Number of neighbours considered (default 1)
    return:
        estimated D(P|Q)
    """
    verify_sample_shapes(s1, s2, k)

    n, m = len(s1), len(s2)
    d = float(s1.shape[1])
    D = np.log(m / (n - 1))

    s1_neighbourhood = NearestNeighbors(k + 1, 10).fit(s1)
    s2_neighbourhood = NearestNeighbors(k, 10).fit(s2)

    for p1 in s1:
        s1_distances, indices = s1_neighbourhood.kneighbors([p1], k + 1)
        s2_distances, indices = s2_neighbourhood.kneighbors([p1], k)
        rho = s1_distances[0][- 1]
        nu = s2_distances[0][- 1]
        D += (d / n) * np.log(nu / rho)
        D += 0
    return D


name_list_ = name_list[:]
identifier_name = name_list_.pop(identifier)
div_joint = {k: {} for k in name_list_}
div_margin = {k: {} for k in name_list_}
div_margin_train = {k: {} for k in name_list_}
# Fix normal train, normal test.
normal_train = latent_dict_train['rec'][identifier_name]
normal_test = latent_dict_test['rec'][identifier_name]

for test_j in name_list_:
    print('Outer loop', test_j)
    # Fix abnormal test
    abnormal_test = latent_dict_test['rec'][test_j]
    div_joint_test = div_joint[test_j]
    div_margin_test = div_margin[test_j]

    for i in name_list_:
        print(i)
        abnormal_train_i = latent_dict_train['rec'][i]

        # Calculating for joint divergence
        left_joint = np.r_[normal_train, abnormal_train_i]
        right_joint = np.r_[normal_test, abnormal_test]
        div_joint_test[i] = .5 * (skl_estimator(left_joint, right_joint) +
                                  skl_estimator(right_joint, left_joint))
        print('KL joint:', div_joint_test[i])

        # Calculating for marginal divergence
        left_margin = normal_train
        right_margin = abnormal_train_i
        div_margin_test[i] = .5 * (skl_estimator(left_margin, right_margin) +
                                   skl_estimator(right_margin, left_margin))
        print('KL marginal:', div_margin_test[i])

joblib.dump(div_joint, 'div_joint.pkl')
joblib.dump(div_margin, 'div_margin.pkl')

for test_j in name_list_:
    print('Outer loop:', test_j)
    # Fix abnormal test
    abnormal_test = latent_dict_test['rec'][test_j]
    div_margin_train_test = div_margin_train[test_j]

    for i in name_list_:
        print(i)
        abnormal_train_i = latent_dict_train['rec'][i]

        # Calculating for marginal divergence
        left = abnormal_train_i
        right = abnormal_test
        div_margin_train_test[i] = .5 * (skl_estimator(left, right) +
                                         skl_estimator(right, left))
        print('KL marginal:', div_margin_train_test[i])

joblib.dump(div_margin_train, 'div_margin_train.pkl')
margin_test_for_identifier = []
normal_test = latent_dict_test['rec'][identifier_name]
for test_j in name_list:
    print(test_j)
    # Fix abnormal test
    abnormal_test = latent_dict_test['rec'][test_j]
    left = normal_test
    right = abnormal_test
    div = .5 * (skl_estimator(left, right) +
                skl_estimator(right, left))
    margin_test_for_identifier.append(div)
    print(div)

joblib.dump(margin_test_for_identifier, 'margin_test_for_identifier.pkl')

div_joint_oc = {k: {} for k in name_list_}
div_margin_oc = {k: {} for k in name_list_}

# Fix normal train, normal test.
normal_train = latent_dict_train['oc'][identifier_name]
normal_test = latent_dict_test['oc'][identifier_name]
for test_j in name_list_:
    print('Outer loop', test_j)
    # Fix abnormal test
    abnormal_test = latent_dict_test['oc'][test_j]
    div_joint_test_oc = div_joint_oc[test_j]
    div_margin_test_oc = div_margin_oc[test_j]

    for i in name_list_:
        print(i)
        abnormal_train_i = latent_dict_train['oc'][i]

        # Calculating for joint divergence
        left_joint = np.r_[normal_train, abnormal_train_i]
        right_joint = np.r_[normal_test, abnormal_test]
        div_joint_test_oc[i] = .5 * (skl_estimator(left_joint, right_joint) +
                                     skl_estimator(right_joint, left_joint))
        print('KL joint:', div_joint_test_oc[i])

        # Calculating for marginal divergence
        left_margin = normal_train
        right_margin = abnormal_train_i
        div_margin_test_oc[i] = .5 * (skl_estimator(left_margin, right_margin) +
                                      skl_estimator(right_margin, left_margin))
        print('KL marginal:', div_margin_test_oc[i])
joblib.dump(div_joint_oc, 'div_joint_oc.pkl')
joblib.dump(div_margin_oc, 'div_margin_oc.pkl')
margin_test_for_identifier_oc = []
normal_test = latent_dict_test['oc'][identifier_name]

for test_j in name_list:
    print(test_j)
    # Fix abnormal test
    abnormal_test = latent_dict_test['oc'][test_j]
    left = normal_test
    right = abnormal_test
    div = .5 * (skl_estimator(left, right) +
                skl_estimator(right, left))
    margin_test_for_identifier_oc.append(div)
    print(div)
joblib.dump(margin_test_for_identifier_oc, 'margin_test_for_identifier_oc.pkl')
div_margin_train_oc = {k: {} for k in name_list_}
for test_j in name_list_:
    print('Outer loop:', test_j)
    # Fix abnormal test
    abnormal_test = latent_dict_test['oc'][test_j]
    div_margin_train_test_oc = div_margin_train_oc[test_j]

    for i in name_list_:
        print(i)
        abnormal_train_i = latent_dict_train['oc'][i]

        # Calculating for marginal divergence
        left = abnormal_train_i
        right = abnormal_test
        div_margin_train_test_oc[i] = .5 * (skl_estimator(left, right) +
                                            skl_estimator(right, left))
        print('KL marginal:', div_margin_train_test_oc[i])
joblib.dump(div_margin_train_oc, 'div_margin_train_oc.pkl')
