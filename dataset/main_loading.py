"""
Title: main_loading.py
Description: The loading functions.
Author: Lek'Sai Ye, University of Chicago
"""


from fmnist_loader import FashionMNISTLoader, FashionMNISTLoaderEval
from cifar10_loader import CIFAR10Loader, CIFAR10LoaderEval
from kmnist_loader import KMNISTLoader, KMNISTLoaderEval

# #########################################################################
# 1. Load Dataset in One Function
# #########################################################################
def load_dataset(loader_name: str='fmnist',
                 root: str='/net/leksai/data/FashionMNIST',
                 label_normal: tuple=(0,),
                 label_abnormal: tuple=(),
                 label_eval:tuple=(1,),
                 ratio_abnormal: float=1.,
                 test_eval: bool=False):

    # known_loaders = ('fmnist', 'fmnist_eval', 'cifar10', 'cifar10_eval')
    # assert loader_name in known_loaders

    if loader_name == 'fmnist':
        return FashionMNISTLoader(root,
                                  label_normal,
                                  label_abnormal,
                                  ratio_abnormal)

    if loader_name == 'fmnist_eval':
        return FashionMNISTLoaderEval(root,
                                      label_eval,
                                      test_eval)

    if loader_name == 'kmnist':
        return KMNISTLoader(root,
                            label_normal,
                            label_abnormal,
                            ratio_abnormal)

    if loader_name == 'kmnist_eval':
        return KMNISTLoaderEval(root,
                                label_eval,
                                test_eval)

    if loader_name == 'cifar10':
        return CIFAR10Loader(root,
                             label_normal,
                             label_abnormal,
                             ratio_abnormal)

    if loader_name == 'cifar10_eval':
        return CIFAR10LoaderEval(root,
                                 label_eval,
                                 test_eval)

    return None
