python training.py -lb 4 -ra 0. -op rec_unsupervised -ln cifar10 -le cifar10_eval -rt /net/leksai/data/CIFAR10 -nt cifar10_LeNet_rec -gpu 0

python training.py -lb 7 -ra 0. -ln cifar10 -le cifar10_eval -rt /net/leksai/data/CIFAR10 -nt cifar10_LeNet_one_class -gpu 0 -pt 1

python training.py -lb 7 -ra 0. -ln cifar10 -le cifar10_eval -rt /net/leksai/data/CIFAR10 -nt cifar10_LeNet_one_class -gpu 0 -pt 0
