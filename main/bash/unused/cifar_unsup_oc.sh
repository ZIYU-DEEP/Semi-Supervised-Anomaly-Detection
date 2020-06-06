cd ..
python training.py -lb 0 -ra 0. -ln cifar10 -le cifar10_eval -rt /net/leksai/data/CIFAR10 -nt cifar10_LeNet_one_class -gpu 2 -pt 1
python training.py -lb 1 -ra 0. -ln cifar10 -le cifar10_eval -rt /net/leksai/data/CIFAR10 -nt cifar10_LeNet_one_class -gpu 2 -pt 1
python training.py -lb 2 -ra 0. -ln cifar10 -le cifar10_eval -rt /net/leksai/data/CIFAR10 -nt cifar10_LeNet_one_class -gpu 2 -pt 1
python training.py -lb 3 -ra 0. -ln cifar10 -le cifar10_eval -rt /net/leksai/data/CIFAR10 -nt cifar10_LeNet_one_class -gpu 2 -pt 1
python training.py -lb 4 -ra 0. -ln cifar10 -le cifar10_eval -rt /net/leksai/data/CIFAR10 -nt cifar10_LeNet_one_class -gpu 2 -pt 1
