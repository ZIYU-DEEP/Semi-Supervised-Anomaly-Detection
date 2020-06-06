cd ..
python training.py -lb 2 -la 0 -ra 0.1 -op rec -ln cifar10 -le cifar10_eval -rt /net/leksai/data/CIFAR10 -nt cifar10_LeNet_rec -gpu 0
python training.py -lb 2 -la 1 -ra 0.1 -op rec -ln cifar10 -le cifar10_eval -rt /net/leksai/data/CIFAR10 -nt cifar10_LeNet_rec -gpu 0
python training.py -lb 2 -la 7 -ra 0.1 -op rec -ln cifar10 -le cifar10_eval -rt /net/leksai/data/CIFAR10 -nt cifar10_LeNet_rec -gpu 0
python training.py -lb 2 -la 8 -ra 0.1 -op rec -ln cifar10 -le cifar10_eval -rt /net/leksai/data/CIFAR10 -nt cifar10_LeNet_rec -gpu 0
python training.py -lb 2 -la 9 -ra 0.1 -op rec -ln cifar10 -le cifar10_eval -rt /net/leksai/data/CIFAR10 -nt cifar10_LeNet_rec -gpu 0
