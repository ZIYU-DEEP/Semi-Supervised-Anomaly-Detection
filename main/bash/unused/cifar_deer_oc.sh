cd ..
python training.py -lb 4 -la 0 -ra 0.1 -op one_class -ln cifar10 -le cifar10_eval -rt /net/leksai/data/CIFAR10 -nt cifar10_LeNet_one_class -gpu 3 -pt 1
python training.py -lb 4 -la 1 -ra 0.1 -op one_class -ln cifar10 -le cifar10_eval -rt /net/leksai/data/CIFAR10 -nt cifar10_LeNet_one_class -gpu 3 -pt 1
python training.py -lb 4 -la 7 -ra 0.1 -op one_class -ln cifar10 -le cifar10_eval -rt /net/leksai/data/CIFAR10 -nt cifar10_LeNet_one_class -gpu 3 -pt 1
python training.py -lb 4 -la 8 -ra 0.1 -op one_class -ln cifar10 -le cifar10_eval -rt /net/leksai/data/CIFAR10 -nt cifar10_LeNet_one_class -gpu 3 -pt 1
python training.py -lb 4 -la 9 -ra 0.1 -op one_class -ln cifar10 -le cifar10_eval -rt /net/leksai/data/CIFAR10 -nt cifar10_LeNet_one_class -gpu 3 -pt 1
