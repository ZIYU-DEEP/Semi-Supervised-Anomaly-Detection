cd ..
python training.py -lb 0 -la 5 -ra 0.1 -op one_class -ln kmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt kmnist_LeNet_one_class -gpu 1 -pt 1
python training.py -lb 0 -la 6 -ra 0.1 -op one_class -ln kmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt kmnist_LeNet_one_class -gpu 1 -pt 1
python training.py -lb 3 -la 5 -ra 0.1 -op one_class -ln kmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt kmnist_LeNet_one_class -gpu 1 -pt 1
python training.py -lb 3 -la 6 -ra 0.1 -op one_class -ln kmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt kmnist_LeNet_one_class -gpu 1 -pt 1
python training.py -lb 4 -la 9 -ra 0.1 -op one_class -ln kmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt kmnist_LeNet_one_class -gpu 1 -pt 1
python training.py -lb 4 -la 0 -ra 0.1 -op one_class -ln kmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt kmnist_LeNet_one_class -gpu 1 -pt 1
