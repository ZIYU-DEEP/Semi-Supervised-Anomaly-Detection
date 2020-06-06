cd ..
python training.py -lb 1 -la 0 -ra 0.1 -op rec -ln kmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt kmnist_LeNet_rec -gpu 0
python training.py -lb 1 -la 2 -ra 0.1 -op rec -ln kmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt kmnist_LeNet_rec -gpu 0
python training.py -lb 9 -la 2 -ra 0.1 -op rec -ln kmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt kmnist_LeNet_rec -gpu 0
python training.py -lb 9 -la 3 -ra 0.1 -op rec -ln kmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt kmnist_LeNet_rec -gpu 0
python training.py -lb 2 -la 1 -ra 0.1 -op rec -ln kmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt kmnist_LeNet_rec -gpu 0
python training.py -lb 2 -la 0 -ra 0.1 -op rec -ln kmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt kmnist_LeNet_rec -gpu 0
