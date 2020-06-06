cd ..
python training.py -lb 5 -la 3 -ra 0.1 -op rec -ln kmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt kmnist_LeNet_rec -gpu 0
python training.py -lb 5 -la 6 -ra 0.1 -op rec -ln kmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt kmnist_LeNet_rec -gpu 0
python training.py -lb 6 -la 2 -ra 0.1 -op rec -ln kmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt kmnist_LeNet_rec -gpu 0
python training.py -lb 6 -la 0 -ra 0.1 -op rec -ln kmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt kmnist_LeNet_rec -gpu 0
python training.py -lb 7 -la 5 -ra 0.1 -op rec -ln kmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt kmnist_LeNet_rec -gpu 0
python training.py -lb 7 -la 1 -ra 0.1 -op rec -ln kmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt kmnist_LeNet_rec -gpu 0
python training.py -lb 8 -la 0 -ra 0.1 -op rec -ln kmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt kmnist_LeNet_rec -gpu 0
python training.py -lb 8 -la 1 -ra 0.1 -op rec -ln kmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt kmnist_LeNet_rec -gpu 0
