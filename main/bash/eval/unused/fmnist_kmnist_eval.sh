cd ../..

python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_rec -op rec_unsupervised -lb 0 -ra 0
python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_rec -op rec -lb 0 -la 6 -ra 0.1 -pt 1
python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_rec -op rec -lb 0 -la 7 -ra 0.1 -pt 1

python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_one_class -op one_class_unsupervised -lb 0 -ra 0
python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_one_class -op one_class -lb 0 -la 6 -ra 0.1 -pt 1
python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_one_class -op one_class -lb 0 -la 7 -ra 0.1 -pt 1


python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_rec -op rec_unsupervised -lb 2 -ra 0
python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_rec -op rec -lb 2 -la 4 -ra 0.1 -pt 1
python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_rec -op rec -lb 2 -la 7 -ra 0.1 -pt 1

python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_one_class -op one_class_unsupervised -lb 2 -ra 0
python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_one_class -op one_class -lb 2 -la 4 -ra 0.1 -pt 1
python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_one_class -op one_class -lb 2 -la 7 -ra 0.1 -pt 1


python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_rec -op rec_unsupervised -lb 4 -ra 0
python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_rec -op rec -lb 4 -la 2 -ra 0.1 -pt 1
python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_rec -op rec -lb 4 -la 5 -ra 0.1 -pt 1

python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_one_class -op one_class_unsupervised -lb 4 -ra 0
python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_one_class -op one_class -lb 4 -la 2 -ra 0.1 -pt 1
python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_one_class -op one_class -lb 4 -la 5 -ra 0.1 -pt 1


python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_rec -op rec_unsupervised -lb 6 -ra 0
python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_rec -op rec -lb 6 -la 2 -ra 0.1 -pt 1
python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_rec -op rec -lb 6 -la 7 -ra 0.1 -pt 1

python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_one_class -op one_class_unsupervised -lb 6 -ra 0
python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_one_class -op one_class -lb 6 -la 2 -ra 0.1 -pt 1
python evaluation.py -ln fmnist -le kmnist_eval -rt /net/leksai/data/KMNIST -nt fmnist_LeNet_one_class -op one_class -lb 6 -la 7 -ra 0.1 -pt 1
