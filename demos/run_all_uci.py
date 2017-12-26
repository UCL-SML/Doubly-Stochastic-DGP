import os

# datasets = ['boston', 'concrete', 'energy', 'kin8nm', 'naval', 'power', 'protein', 'wine_red']
datasets = ['concrete', 'energy', 'kin8nm', 'naval', 'power', 'protein']

Ls = [1, 2, 3]
splits = range(5)

for split in splits:
    for dataset in datasets:
        for L in Ls:
            os.system('python run_regression.py {} {} {}'.format(dataset, L, split))
