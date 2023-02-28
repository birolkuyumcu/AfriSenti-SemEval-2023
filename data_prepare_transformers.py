"""
Sefamerve R&D Center

"""

import pandas as pd

langs = ['ma', 'am', 'kr', 'pt', 'ts', 'yo', 'dz', 'pcm', 'twi', 'ig', 'ha', 'sw', 'multilingual' ]

for lang in langs:
    if lang == 'multilingual':
        train = 'data/public_data/TaskB/multilingual_train.tsv'
    else:
        train = 'data/public_data/TaskA/train/{}_train.tsv'.format(lang)
    new_train = 'data/new_data/{}_train.tsv'.format(lang)
    dev = 'data/dev_with_labels_afrisenti_2023/{}_dev_gold_label.tsv'.format(lang)
    df_t = pd.read_csv(train, sep='\t')
    df_d = pd.read_csv(dev, sep='\t')
    df = pd.concat([df_t,df_d])
    df_t = pd.read_csv(train, sep='\t')
    df.to_csv(new_train, index=False, sep='\t')