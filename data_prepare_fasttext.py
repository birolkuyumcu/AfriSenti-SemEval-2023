"""
Sefamerve R&D Center

"""

import glob
import pandas as pd
import random
import fire
import os


langs = ['ma', 'am', 'kr', 'pt', 'ts', 'yo', 'dz', 'pcm', 'twi', 'ig', 'ha', 'sw', 'multilingual' ]

extensions = ['train','valid','full']


def split_df(df,fraction = 0.8):
    sidx = df.index.tolist()
    tlen = int(len(sidx)*fraction)
    random.shuffle(sidx)
    df_train  = df.iloc[sidx[:tlen]]
    df_valid = df.iloc[sidx[tlen:]]    
    return df_train, df_valid 


def to_fasttext(df, file_name):
    df_out = pd.DataFrame({'label' : ['__label__{}'.format(x) for x in df.label], 'text':df.tweet})
    df_out.to_csv(file_name, sep='\t',index = False, header = None)   


def main(lang):
    if lang in langs:
        if lang == 'multilingual':
            train = 'data/public_data/TaskB/{}_train.tsv'.format(lang)
            dev = 'data/dev_with_labels_afrisenti_2023/{}_dev_gold_label.tsv'.format(lang)
        else:
            train = 'data/public_data/TaskA/train/{}_train.tsv'.format(lang)
            dev = 'data/dev_with_labels_afrisenti_2023/{}_dev_gold_label.tsv'.format(lang)

        df_t = pd.read_csv(train, sep='\t')
        df_d = pd.read_csv(dev, sep='\t')
        df = pd.concat([df_t,df_d])
        df_train, df_valid = split_df(df)

        for dtemp,ext in zip([df_train,df_valid,df],extensions):
            to_fasttext(dtemp,"fdata/{}.{}".format(lang,ext))
    else:
        print('Error languge code not valid')
    

if __name__ == '__main__':
  fire.Fire(main)