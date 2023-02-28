from transformers import pipeline
import pandas as pd
import fire

def main(lang):
    model_name = "./results_fine/{}/best_model".format(lang)
    fname = 'data/test_afrisenti_2023/{}_test_participants.tsv'.format(lang)
    pip = pipeline(task= 'text-classification', model= model_name )
    df = pd.read_csv(fname, sep='\t')
    preds = [pip(x)[0]['label'] for x in df.tweet]
    df['pred_label'] = preds
    del df['tweet']
    df.to_csv('submission/language_fine/pred_{}.tsv'.format(lang), sep='\t', index= False)



if __name__ == '__main__':
  fire.Fire(main)