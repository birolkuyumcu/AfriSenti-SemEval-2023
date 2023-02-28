import fasttext
import pandas as pd 
import fire


langs = ['ma', 'am', 'kr', 'pt', 'ts', 'yo', 'dz', 'pcm', 'twi', 'ig', 'ha', 'sw','multilingual']


def main(lang):
    if lang in langs:
        try:
            print('Fastext Submission for : ',lang)
            fname = 'data/test_afrisenti_2023/{}_test_participants.tsv'.format(lang)
            mname = 'models/ftext/lang_{}_full'.format(lang)
            oname = "submission/fasttext/pred_{}.tsv".format(lang)
            df = pd.read_csv(fname, sep = '\t')
            print(fname, ' ', len(df), df.columns)

            print( 'Loading model : ', mname)
            model = fasttext.load_model(mname)
            preds = [model.predict(x) for x in df.tweet]
            preds = [x[0][0].split('_')[-1].lower() for x in preds]
            df['pred_label'] = preds
            del df['tweet']
            print('Resut counts : ',{x:preds.count(x) for x in set(preds)})
            df.to_csv(oname,sep='\t', index=False)
            print(oname , ' saved ...')

        except Exception as e:
            print('Error ',e ,lang)
    else:
        print('Error languge code not valid')


if __name__ == '__main__':
  fire.Fire(main)
