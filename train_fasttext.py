"""
Sefamerve R&D Center

"""

import fasttext
import os
import fire

langs = ['ma', 'am', 'kr', 'pt', 'ts', 'yo', 'dz', 'pcm', 'twi', 'ig', 'ha', 'sw', 'multilingual' ]



def get_args(model):
    args = args = model.epoch , model.lr,model.dim, model.minCount, model.wordNgrams, model.maxn, model.minn,model.bucket,str(model.loss).split('.')[-1]
    return args


def main(lang):
    if lang in langs:
        train_file = "fdata/{}.train".format(lang)
        valid_file = "fdata/{}.valid".format(lang)
        full_file = "fdata/{}.full".format(lang)
        
        model_name_full = 'models/ftext/lang_{}_full'.format(lang)


        print("Language {} started ... \n".format(lang), train_file, valid_file, full_file)

        if not os.path.exists(model_name_full):

            print (' Autotuning .... ')
            model = fasttext.train_supervised(input= train_file, autotuneValidationFile= valid_file, autotuneDuration= 120)

            print (' Autotuning ended .... ')

            res = model.test(valid_file)
            
            model_name_valid = 'models/ftext/lang_{}_valid_{}'.format(lang,res[1])
            print( model_name_valid , res , "saving..... ")
            try:
                model.save_model(model_name_valid)
                print(model_name_valid , "saved.... ")
            except:
                print("Error model not saved ")


            print(lang,)
            args = get_args(model)

            epoch , lr, dim, minCount, wordNgrams, maxn, minn, bucket, loss = args

            print('Best Model  result {} with args : '.format(res), args)
            
            model = fasttext.train_supervised(input= full_file,
                                            epoch= epoch,
                                            lr= lr,
                                            dim= dim,
                                            minCount= minCount,
                                            wordNgrams= wordNgrams,
                                            minn= minn,
                                            maxn= maxn,
                                            bucket= bucket,
                                            loss= loss,
                                            )
        
            model.save_model(model_name_full)
        else:
            print("Language {} already trained ... \n".format(lang))

    else:
        print('Error languge code not valid')


if __name__ == '__main__':
  fire.Fire(main)