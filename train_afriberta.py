"""
Sefamerve R&D Center

"""

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
from transformers import EvalPrediction
import torch
import fire

langs = ['ma', 'am', 'kr', 'pt', 'ts', 'yo', 'dz', 'pcm', 'twi', 'ig', 'ha', 'sw', 'multilingual' ]

LR = 2e-5
EPOCHS = 5
BATCH_SIZE = 8
MODEL = "castorini/afriberta_base"


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def metric_function(predictions, labels, threshold=0.5):
    y_pred = np.argmax(predictions, axis= 1)
    y_true = labels
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'f1': f1_macro_average,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = metric_function(predictions=preds,labels=p.label_ids)

    return result




def main(lang):

    if lang in langs:
        cats = ['negative', 'neutral','positive' ]
        id2label = {x:cats[x].strip() for x in range(len(cats))}
        label2id = {cats[x].strip():x for x in range(len(cats))}
        df = pd.read_csv('data/new_data/{}_train.tsv'.format(lang), sep='\t')
        df = df.sample(frac= 1.0, random_state= 123)
        print('Language : ',lang)
        print('Data Len : ',len(df))
        split_point = int(0.8*len(df))

        dataset_dict = dict()
        dataset_dict['train'] = dict()
        dataset_dict['train']['text'] = df.tweet.iloc[:split_point].tolist()
        dataset_dict['train']['labels'] = [label2id[x] for x in df.label.iloc[:split_point]]
        dataset_dict['val'] = dict()
        dataset_dict['val']['text'] = df.tweet.iloc[split_point:].tolist()
        dataset_dict['val']['labels'] = [label2id[x] for x in df.label.iloc[split_point:]]

        print('Train Len : ',len(dataset_dict['train']['text']))
        print('Validation Len : ',len(dataset_dict['val']['text']))

        tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

        train_encodings = tokenizer(dataset_dict['train']['text'], truncation=True, padding=True)
        val_encodings = tokenizer(dataset_dict['val']['text'], truncation=True, padding=True)


        train_dataset = MyDataset(train_encodings, dataset_dict['train']['labels'])
        val_dataset = MyDataset(val_encodings, dataset_dict['val']['labels'])

        lstep = int(len(dataset_dict['train']['text'])/ (BATCH_SIZE* 4))
        print( 'Logging step size : ',lstep)

        training_args = TrainingArguments(
            output_dir='./results/{}'.format(lang),                   # output directory
            num_train_epochs=EPOCHS,                  # total number of training epochs
            per_device_train_batch_size=BATCH_SIZE,   # batch size per device during training
            per_device_eval_batch_size=BATCH_SIZE,    # batch size for evaluation
            warmup_steps= lstep ,                         # number of warmup steps for learning rate scheduler
            weight_decay=0.01,                        # strength of weight decay
            logging_dir='./logs/{}'.format(lang),                     # directory for storing logs
            logging_steps= lstep ,                          # when to print log
            load_best_model_at_end=True,             # load or not best model at the end
            report_to = "tensorboard",
            evaluation_strategy = 'steps',
            save_steps = lstep,
            metric_for_best_model = 'f1',
            save_total_limit= 2

        )

        num_labels = len(cats)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)
        model.config.label2id = label2id
        model.config.id2label = id2label

        trainer = Trainer(
            model=model,                              # the instantiated 🤗 Transformers model to be trained
            args=training_args,                       # training arguments, defined above
            train_dataset=train_dataset,              # training dataset
            eval_dataset=val_dataset,                  # evaluation dataset
            compute_metrics=compute_metrics
        )


        trainer.train()

        trainer.save_model("./results/{}/best_model".format(lang))
        tokenizer.save_pretrained("./results/{}/best_model".format(lang))

        test_preds_raw, test_labels , _ = trainer.predict(val_dataset)
        test_preds = np.argmax(test_preds_raw, axis=-1)
        print(classification_report(test_labels, test_preds, digits=3))

        with open("./results/{}/best_model_results.txt".format(lang), 'wt') as fp:        
            fp.write('Language {} and model : {} \n'. format(lang, MODEL)) 
            fp.write(classification_report(test_labels, test_preds, digits=3))
    

if __name__ == '__main__':
  fire.Fire(main)
    

