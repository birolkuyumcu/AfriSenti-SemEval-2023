

![](afrisenti.png)

# AfriSenti-SemEval-2023

This repository contains code for the SemEval 2023 Shared Task 12: Sentiment Analysis in African Languages (AfriSenti-SemEval). More information can be found at the: [shared task](https://afrisenti-semeval.github.io/) and [competition](https://codalab.lisn.upsaclay.fr/competitions/7320) websites.

In our system we utilized three different models; fastText, MultiLang Transformers, and Language-Specific Transformers to find the best working model for the classification challenge. We experimented with mentioned models and mostly  reached the best prediction scores using the Language Specific Transformers. Our best-submitted result was ranked 3rd among submissions for the Amharic language, obtaining an F1 score of 0.702 behind the second-ranked system.

language codes 

```
['ma', 'am', 'kr', 'pt', 'ts', 'yo', 'dz', 'pcm', 'twi', 'ig', 'ha', 'sw', 'multilingual' ]
```

Researchers interested in reproducing or building upon our work  can access the code and
follow steps.

1. Download Data from [competition](https://codalab.lisn.upsaclay.fr/competitions/7320) websites and  extract to data directory

2. Run prepare data codes 

   ```bash
   python data_prepare_fasttext.py language_code
   python data_prepare_transformers.py
   ```

3.  Run Training codes

   ```bash
   # fastText
   python train_fasttext.py language_code
   # MultiLang - Afriberta - Transformer
   python train_afriberta.py language_code
   # Language-Specific Transformers
   python train_language_fine.py language_code
   ```

   

4. Run Submission codes

   ```bash
   # fastText
   python fasttext_submission.py language_code
   # MultiLang - Afriberta - Transformer
   python afriberta_submission.py language_code
   # Language-Specific Transformers
   python language_fine_submission.py language_code
   ```

   

