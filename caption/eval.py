import pandas as pd

PREDICTION={
    'brazil':'./brazil_image_caption_question_prediction_4o.csv',
    'israel':'./israel_image_caption_question_prediction_4o.csv',
    'japan':'./japan_image_caption_question_prediction_4o.csv',
    'spain':'./spain_image_caption_question_prediction_4o.csv'
}

GT = {
    'brazil':'./brazil_english_processed.tsv',
    'israel':'./israel_english_processed.tsv',
    'japan':'./japan_english_processed.tsv',
    'spain':'./spain_english_processed.tsv'
}

LANGUAGES = ['brazil', 'israel', 'japan', 'spain']

for language in LANGUAGES:
    print(language)
    df_p = pd.read_csv(PREDICTION[language])
    df_gt = pd.read_csv(GT[language], sep='\t')
    num_data = len(df_p)
    print(num_data)
    acc = 0
    for i in range(num_data):
        if df_p['answer'][i][0]==df_gt.iloc[i,8]:
            acc+=1
    print('#Correct:',acc,'  Accuracy:',acc/num_data)
    