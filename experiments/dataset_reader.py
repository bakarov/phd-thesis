from pandas import read_csv, DataFrame
from os import path, mkdir

def parse_xnli(xnli_path: str='xnli/xnli.dev.tsv', save_directory: str='xnli_parsed'):
    xnli = read_csv(xnli_path, sep='\t')
    langs = list(xnli.language.unique())
    langs.remove('en')
    for lang in langs:
        first_id = 1
        last_id = 2007
        source_lang = 'en'
        target_lang = lang
        dataset = []
        errors = []
        for pair_id in range(first_id, last_id):
            pair_chunk = xnli.loc[xnli['pairID'] == pair_id]
            try:
                source_lang_premise = pair_chunk.loc[pair_chunk['language'] == source_lang]['sentence1'].values[0]
                target_lang_hypothesis = pair_chunk.loc[pair_chunk['language'] == target_lang]['sentence2'].values[0]
                label = pair_chunk.iloc[0]['gold_label']
                dataset.append([source_lang_premise, target_lang_hypothesis, label])
            except IndexError:
                errors.append(pair_id)
        dataset = DataFrame(dataset)
        dataset.columns = ['premise', 'hypothesis', 'label']
        while(True):
            try:
                dataset.to_csv(path.join(save_directory, 'xnli_en_{}.tsv'.format(target_lang)), sep='\t', index=False)
                break
            except FileNotFoundError:
                mkdir(save_directory)
                
def parse_sts2017():
    source_languages_with_track = [
        ('2', 'ar', ['none']), 
        ('6', 'tr', ['none']),
        ('4', 'es', ['a', 'b'])
    ]

    for track, source_language, tasks in source_languages_with_track:
        sts_data = []
        for task in tasks:
            if source_language == 'es':
                task_text = read_csv('sts2017/STS.input.track4{}.{}-en.txt'.format(task, source_language), sep='\t', names=['sentence_{}'.format(source_language), 'sentence_en'])
                task_labels = read_csv('sts2017/STS.gs.track4{}.{}-en.txt'.format(task, source_language), sep='\t', names=['similarity'])
            else:
                task_text = read_csv('sts2017/STS.input.track{}.{}-en.txt'.format(track, source_language), sep='\t', names=['sentence_{}'.format(source_language), 'sentence_en'])[:250]
                if source_language == 'tr':
                    task_text = task_text[:250]
                task_labels = read_csv('sts2017/STS.gs.track{}.{}-en.txt'.format(track, source_language), sep='\t', names=['similarity'])
            for row_id, row_values in task_text.iterrows():
                sts_data.append([row_values['sentence_en'], row_values['sentence_{}'.format(source_language)], task_labels.iloc[row_id].values[0], task])  
        sts_data = DataFrame(sts_data)
        sts_data.columns = ['sentence_{}'.format(source_language), 'sentence_en', 'similarity', 'task']
        sts_data.to_csv('sts_2017_{}_en.tsv'.format(source_language), sep='\t', index=False)
