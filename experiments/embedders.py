import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
from pandas import DataFrame
from gensim.models import KeyedVectors
from os import path
from utils import get_vector


def get_use_vectors(dataset: DataFrame, model_path: str = '../use/use-multilingual/'):
    use_embeddings = hub.load(model_path)
    source_vecs = use_embeddings(dataset['sentence1'].values)['outputs'].numpy()
    target_vecs = use_embeddings(dataset['sentence2'].values)['outputs'].numpy()
    return source_vecs, target_vecs, dataset['label'].values


def get_sbert_vectors(dataset: DataFrame,
                      model_path: str = '../sentence-bert/models/training_stsbenchmark_bert-2019-11-26_14-23-44/'):
    embedder = SentenceTransformer(model_path)
    source_vecs = embedder.encode(dataset['sentence1'].values)
    target_vecs = embedder.encode(dataset['sentence2'].values)
    return source_vecs, target_vecs, dataset['label'].values


def get_muse_vectors(dataset: DataFrame, model_path: str = '../muse-aligned/', source_model: str = 'wiki.multi.en.vec',
                     target_model: str = 'wiki.multi.ru.vec'):
    source_model = KeyedVectors.load_word2vec_format(path.join(model_path, source_model), encoding='utf-8')
    target_model = KeyedVectors.load_word2vec_format(path.join(model_path, target_model))

    def convert_data_to_vectors(data, source_model, target_model):
        source_vecs = []
        target_vecs = []
        labels = []
        for row_id, row in data.iterrows():
            source_vecs.append(get_vector(row['sentence1'].lower().split(), source_model))
            target_vecs.append(get_vector(row['sentence2'].lower().split(), target_model))
            labels.append(row['label'])
        return source_vecs, target_vecs, labels

    return convert_data_to_vectors(dataset, source_model, target_model)
