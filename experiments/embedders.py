import tensorflow_hub as hub
import tensorflow_text
from sentence_transformers import SentenceTransformer
from pandas import DataFrame
from gensim.models import KeyedVectors
from os import path
from utils import get_vector
from typing import List


def get_use_vectors(source_texts: List, target_texts: List, model_path: str):
    use_embeddings = hub.load(model_path)
    source_vecs = use_embeddings(source_texts)['outputs'].numpy()
    target_vecs = use_embeddings(target_texts)['outputs'].numpy()
    return source_vecs, target_vecs


def get_sbert_vectors(source_texts: List, target_texts: List, model_path: str):
    embedder = SentenceTransformer(model_path)
    source_vecs = embedder.encode(source_texts)
    target_vecs = embedder.encode(target_texts)
    return source_vecs, target_vecs


def get_muse_vectors(source_texts: List, target_texts: List, source_model_path: str, target_model_path: str):
    source_model = KeyedVectors.load_word2vec_format(source_model_path, encoding='utf-8')
    target_model = KeyedVectors.load_word2vec_format(target_model_path, encoding='utf-8')
    source_vecs = [get_vector(text.lower().split(), source_model) for text in source_texts]
    target_vecs = [get_vector(text.lower().split(), target_model) for text in target_texts]
    return source_vecs, target_vecs
